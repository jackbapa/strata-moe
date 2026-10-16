"""Same-GGUF CPU / GPU-trunk / cached-expert / optional MTP comparison."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "cpu-reference": ["--cpu"],
    "gpu-trunk-cpu-experts": ["--cache-mib", "0"],
    "gpu-cache-16g-budget": [],
    "gpu-cache-16g-mtp": [],
    "gpu-cache-full-mtp": [],
    "gpu-native": ["--native-gpu"],
    "gpu-native-mtp": ["--native-gpu"],
    "gpu-cache-gpu-only": ["--gpu-only-experts"],
    "gpu-cache-gpu-only-mtp": ["--gpu-only-experts"],
}

def gpu_used(device):
    result = subprocess.run(["nvidia-smi", f"--id={device}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    return int(result.stdout.strip())

def adaptive_counters(log):
    """Read per-K completed-window counts and accumulated round wall time."""
    match = re.search(r"^A3B MTP adaptive:([^\r\n]*)", log, re.MULTILINE)
    if not match:
        return {}
    counts, milliseconds, mean = {}, {}, {}
    for item in match[1].split():
        entry = re.fullmatch(r"K([0-8])=(\d+)/([\d]+(?:\.[\d]+)?)ms", item)
        if not entry or entry[1] in counts:
            raise ValueError("invalid or duplicate adaptive MTP counter: " + item)
        key, count, elapsed = entry[1], int(entry[2]), float(entry[3])
        counts[key], milliseconds[key] = count, elapsed
        mean[key] = elapsed / count if count else None
    if not counts or list(counts) != [str(k) for k in range(len(counts))]:
        raise ValueError("adaptive MTP counters must list consecutive K0..max in order")
    return {"mtp_adaptive_windows_by_k": counts, "mtp_adaptive_round_ms_by_k": milliseconds,
            "mtp_adaptive_mean_round_ms_by_k": mean}

def cache_decay_counters(log):
    match = re.search(r"A3B cache decay: prefill (\d+);? decode (\d+)", log)
    return {"prefill_cache_decay_sweeps": int(match[1]), "decode_cache_decay_sweeps": int(match[2])} if match else {}

def mtp_fused_counters(log):
    """Fused time is separate from the existing draft/catch-up phase totals."""
    match = re.search(r"^A3B MTP fused rounds: ([^\r\n]*)", log, re.MULTILINE)
    if not match:
        return {}
    values = re.fullmatch(r"calls (\d+); accepted catch-up rows (\d+); fused (\d+(?:\.\d+)?) ms \(excluded from draft/catchup\)", match[1])
    if not values:
        raise ValueError("invalid fused MTP counters")
    return {"mtp_fused_rounds": int(values[1]), "mtp_fused_catchup_rows": int(values[2]),
            "mtp_fused_round_s": float(values[3]) / 1000}

def prefill_share_counters(log):
    match = re.search(r"^A3B prefill CPU share: ([^\r\n]*)", log, re.MULTILINE)
    if not match:
        return {}
    values = re.fullmatch(r"calls (\d+); CPU jobs (\d+) experts (\d+)", match[1])
    if not values:
        raise ValueError("invalid prefill CPU-share counters")
    return {"prefill_share_calls": int(values[1]), "prefill_share_cpu_jobs": int(values[2]),
            "prefill_share_cpu_experts": int(values[3])}

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--predict", type=int, default=96)
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--exe", type=Path, default=ROOT / "build-a3b/bin" / ("strata-moe.exe" if os.name == "nt" else "strata-moe"))
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS)[:3])
    ap.add_argument("--repeat", type=int, default=1, help="Separate fresh-process trials per arm; not warm requests")
    ap.add_argument("--mtp-model", type=Path, help="MTP weights, required by every explicit *-mtp arm")
    ap.add_argument("--mtp-draft", type=int, default=3)
    ap.add_argument("--extra-arg", action="append", default=[], help="Append engine argument to every arm, e.g. --extra-arg=--mmap")
    ap.add_argument("--prompt-file", type=Path, help="Already formatted prompt; otherwise use the fixed Chinese prompt")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--vram-mib", type=int, default=16384, help="0 uses all available VRAM")
    ap.add_argument("--no-logits", action="store_true", help="performance run without a first-logit dump; permits target GPU sampling")
    args = ap.parse_args()
    if args.repeat < 1 or args.mtp_draft < 1:
        ap.error("--repeat and --mtp-draft must be positive")
    if any(name.endswith("-mtp") for name in args.arms) and not args.mtp_model:
        ap.error("MTP benchmark arms require --mtp-model")
    args.out.mkdir(parents=True, exist_ok=True)
    prompt = args.out / "prompt.txt"
    prompt.write_text(args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else
                      "<|im_start|>user\n请用中文简要解释混合专家模型如何工作。\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", encoding="utf-8")
    results = []
    reference = reference_name = reference_output = None
    runs = [(arm, repeat) for arm in args.arms for repeat in range(1, args.repeat + 1)]
    for name, repeat in runs:
        trial = name if args.repeat == 1 else f"{name}-repeat{repeat:02d}"
        extra = list(ARMS[name])
        if name.endswith("-mtp"):
            extra += ["--mtp-model", str(args.mtp_model.resolve()), "--mtp-draft", str(args.mtp_draft)]
        logits = args.out / f"{trial}.f32"
        command = [str(args.exe.resolve()), "--model", str(args.model.resolve()), "--raw", "--prompt-file", str(prompt.resolve()),
                   "--predict", str(args.predict), "--threads", str(args.threads), "--device", str(args.device),
                   "--vram-mib", str(args.vram_mib), "--ctx", str(args.ctx), "--batch", str(args.batch),
                   *([] if args.no_logits else ["--logits", str(logits.resolve())]), *extra, *args.extra_arg]
        base = gpu_used(args.device); peak = base
        exe_sha256 = hashlib.sha256(args.exe.read_bytes()).hexdigest()
        available_before = available_low = psutil.virtual_memory().available
        host_rss_peak = host_private_peak = host_private_commit_peak = 0
        host_samples = 0
        started = time.monotonic()
        with (args.out / f"{trial}.stdout.txt").open("wb") as out, (args.out / f"{trial}.stderr.txt").open("wb") as err:
            proc = subprocess.Popen(command, stdout=out, stderr=err)
            observed = psutil.Process(proc.pid)
            next_gpu = 0
            while proc.poll() is None:
                try:
                    mem = observed.memory_info()
                    host_rss_peak = max(host_rss_peak, mem.rss, getattr(mem, "peak_wset", 0))
                    host_private_peak = max(host_private_peak, getattr(mem, "private", 0))
                    host_private_commit_peak = max(host_private_commit_peak, getattr(mem, "peak_pagefile", 0))
                    host_samples += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                available_low = min(available_low, psutil.virtual_memory().available)
                if time.monotonic() >= next_gpu:
                    peak = max(peak, gpu_used(args.device))
                    next_gpu = time.monotonic() + 0.5
                time.sleep(0.1)
        if proc.returncode: raise RuntimeError(f"{name} failed; see its stderr file")
        log = (args.out / f"{trial}.stderr.txt").read_text(encoding="utf-8", errors="replace")
        match = re.search(r"A3B: prompt (\d+) tokens ([\d.]+)s ([\d.]+) tok/s; output (\d+) tokens ([\d.]+)s ([\d.]+) tok/s", log)
        if not match: raise RuntimeError("missing engine performance counters")
        result = {"name": name, "repeat": repeat, "trial": trial, "command": command, "exe_sha256": exe_sha256,
                  "environment": {key: os.environ[key] for key in
                                  ("STRATA_A3B_ARGMAX_MULTI", "STRATA_A3B_DENSE_Q8_REUSE", "STRATA_A3B_EXPERT_Q8_REUSE",
                                   "STRATA_A3B_NATIVE_EXPERT_REUSE", "STRATA_A3B_ATTENTION_MASK_REUSE",
                                   "CUDA_VISIBLE_DEVICES", "GGML_CUDA_PDL", "GGML_CUDA_DISABLE_GRAPHS")
                                  if key in os.environ},
                  "wall_s": time.monotonic() - started, "gpu_baseline_mib": base,
                  "gpu_peak_mib": peak, "gpu_increment_peak_mib": peak - base,
                  "host_rss_peak_bytes": host_rss_peak, "host_private_peak_bytes": host_private_peak,
                  "host_private_commit_peak_bytes": host_private_commit_peak,
                  "host_samples": host_samples, "host_sample_interval_s": 0.1,
                  "system_available_before_bytes": available_before, "system_available_low_bytes": available_low,
                  "system_available_after_bytes": psutil.virtual_memory().available,
                  "prompt_tokens": int(match[1]), "prefill_s": float(match[2]), "prefill_tok_s": float(match[3]),
                  "output_tokens": int(match[4]), "decode_s": float(match[5]), "decode_tok_s": float(match[6])}
        output = (args.out / f"{trial}.stdout.txt").read_bytes()
        result["output_sha256"] = hashlib.sha256(output).hexdigest()
        mtp = re.search(r"A3B MTP: windows (\d+) offered (\d+) accepted (\d+); draft ([\d.]+) ms catchup ([\d.]+) ms", log)
        if mtp:
            result["mtp_windows"] = int(mtp[1])
            result["mtp_offered_tokens"] = int(mtp[2])
            result["mtp_accepted_tokens"] = int(mtp[3])
            result["mtp_acceptance_fraction"] = int(mtp[3]) / max(1, int(mtp[2]))
            result["mtp_output_tokens_per_window"] = result["output_tokens"] / max(1, int(mtp[1]))
            result["mtp_draft_s"] = float(mtp[4]) / 1000
            result["mtp_catchup_s"] = float(mtp[5]) / 1000
        result.update(adaptive_counters(log))
        result.update(cache_decay_counters(log))
        result.update(mtp_fused_counters(log))
        result.update(prefill_share_counters(log))
        static = re.search(r"A3B static: calls (\d+); mapped host ([\d.]+) MiB; shared GPU scratch ([\d.]+) MiB; prefill route reads (\d+) decode route reads (\d+)", log)
        if static:
            result.update(gpu_static_calls=int(static[1]), gpu_static_mapped_host_mib=float(static[2]),
                          gpu_static_shared_scratch_mib=float(static[3]),
                          prefill_route_reads=int(static[4]), decode_route_reads=int(static[5]))
        hybrid = re.search(r"A3B hybrid: calls (\d+); PCIe fraction ([\d.]+); GPU miss rows (\d+) experts (\d+); input allocations (\d+) bytes (\d+) fallbacks (\d+)", log)
        if hybrid:
            result.update(hybrid_calls=int(hybrid[1]), pcie_fraction=float(hybrid[2]),
                          hybrid_gpu_miss_rows=int(hybrid[3]), hybrid_gpu_miss_experts=int(hybrid[4]),
                          hybrid_input_allocations=int(hybrid[5]), hybrid_input_bytes=int(hybrid[6]),
                          hybrid_input_fallbacks=int(hybrid[7]))
        pool = re.search(r"A3B CPU pool: calls (\d+) jobs (\d+) unique inputs (\d+) experts (\d+); input quant ([\d.]+) ms gate/up ([\d.]+) ms hidden quant ([\d.]+) ms down ([\d.]+) ms", log)
        if pool:
            result.update(cpu_pool_calls=int(pool[1]), cpu_pool_jobs=int(pool[2]),
                          cpu_pool_unique_inputs=int(pool[3]), cpu_pool_unique_experts=int(pool[4]),
                          cpu_pool_input_quant_ms=float(pool[5]), cpu_pool_gate_up_ms=float(pool[6]),
                          cpu_pool_hidden_quant_ms=float(pool[7]), cpu_pool_down_ms=float(pool[8]))
        doorbell = re.search(r"A3B doorbell: calls (\d+) CPU requests (\d+) callback ([\d.]+) ms; mailbox to host (\d+) bytes to GPU (\d+) bytes", log)
        if doorbell:
            result.update(doorbell_calls=int(doorbell[1]), doorbell_cpu_requests=int(doorbell[2]),
                          doorbell_callback_ms=float(doorbell[3]), doorbell_mailbox_to_host_bytes=int(doorbell[4]),
                          doorbell_mailbox_to_gpu_bytes=int(doorbell[5]))
        backend = re.search(r"A3B backend: route reads (\d+); native layout calls (\d+)(?:; single-column calls (\d+))?", log)
        if backend:
            result["backend_route_reads"] = int(backend[1])
            result["backend_native_layout_calls"] = int(backend[2])
            if backend[3] is not None:
                result["exact_verify_calls"] = int(backend[3])
        transfer = re.search(r"A3B transfer: prefill ([\d.]+) MiB; decode ([\d.]+) MiB; decode GPU rows (\d+) CPU rows (\d+); decode hit (\d+) miss (\d+)", log)
        if transfer:
            result.update(prefill_upload_mib=float(transfer[1]), decode_upload_mib=float(transfer[2]),
                          decode_gpu_rows=int(transfer[3]), decode_cpu_rows=int(transfer[4]),
                          decode_cache_hits=int(transfer[5]), decode_cache_misses=int(transfer[6]))
        if not args.no_logits:
            values = np.fromfile(logits, dtype="<f4").astype(np.float64)
            if values.size != 248320 or not np.isfinite(values).all(): raise RuntimeError("invalid logits")
            if reference is None:
                reference = values
                reference_name = name
                reference_output = output
            else:
                def log_softmax(x):
                    x = x - x.max()
                    return x - np.log(np.exp(x).sum())
                a, b = log_softmax(reference), log_softmax(values)
                result["first_token_reference_arm"] = reference_name
                result["output_matches_reference"] = output == reference_output
                result["first_token_kl_reference_to_arm"] = float(np.sum(np.exp(a) * (a - b)))
                result["first_token_top1_matches_reference"] = bool(reference.argmax() == values.argmax())
                if reference_name == "cpu-reference":
                    result["first_token_kl_cpu_to_arm"] = result["first_token_kl_reference_to_arm"]
                    result["first_token_top1_matches_cpu"] = result["first_token_top1_matches_reference"]
                result["first_token_logit_max_abs_error"] = float(np.abs(reference - values).max())
        results.append(result)
        (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False), flush=True)

if __name__ == "__main__": main()
