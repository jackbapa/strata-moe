"""Reproducible resident A3B matrix: fixed 256K advertised context, exact token counts.

Run one mode at a time so GPU 0 service and GPU 1 low-budget trials can be
scheduled independently. Each row uses a fresh process; startup is excluded
from request timings and recorded separately. Partial results are saved after
every completed trial.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import Tokenizer

MODEL = ROOT / "models/a3b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
MTP = ROOT / "models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0.gguf"
PROMPT = "Explain mixture of experts briefly."
SNAPSHOT = ROOT / "bench/results/2026-10-16-a3b-iq3s-k2-k3-ab/selected.snapshot.json"


def gpu_used(device: int) -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={device}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def command(mode: str) -> tuple[list[str], dict[str, str], int | None]:
    if mode == "full":
        cfg = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        return [str(ROOT / "build-a3b/bin/strata-moe.exe"), "--serve", *cfg["args"]], {
            **os.environ, **cfg["env"], "STRATA_A3B_MTP_PROPOSAL_TRACE": "0",
        }, 0
    if mode in {"16g-hybrid-mtp", "16g-hybrid-no-mtp"}:
        args = [str(ROOT / "build-a3b/bin/strata-moe.exe"), "--serve", "--model", str(MODEL),
                "--ctx", "262144", "--batch", "2048", "--device", "0", "--vram-mib", "16384",
                "--cache-reserve-mib", "640", "--threads", "16", "--kv-tiered", "--kv-tiered-f16-short",
                "--hybrid-experts", "--cpu-expert-pool", "--pcie-frac", "0.25",
                "--expert-staging-slots", "8", "--gpu-embedding", "--gpu-sampling", "--fused-moe"]
        if mode == "16g-hybrid-mtp":
            args += ["--mtp-model", str(MTP), "--mtp-draft", "2", "--mtp-ubatch", "512",
                     "--exact-verify", "--defer-mtp-catchup", "--native-attention-exact"]
        return args, os.environ.copy(), 0
    if mode == "2g":
        return [str(ROOT / "build-a3b/bin/strata-moe.exe"), "--serve", "--model", str(MODEL),
                "--ctx", "262144", "--batch", "32", "--device", "1", "--vram-mib", "2048",
                "--cache-reserve-mib", "128", "--cache-mib", "0", "--kv-tiered", "--kv-tier-low-vram"], os.environ.copy(), 1
    if mode == "6g":
        return [str(ROOT / "build-a3b/bin/strata-moe.exe"), "--serve", "--model", str(MODEL),
                "--ctx", "262144", "--batch", "128", "--device", "1", "--vram-mib", "6144",
                "--cache-reserve-mib", "640", "--cache-mib", "2048", "--kv-tiered", "--kv-tier-low-vram"], os.environ.copy(), 1
    return [str(ROOT / "build-a3b-cpu/bin/strata-moe.exe"), "--serve", "--model", str(MODEL),
            "--ctx", "262144", "--batch", "128", "--cpu", "--threads", "16", "--kv-tiered"], os.environ.copy(), None


def trial(mode: str, ids: list[int], output_tokens: int, number: int, folder: Path) -> dict:
    args, env, device = command(mode)
    gpu_before = gpu_used(device) if device is not None else None
    started = time.perf_counter()
    proc = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    stderr: list[str] = []
    gpu_samples: list[int] = []
    rss_samples: list[int] = []
    stop = threading.Event()

    def read_errors() -> None:
        assert proc.stderr
        stderr.extend(proc.stderr.readlines())

    def sample() -> None:
        watched = psutil.Process(proc.pid)
        while not stop.wait(0.2):
            if device is not None:
                value = gpu_used(device)
                if value is not None:
                    gpu_samples.append(value)
            try:
                rss_samples.append(watched.memory_info().rss)
            except psutil.NoSuchProcess:
                break

    threading.Thread(target=read_errors, daemon=True).start()
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    ready = sent = first = None
    count = 0
    output_ids: list[int] = []
    done = ""
    try:
        assert proc.stdin and proc.stdout
        for line in proc.stdout:
            if line.startswith("READY "):
                ready = time.perf_counter()
                proc.stdin.write(f"GEN {output_tokens} temperature=0 " + ",".join(map(str, ids)) + "\n")
                proc.stdin.flush()
                sent = time.perf_counter()
            elif line.startswith("T "):
                count += 1
                output_ids.append(int(line.split()[1]))
                if first is None:
                    first = time.perf_counter()
            elif line.startswith("DONE "):
                done = line.strip()
                break
            elif line.startswith("ERR "):
                raise RuntimeError(line.strip())
        if proc.poll() is None:
            try:
                proc.stdin.write("QUIT\n")
                proc.stdin.flush()
            except OSError:
                # An external stop can race the cooperative QUIT. The
                # completion checks below still reject that partial trial.
                pass
        proc.wait(timeout=30)
        if proc.returncode != 0 or not done or first is None or sent is None or ready is None:
            raise RuntimeError(f"incomplete: rc={proc.returncode} done={done!r}; {''.join(stderr)[-1000:]}")
        fields = done.split()
        if count != int(fields[1]):
            raise RuntimeError(f"token line count {count} differs from DONE {fields[1]}")
        prompt_ms, decode_ms = float(fields[3]), float(fields[4])
        result = {
            "mode": mode, "trial": number, "configured_context_tokens": 262144,
            "input_tokens": len(ids), "requested_output_tokens": output_tokens,
            "actual_output_tokens": int(fields[1]), "observed_token_lines": count,
            "output_ids_sha256": hashlib.sha256(",".join(map(str, output_ids)).encode("ascii")).hexdigest(),
            "startup_to_ready_s": round(ready - started, 3),
            "request_to_first_model_token_s": round(first - sent, 3),
            "engine_prefill_tokens_per_s": round(len(ids) / (prompt_ms / 1000), 2),
            "engine_output_tokens_per_s": round(int(fields[1]) / (decode_ms / 1000), 2),
            "engine_prompt_ms": prompt_ms, "engine_decode_ms": decode_ms,
            "gpu_whole_card_before_mib": gpu_before,
            "gpu_whole_card_peak_mib": max(gpu_samples) if gpu_samples else None,
            "engine_rss_peak_mib": round(max(rss_samples) / 2**20, 1) if rss_samples else None,
            "device": device, "args": args,
        }
        return result
    finally:
        stop.set()
        sampler.join(timeout=6)
        (folder / f"{mode}-{len(ids)}-{number:02d}.log").write_text("".join(stderr), encoding="utf-8")
        if proc.poll() is None:
            proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["cpu", "2g", "6g", "full", "16g-hybrid-mtp", "16g-hybrid-no-mtp"], required=True)
    parser.add_argument("--lengths", nargs="+", type=int, default=[7, 1024, 8192, 32768])
    parser.add_argument("--output", type=int, default=128)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--folder", type=Path, default=ROOT / "bench/results/2026-10-16-256k-context-matrix")
    parser.add_argument("--input-text", type=Path,
                        help="use tokens from a UTF-8 text file instead of the repeated short prompt")
    options = parser.parse_args()
    options.folder.mkdir(parents=True, exist_ok=True)
    source = options.input_text.read_text(encoding="utf-8") if options.input_text else PROMPT
    base = Tokenizer.from_gguf(MODEL).encode(source, parse_special=False)
    if not base:
        parser.error("input text contains no tokens")
    results_path = options.folder / f"{options.mode}.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []
    for length in options.lengths:
        if length < 1 or length + options.output > 262144:
            parser.error("input and requested output must fit configured 256K context")
        ids = (base * ((length + len(base) - 1) // len(base)))[:length]
        input_hash = hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()
        for number in range(1, options.trials + 1):
            prior = next((row for row in results if row["input_tokens"] == length and row["trial"] == number), None)
            if prior and (prior["requested_output_tokens"] != options.output or
                          prior.get("input_ids_sha256") not in (None, input_hash) or
                          (options.input_text and not prior.get("input_ids_sha256"))):
                parser.error(f"{results_path} has a different input/output for {length} tokens, trial {number}; use --folder")
            if prior:
                continue
            result = trial(options.mode, ids, options.output, number, options.folder)
            result["input_source"] = str(options.input_text) if options.input_text else "repeated short prompt"
            result["input_ids_sha256"] = input_hash
            results.append(result)
            results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
