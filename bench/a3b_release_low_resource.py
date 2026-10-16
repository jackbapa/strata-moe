"""Resident, same-prompt CPU / 2 GiB A3B measurement for release notes."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/results/2026-10-16-release-low-resource"
MODEL = ROOT / "models/a3b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import Tokenizer

PROMPT = "Explain mixture of experts briefly."
OUTPUT_TOKENS = 128


def gpu_used() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--id=1", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
        return int(output.strip().splitlines()[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def trial(mode: str, ids: list[int], number: int) -> dict:
    exe = ROOT / ("build-a3b-cpu" if mode == "cpu" else "build-a3b") / "bin/strata-moe.exe"
    args = [str(exe), "--serve", "--model", str(MODEL), "--ctx", "1024", "--batch", "32"]
    if mode == "cpu":
        args += ["--cpu", "--threads", "16"]
    else:
        args += ["--device", "1", "--vram-mib", "2048", "--cache-reserve-mib", "128", "--cache-mib", "0"]
    baseline = gpu_used()
    system_before = psutil.virtual_memory().used
    start = time.perf_counter()
    proc = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, bufsize=1)
    stderr: list[str] = []
    samples: list[int] = []
    ram_samples: list[int] = []
    system_samples: list[int] = []
    stop = threading.Event()

    def errors() -> None:
        assert proc.stderr
        stderr.extend(proc.stderr.readlines())

    def monitor() -> None:
        watched = psutil.Process(proc.pid)
        while not stop.wait(.2):
            value = gpu_used()
            if value is not None:
                samples.append(value)
            try:
                ram_samples.append(watched.memory_info().rss)
                system_samples.append(psutil.virtual_memory().used)
            except psutil.NoSuchProcess:
                break

    threading.Thread(target=errors, daemon=True).start()
    sampler = threading.Thread(target=monitor, daemon=True)
    sampler.start()
    first: float | None = None
    ready: float | None = None
    sent: float | None = None
    done = ""
    count = 0
    try:
        assert proc.stdin and proc.stdout
        for line in proc.stdout:
            if line.startswith("READY "):
                ready = time.perf_counter()
                proc.stdin.write(f"GEN {OUTPUT_TOKENS} temperature=0 " + ",".join(map(str, ids)) + "\n")
                proc.stdin.flush()
                sent = time.perf_counter()
            elif line.startswith("T "):
                count += 1
                if first is None:
                    first = time.perf_counter()
            elif line.startswith("DONE "):
                done = line.strip()
                break
            elif line.startswith("ERR "):
                raise RuntimeError(line.strip())
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.wait(timeout=10)
        if proc.returncode != 0 or not done or first is None or sent is None or ready is None:
            raise RuntimeError(f"incomplete run: rc={proc.returncode}, done={done!r}, stderr={''.join(stderr)[-600:]}")
        parts = done.split()
        result = {
            "mode": mode, "trial": number, "prompt": PROMPT, "prompt_tokens": len(ids),
            "completion_tokens": int(parts[1]), "startup_to_ready_s": round(ready - start, 3),
            "request_to_first_model_token_s": round(first - sent, 3),
            "engine_prompt_ms": float(parts[3]), "engine_decode_ms": float(parts[4]),
            "engine_output_tokens_per_s": round(int(parts[1]) / (float(parts[4]) / 1000), 2),
            "gpu1_whole_card_before_mib": baseline,
            "gpu1_whole_card_observed_peak_mib": max(samples) if samples else None,
            "gpu1_whole_card_peak_over_before_mib": max(samples) - baseline if samples and baseline is not None else None,
            "engine_process_rss_observed_peak_mib": round(max(ram_samples) / 2**20, 1) if ram_samples else None,
            "system_used_before_gib": round(system_before / 2**30, 3),
            "system_used_observed_peak_gib": round(max(system_samples) / 2**30, 3) if system_samples else None,
            "device": 1 if mode != "cpu" else None,
            "args": args,
        }
        log = OUT / f"{mode}-{number:02d}.txt"
        log.write_text("".join(stderr), encoding="utf-8")
        return result
    finally:
        stop.set()
        sampler.join(timeout=6)
        if proc.poll() is None:
            proc.kill()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_gguf(MODEL)
    ids = tok.encode(PROMPT, parse_special=False)
    results = []
    for mode in ("2g", "cpu"):
        for number in range(1, 4):
            result = trial(mode, ids, number)
            print(result, flush=True)
            results.append(result)
    (OUT / "resident-results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
