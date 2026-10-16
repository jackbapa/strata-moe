"""Measure fixed physical KV reservations with identical input and MTP settings."""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import re
import sys

import a3b_256k_matrix as matrix

sys.path.insert(0, str(matrix.ROOT / "tools"))
from setup_moe import config_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=int, nargs="+", default=[32768, 65536, 131072, 262144])
    parser.add_argument("--input", type=int, default=8192)
    parser.add_argument("--output", type=int, default=128)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--allocation", choices=["fixed", "tiered", "both"], default="fixed")
    parser.add_argument("--folder", type=Path, required=True)
    options = parser.parse_args()
    if options.trials < 1 or options.input < 1 or options.output < 1:
        parser.error("input, output and trials must be positive")
    if any(c not in (32768, 65536, 131072, 262144) or options.input + options.output > c
           for c in options.contexts):
        parser.error("select 32K/64K/128K/256K contexts that fit the request")
    options.folder.mkdir(parents=True, exist_ok=True)
    config = config_for("full", 0, 8096, Path("unused-tokenizer"))
    exe = matrix.ROOT / "build-a3b/bin/strata-moe.exe"
    engine_sha = hashlib.sha256(exe.read_bytes()).hexdigest()
    base = matrix.Tokenizer.from_gguf(matrix.MODEL).encode(matrix.PROMPT, parse_special=False)
    ids = (base * ((options.input + len(base) - 1) // len(base)))[:options.input]
    input_sha = hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest()
    results_path = options.folder / "results.json"
    if results_path.exists():
        parser.error("results.json already exists; use a new --folder")
    results = []
    allocations = ["tiered", "fixed"] if options.allocation == "both" else [options.allocation]
    for number in range(1, options.trials + 1):
        for capacity, allocation in product(options.contexts, allocations):
            args = list(config["args"])
            args[args.index("--ctx") + 1] = str(capacity)
            if allocation == "fixed" or capacity <= 65536:
                args = [a for a in args if a not in ("--kv-tiered", "--kv-tiered-f16-short")]
                if capacity > 131072:
                    args.append("--kv-q8")
            env = {**os.environ, **config["env"]}
            telemetry = []
            def gpu_used(device):
                import subprocess
                fields = "memory.used,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu"
                line = subprocess.check_output(
                    ["nvidia-smi", f"--id={device}", f"--query-gpu={fields}",
                     "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
                values = [float(value.strip()) for value in line.split(",")]
                telemetry.append(values)
                return int(values[0])
            matrix.gpu_used = gpu_used
            matrix.command = lambda mode: ([str(exe), "--serve", *args], env, 0)
            mode = f"{allocation}-{capacity}"
            row = matrix.trial(mode, ids, options.output, number, options.folder)
            log = (options.folder / f"{mode}-{len(ids)}-{number:02d}.log").read_text(encoding="utf-8")
            kv = re.findall(r"KV buffer size =\s*([\d.]+) MiB", log)
            cells = re.findall(r"llama_kv_cache: size = .*?\(\s*(\d+) cells", log)
            expected_cells = capacity if allocation == "fixed" else min(capacity, 65536)
            while expected_cells < options.input + options.output:
                expected_cells = min(capacity, expected_cells * 2)
            if not cells or any(int(c) != expected_cells for c in cells):
                raise RuntimeError(f"unexpected physical KV allocation: {cells}, expected {expected_cells}")
            row.update(configured_context_tokens=capacity, allocation=allocation,
                       physical_kv_cells=expected_cells, kv_buffer_mib=sum(map(float, kv)),
                       engine_sha256=engine_sha, input_ids_sha256=input_sha,
                       environment=config["env"], telemetry_fields=["memory_mib", "sm_mhz", "mem_mhz",
                                                                  "temperature_c", "power_w", "gpu_percent"],
                       telemetry=telemetry)
            results.append(row)
            results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({key: row[key] for key in ("allocation", "configured_context_tokens", "trial", "physical_kv_cells",
                             "kv_buffer_mib", "request_to_first_model_token_s", "engine_output_tokens_per_s",
                             "gpu_whole_card_peak_mib")}), flush=True)


if __name__ == "__main__":
    main()
