"""Sample one running A3B engine and the GPUs during a benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import time

import psutil


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-pid", type=int, required=True)
    parser.add_argument("--benchmark-pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    engine = psutil.Process(args.engine_pid)
    benchmark = psutil.Process(args.benchmark_pid)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as output:
        while benchmark.is_running() and engine.is_running():
            if benchmark.status() == psutil.STATUS_ZOMBIE:
                break
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            )
            memory = engine.memory_info()
            record = {
                "wall_time": time.time(),
                "engine_pid": args.engine_pid,
                "engine_rss_bytes": memory.rss,
                "engine_peak_working_set_bytes": getattr(memory, "peak_wset", None),
                "system_memory_used_bytes": psutil.virtual_memory().used,
                "gpu": [dict(zip(("index", "memory_mib", "utilization_pct", "sm_mhz", "memory_mhz", "temperature_c", "power_w"),
                                 (float(re.match(r"[0-9.]+", field.strip()).group())
                                  for field in line.split(","))))
                        for line in gpu.stdout.splitlines() if line.strip()],
            }
            output.write(json.dumps(record) + "\n")
            output.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
