"""Twelve resident doorbell API requests with socket latency and RAM/VRAM evidence.

Never starts, reloads, unloads, or stops a model. Default order: identical BFS
and transaction prompts three times each, then 32/256 x3 and 2048/256 x3.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

import psutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from bench_a3b_latency import Client, prepare_prompt, stream_request, summarize_cases
from bench_a3b_supplemental import natural_case
from audit_helpers import extract_counters, latest_process_log, save_sse, validate_result

source_runner = HERE.parent / "2026-10-14-a3b-q8-global-api/heldout-run.py"
spec = importlib.util.spec_from_file_location("frozen_q8_heldout_prompts", source_runner)
frozen_prompts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen_prompts)
PROMPTS = frozen_prompts.PROMPTS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_engine(config, pid):
    expected = os.path.normcase(os.path.abspath(config["exe"]))
    candidates = []
    processes = [psutil.Process(pid)] if pid else psutil.process_iter(["pid", "exe", "cmdline"])
    for process in processes:
        try:
            command = process.cmdline()
            executable = os.path.normcase(os.path.abspath(process.exe()))
            if executable == expected and "--serve" in command and "--hybrid-doorbell" in command:
                candidates.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if len(candidates) != 1:
        raise RuntimeError("cannot uniquely identify the experimental service engine; supply --engine-pid for that process")
    return candidates[0]


def snapshot(process):
    query = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=10)
    gpu = []
    for line in query.stdout.strip().splitlines():
        index, name, used, total, utilization = [field.strip() for field in line.split(",")]
        gpu.append({"index": int(index), "name": name, "used_mib": int(used), "total_mib": int(total),
                    "utilization_pct": int(utilization)})
    return {"utc": datetime.now(timezone.utc).isoformat(), "monotonic_s": time.monotonic(),
            "pid": process.pid, "memory": process.memory_info()._asdict(),
            "system_memory": psutil.virtual_memory()._asdict(), "gpu": gpu}


class ResourceSampler:
    def __init__(self, process, directory, interval):
        self.process, self.directory, self.interval = process, directory, interval
        self.stop = threading.Event()
        self.samples, self.errors = [], []
        self.thread = None

    def collect(self):
        try:
            sample = snapshot(self.process)
            self.samples.append(sample)
            with (self.directory / "resource-samples.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(sample) + "\n")
            return sample
        except Exception as error:
            self.errors.append({"utc": datetime.now(timezone.utc).isoformat(), "error": str(error)})
            return None

    def start(self):
        initial = self.collect()
        if initial is None:
            raise RuntimeError("initial resource sample failed: " + self.errors[-1]["error"])
        (self.directory / "resources-before.json").write_text(json.dumps(initial, indent=2) + "\n", encoding="utf-8")
        def run():
            while not self.stop.wait(self.interval):
                self.collect()
        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=15)
        final = self.collect()
        if final:
            (self.directory / "resources-final.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
        summary = {"interval_s": self.interval, "samples": len(self.samples), "errors": self.errors,
                   "scope": "Periodic samples during the resident API test, including before/after. Sampled GPU maxima are not absolute instantaneous peaks. Windows peak_wset is OS process-lifetime physical working-set peak; private commitment is not RAM.",
                   "sampled_rss_max_bytes": max((s["memory"]["rss"] for s in self.samples), default=0),
                   "os_lifetime_working_set_peak_bytes": max((s["memory"].get("peak_wset", s["memory"]["rss"]) for s in self.samples), default=0),
                   "sampled_system_used_max_bytes": max((s["system_memory"]["used"] for s in self.samples), default=0),
                   "sampled_gpu_used_max_mib": {str(index): max((gpu["used_mib"] for s in self.samples for gpu in s["gpu"] if gpu["index"] == index), default=0)
                                                for index in {gpu["index"] for s in self.samples for gpu in s["gpu"]}}}
        (self.directory / "resource-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def summarize(results):
    report = {}
    for name in dict.fromkeys(row["case_id"] for row in results):
        report[name] = next(iter(summarize_cases([row for row in results if row["case_id"] == name]).values()))
    return report




def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8096")
    parser.add_argument("--config", type=Path, default=HERE / "config.json")
    parser.add_argument("--out", type=Path, default=HERE / "measurements")
    parser.add_argument("--phase", choices=["all", "heldout", "short", "long"], default="all")
    parser.add_argument("--engine-pid", type=int)
    parser.add_argument("--expected-exe-sha256", required=True, help="Required root-approved executable fingerprint")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--describe-only", action="store_true", help="No service/process/GPU access")
    args = parser.parse_args()
    if args.describe_only:
        print(json.dumps({"order": list(PROMPTS) + ["prompt32-output256", "prompt2048-output256"], "prompts": PROMPTS,
                          "repeats": 3, "total_requests": 12, "natural_max_tokens": 512, "short": [32, 256], "long": [2048, 256],
                          "temperature": 0, "seed": 12345, "resources": "1-second sampled RSS/VRAM; no model load/unload"}, ensure_ascii=False, indent=2))
        return
    if not 0.2 <= args.sample_interval <= 60:
        parser.error("sample interval must be .2..60 seconds")
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("use a new measurement directory; existing evidence is preserved")
    config_bytes = args.config.read_bytes()
    config = json.loads(config_bytes)
    required = {"--hybrid-doorbell", "--expert-profile", "--mtp-model"}
    if not required.issubset(config["args"]) or any(flag in config["args"] for flag in ("--gpu-only-experts", "--native-gpu", "--gpu-static-experts", "--gpu-dynamic-experts")):
        raise ValueError("runner requires the explicit doorbell/profile/MTP experiment config")
    for flag, default in (("--device", "0"), ("--pcie-frac", "0"), ("--cache-decay-interval", "0")):
        value = config["args"][config["args"].index(flag) + 1] if flag in config["args"] else default
        if float(value) != 0:
            raise ValueError(f"{flag} must be zero in the fixed-profile doorbell experiment")
    if config.get("idle_unload_s") != 0:
        raise ValueError("experiment requires resident model loading (idle_unload_s=0)")
    executable_sha = sha(config["exe"])
    if args.expected_exe_sha256 and executable_sha.lower() != args.expected_exe_sha256.lower():
        raise ValueError("doorbell executable hash differs from --expected-exe-sha256")
    mode = "hybrid-doorbell-fixed-profile-pcie0"
    client = Client(args.url, args.timeout)
    health = client.json("/health")
    if not health.get("loaded"):
        raise RuntimeError("service must already be ready; runner will not load it")
    model = client.json("/v1/models")["data"][0]["id"]
    process = find_engine(config, args.engine_pid)
    relevant_env = ("CUDA_VISIBLE_DEVICES", "STRATA_A3B_NATIVE_EXPERT_REUSE", "STRATA_A3B_EXPERT_Q8_REUSE",
                    "STRATA_A3B_DENSE_Q8_REUSE", "STRATA_A3B_ARGMAX_MULTI")
    engine_environment = process.environ()
    execution_env = {key: engine_environment.get(key) for key in relevant_env}
    cases = []
    if args.phase in ("all", "heldout"):
        cases += [natural_case(client, model, name, text, 512) for name, text in PROMPTS.items()]
    if args.phase in ("all", "short"):
        short = prepare_prompt(client, model, 32, False)
        short.update(case_id="prompt32-output256", max_tokens=256)
        cases.append(short)
    if args.phase in ("all", "long"):
        long = prepare_prompt(client, model, 2048, False)
        long.update(case_id="prompt2048-output256", max_tokens=256)
        cases.append(long)
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = Path(config["log"])
    initial_log = log_path.read_text(encoding="utf-8", errors="replace")
    latest = latest_process_log(initial_log)
    prior = len(re.findall(r"^A3B: prompt ", latest, re.MULTILINE))
    plan = {"model": model, "url": args.url, "phase": args.phase, "cases": cases, "repeats": 3,
            "temperature": 0, "seed": 12345, "thinking": False, "initial_health": health,
            "engine_pid": process.pid, "engine_create_time": process.create_time(), "engine_cmdline": process.cmdline(),
            "actual_exe_sha256": executable_sha, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "expected_exe_sha256": args.expected_exe_sha256, "engine_execution_env": execution_env,
            "runner_sha256": sha(__file__), "prompt_source_sha256": sha(source_runner),
            "latency_helper_sha256": sha(ROOT / "tools/bench_a3b_latency.py"),
            "audit_helpers_sha256": sha(HERE / "audit_helpers.py"),
            "server_source_sha256": sha(ROOT / "serve/server.py"),
            "timing_definitions": {"TTFB": "first underlying socket response read; may contain only headers",
                                   "TTFT": "first nonempty model content/reasoning SSE delta, excluding progress/role/comments",
                                   "engine_rates": "engine DONE clock/counts exported in final SSE timings",
                                   "client_output_rate": "(completion_tokens-1)/(last nonempty text delta-first nonempty text delta), subject to MTP bursts"},
            "completed_requests_in_process_before_run": prior,
            "execution_mode": mode,
            "cohort": f"Separate {mode} process; do not pool with frozen GPU-only Q8 or the earlier host-planned hybrid process.",
            "resource_scope": "Periodic sampling starts after model readiness; startup footprint/time read from engine log separately."}
    (args.out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out / "config.json").write_bytes(config_bytes)
    sampler = ResourceSampler(process, args.out, args.sample_interval)
    results = []
    sampler.start()
    try:
        for case_index, case in enumerate(cases):
            for repeat in range(1, 4):
                trial = f"{case['case_id']}-repeat{repeat:02d}"
                body = {"model": model, "messages": case["messages"], "max_tokens": case["max_tokens"],
                        "temperature": 0, "seed": 12345, "stream": True, "stream_options": {"include_usage": True},
                        "chat_template_kwargs": {"enable_thinking": False}}
                result = stream_request(client, body, args.out, trial)
                result.update(trial=trial, case_id=case["case_id"], repeat=repeat,
                              cache_state="fresh-service-first-measured-request" if prior == 0 and case_index == 0 and repeat == 1 else
                                          ("resident-case-first" if repeat == 1 else "resident-repeat"),
                              target_prompt_tokens=case["target_prompt_tokens"],
                              count_endpoint_prompt_tokens=case["count_endpoint_prompt_tokens"],
                              requested_max_tokens=case["max_tokens"], resident_process=True, temperature=0,
                              process_cohort=f"experimental-GPU-{mode}")
                result["count_endpoint_matches_actual_usage"] = result["usage"]["prompt_tokens"] == case["count_endpoint_prompt_tokens"]
                save_sse(args.out, trial)
                validate_result(result)
                results.append(result)
                (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                (args.out / "case-summary.json").write_text(json.dumps(summarize(results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print(json.dumps({key: value for key, value in result.items() if key != "rate_note"}, ensure_ascii=False), flush=True)
    finally:
        sampler.finish()
        log_bytes = log_path.read_bytes()
        (args.out / "engine.log").write_bytes(log_bytes)
    counters = extract_counters(log_bytes, results, prior)
    (args.out / "engine-counters.json").write_text(json.dumps(counters, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
