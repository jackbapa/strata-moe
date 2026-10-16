"""Run six alternating 50K/256 I32 scatter/compact A/B trials.

The arms share one binary and one I32 MTP GGUF; only the opt-in compact GPU
sampler flag differs. Requires completed real-model five-arm strict parity.
Always restores the selected 8096 service in finally after taking ownership.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "a3b" / "research" / "combined-exact"))
sys.path.insert(0, str(ROOT / "tools"))
import run_six_arm as lifecycle  # noqa: E402
from bench_a3b_ab import load_config, sha256_file  # noqa: E402
sys.path.insert(0, str(ROOT / "a3b" / "work" / "draft-compact"))
import compare_short_strict  # noqa: E402


CONFIGS = {arm: HERE / f"{arm}.snapshot.json" for arm in ("i32_scatter", "i32_compact")}
ORDER = ("i32_scatter", "i32_compact") * 3
SELECTED = ROOT / "strata-a3b-iq3s-strata-optimized.json"
STRICT = ROOT / "bench/results/2026-10-16-a3b-draft-compact-short-strict/strict"
STRICT_BINARY = ROOT / "build-a3b-draft-compact-cuda/bin/Release/strata-moe-speculative-test.exe"
EXPECTED = {
    "engine": "074122f906d5a46eb9f3bdf41239c32fdecf4ddbe27c86c1ee3972bdbf3c7cc3",
    "target": "66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66",
    "draft": "cd2a20b807b38dfa199b530809e3fc7f3811e992c0511c77fe1f766fb31cbf93",
}
METRICS = ("http_response_first_byte_s", "ttft_any_content_or_reasoning_s",
           "engine_prompt_tokens_per_s", "engine_predicted_tokens_per_s",
           "gpu0_total_used_mib_observed_peak", "gpu0_total_used_mib_observed_peak_over_before",
           "engine_rss_bytes_observed_peak", "mtp_compact_gpu_sample_rows",
           "mtp_offered_tokens", "mtp_accepted_tokens", "mtp_acceptance_rate")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strict_gate() -> dict:
    all_arms = compare_short_strict.compare_strict(False)
    if all_arms["same_binary_sha256"] != sha256_file(STRICT_BINARY):
        raise RuntimeError("five-arm strict fixture binary changed after its run")
    scatter = json.loads((STRICT / "i32_scatter/manifest.json").read_text(encoding="utf-8"))
    compact = json.loads((STRICT / "i32_compact/manifest.json").read_text(encoding="utf-8"))
    if (scatter["draft_sha256"] != EXPECTED["draft"] or
            compact["draft_sha256"] != EXPECTED["draft"] or
            len(scatter["proposal_traces"]) != 29 or
            len(scatter["state_sha256"]) != 16 or
            scatter["proposal_traces"] != compact["proposal_traces"] or
            scatter["state_sha256"] != compact["state_sha256"] or
            scatter["compact_gpu_sample_rows"] != 0 or
            compact["compact_gpu_sample_rows"] <= 0):
        raise RuntimeError("I32 compact/scatter short strict gate failed")
    return {**all_arms, "state_pairs_bitwise_equal": 16,
            "i32_compact_graph_rows_short": compact["compact_gpu_sample_rows"]}


def preflight() -> dict:
    configs = {arm: json.loads(path.read_text(encoding="utf-8")) for arm, path in CONFIGS.items()}
    scatter, compact = configs["i32_scatter"], configs["i32_compact"]
    ignore = {"log", "env"}
    if {k: v for k, v in scatter.items() if k not in ignore} != \
       {k: v for k, v in compact.items() if k not in ignore}:
        raise RuntimeError("configs differ outside sampler flag and log")
    env_a, env_b = scatter["env"].copy(), compact["env"].copy()
    if env_a.pop("STRATA_A3B_DRAFT_COMPACT_GPU", None) != "0" or \
       env_b.pop("STRATA_A3B_DRAFT_COMPACT_GPU", None) != "1" or \
       env_a != env_b or scatter["log"] == compact["log"]:
        raise RuntimeError("configs differ in unexpected environment settings")
    args = scatter["args"]
    if (args[args.index("--ctx") + 1] != "262144" or
            args[args.index("--mtp-draft") + 1] != "2" or
            "--strata-native-experts" not in args or
            env_a.get("STRATA_A3B_MTP_DEVICE_CHAIN_K2") == "1"):
        raise RuntimeError("configs lost selected 256K / IQ expert / ordinary K2 path")
    paths = {}
    hash_cache: dict[Path, str] = {}
    def file_hash(path: Path) -> str:
        key = path.resolve()
        if key not in hash_cache:
            hash_cache[key] = sha256_file(key)
        return hash_cache[key]
    for arm, path in CONFIGS.items():
        _, target, exe, _log, mtp = load_config(path)
        if not mtp:
            raise RuntimeError("missing MTP GGUF")
        paths[arm] = {"engine": file_hash(exe), "target": file_hash(target),
                      "mtp": file_hash(mtp), "mtp_path": str(mtp),
                      "config_sha256": sha256_file(path)}
        if (paths[arm]["engine"] != EXPECTED["engine"] or
                paths[arm]["target"] != EXPECTED["target"] or
                paths[arm]["mtp"] != EXPECTED["draft"]):
            raise RuntimeError(f"{arm} binary/model hash changed")
    return {"config_and_model": paths, "strict": strict_gate(), "order": ORDER,
            "request": {"prompt_tokens": 50000, "completion_tokens": 256,
                        "temperature": 0, "seed": 12345, "thinking": False},
            "selected_config_sha256": sha256_file(SELECTED),
            "only_execution_difference": "STRATA_A3B_DRAFT_COMPACT_GPU=0/1"}


def trial_command(suite: Path, arm: str) -> list[str]:
    return [sys.executable, str(ROOT / "tools" / "bench_a3b_ab.py"), "run",
            "--suite", str(suite / arm), "--arm", arm,
            "--config", str(CONFIGS[arm]), "--prompt-tokens", "50000",
            "--max-tokens", "256", "--repeats", "1", "--temperature", "0",
            "--seed", "12345", "--sample-interval", "0.5", "--timeout", "900"]


def run_trial(suite: Path, arm: str, index: int, environment: dict[str, str]) -> dict:
    arm_suite = suite / arm
    if arm == "i32_compact":
        template = suite / "i32_scatter" / "request-template.json"
        if not template.exists():
            raise RuntimeError("scatter arm did not establish the prompt")
        target = arm_suite / "request-template.json"
        if not target.exists():
            arm_suite.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(template, target)
        elif target.read_bytes() != template.read_bytes():
            raise RuntimeError("scatter/compact prompt templates differ")
    log = suite / "runner-logs" / f"{index:02d}-{arm}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = trial_command(suite, arm)
    started = time.monotonic()
    with log.open("wb") as output:
        done = subprocess.run(command, cwd=ROOT, env=environment, stdout=output,
                              stderr=subprocess.STDOUT, timeout=1800,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
    if done.returncode:
        raise RuntimeError(f"{arm} trial {index} failed; inspect {log}")
    files = sorted((arm_suite / "arms" / arm).glob("run[0-9][0-9][0-9].json"))
    if len(files) != (index + 1) // 2:
        raise RuntimeError(f"{arm} trial record count unexpected")
    return {"index": index, "arm": arm, "elapsed_s": round(time.monotonic() - started, 3),
            "record_path": str(files[-1]), "runner_log": str(log)}


def report(suite: Path, state: dict) -> dict:
    records = {arm: [json.loads(path.read_text(encoding="utf-8")) for path in sorted(
        (suite / arm / "arms" / arm).glob("run[0-9][0-9][0-9].json"))] for arm in CONFIGS}
    if any(len(items) != 3 for items in records.values()):
        raise RuntimeError("A/B did not finish three trials per arm")
    for arm, items in records.items():
        for item in items:
            if any(item.get(key) is None or not math.isfinite(float(item[key])) for key in METRICS):
                raise RuntimeError(f"{arm} trial omitted a timing, rate, memory, or compact count")
            if not item.get("output_sha256") or not item.get("mtp_proposal_trace_sha256") or \
               item.get("resource_samples", 0) <= 0:
                raise RuntimeError(f"{arm} trial omitted output/proposal/resource evidence")
    common = ("binary_sha256", "model_sha256", "mtp_sha256", "prompt_sha256", "cached_prompt_tokens")
    for key in common:
        if len({item[key] for items in records.values() for item in items}) != 1:
            raise RuntimeError(f"A/B shared field changed: {key}")
    if {item["mtp_sha256"] for items in records.values() for item in items} != {EXPECTED["draft"]}:
        raise RuntimeError("A/B MTP hashes differ from the same I32 draft")
    out = {"identity": state["preflight"], "order": ORDER,
           "arms": {arm: {"runs": len(items),
                            "output_sha256": sorted({item["output_sha256"] for item in items}),
                            "proposal_sha256": sorted({item.get("mtp_proposal_trace_sha256") for item in items}),
                            **{key + "_median": statistics.median(item[key] for item in items
                                   if item.get(key) is not None) for key in METRICS}}
                    for arm, items in records.items()}}
    out["compact_minus_scatter"] = {key: out["arms"]["i32_compact"][key + "_median"] -
                                    out["arms"]["i32_scatter"][key + "_median"] for key in METRICS}
    out["final_output_identical"] = out["arms"]["i32_scatter"]["output_sha256"] == \
        out["arms"]["i32_compact"]["output_sha256"]
    out["proposal_trace_identical"] = out["arms"]["i32_scatter"]["proposal_sha256"] == \
        out["arms"]["i32_compact"]["proposal_sha256"]
    out["note"] = "One 50K Chinese repetition workload; three alternating trials per arm. Whole GPU 0 memory, not process VRAM."
    write_json(suite / "comparison.json", out)
    if not out["final_output_identical"] or not out["proposal_trace_identical"] or \
       len(out["arms"]["i32_scatter"]["output_sha256"]) != 1 or \
       len(out["arms"]["i32_compact"]["output_sha256"]) != 1 or \
       len(out["arms"]["i32_scatter"]["proposal_sha256"]) != 1 or \
       len(out["arms"]["i32_compact"]["proposal_sha256"]) != 1:
        raise RuntimeError("50K/256 output or proposal SHA differs between compact and scatter")
    return out


def audit_compact_after_stop(managed: lifecycle.ManagedService, trial: dict) -> dict:
    if managed.engine_log_path is None:
        raise RuntimeError("engine log path missing")
    with managed.engine_log_path.open("rb") as source:
        source.seek(managed.engine_log_start)
        segment = source.read().decode("utf-8", "replace")
    found = re.findall(r"A3B MTP compact GPU sampler: graph sampled rows (\d+)", segment)
    if len(found) != 1:
        raise RuntimeError("one actual compact graph sample count was not logged")
    rows = int(found[0])
    if bool(rows) != (trial["arm"] == "i32_compact"):
        raise RuntimeError(f"compact graph count does not match {trial['arm']}: {rows}")
    path = Path(trial["record_path"])
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record.get("strata_native_calls_request") or \
       not record.get("mtp_gpu_draft_samples") or \
       record.get("engine_log_prompt_tokens") != 50000 or \
       record.get("engine_log_output_tokens") != 256:
        raise RuntimeError("Strata IQ expert / GPU MTP / 50K-256 execution evidence is missing")
    record["mtp_compact_gpu_sample_rows"] = rows
    write_json(path, record)
    return {"compact_gpu_sample_rows": rows,
            "strata_native_calls_request": record["strata_native_calls_request"]}


def execute(suite: Path) -> dict:
    if lifecycle.port_open():
        raise RuntimeError("8096 is occupied; refusing to touch an unrelated service")
    if suite.exists():
        raise RuntimeError("use a fresh output suite")
    suite.mkdir(parents=True)
    snapshot = suite / "selected.snapshot.json"
    snapshot.write_bytes(SELECTED.read_bytes())
    state = {"preflight": None, "trials": [], "experiment_complete": False,
             "restored": None, "error": None}
    write_json(suite / "runner-state.json", state)
    environment = lifecycle.clean_environment()
    active = None
    primary_error = None
    try:
        # Gate inside the restore scope: even a failed short strict/hash check
        # returns the selected service after the user has cleared port 8096.
        state["preflight"] = preflight()
        write_json(suite / "runner-state.json", state)
        for index, arm in enumerate(ORDER, 1):
            active, _model, _identity, _status = lifecycle.start_service(
                CONFIGS[arm], suite, f"{index:02d}-{arm}", environment, 900)
            try:
                result = run_trial(suite, arm, index, environment)
            finally:
                lifecycle.stop_owned(active)
            result.update(lifecycle.audit_proposal_after_stop(active, result))
            result.update(audit_compact_after_stop(active, result))
            state["trials"].append(result)
            write_json(suite / "runner-state.json", state)
            active = None
            print(json.dumps({"trial": index, "arm": arm, "seconds": result["elapsed_s"]}), flush=True)
        report(suite, state)
        state["experiment_complete"] = True
    except BaseException as error:
        primary_error = error
        state["error"] = f"{type(error).__name__}: {error}"
    finally:
        if active is not None:
            try:
                lifecycle.stop_owned(active)
            except Exception as error:
                state["cleanup_error"] = f"{type(error).__name__}: {error}"
        try:
            restored, _model, _identity, status = lifecycle.start_service(
                snapshot, suite, "restored-selected", environment, 900)
            state["restored"] = {"ok": True, "server_pid": restored.process.pid,
                                 "engine_pid": restored.engine_pid, "loaded": status["loaded"],
                                 "n_ctx": status["context"]["native"]}
        except BaseException as error:
            state["restored"] = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        write_json(suite / "runner-state.json", state)
    if not state["restored"]["ok"]:
        raise RuntimeError(f"selected service restoration failed: {state['restored']}")
    if primary_error is not None:
        raise primary_error
    return state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(preflight(), indent=2))
    else:
        print(json.dumps(execute(args.suite.resolve()), ensure_ascii=False, indent=2))
