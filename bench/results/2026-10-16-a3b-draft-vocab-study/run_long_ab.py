"""Run an isolated same-binary 50K/256 full/subset MTP vocabulary A/B.

The two arms intentionally use different MTP GGUFs.  They therefore live in
separate bench_a3b_ab suites; this runner copies one exact request template
between them and reports the MTP hashes as the sole inference difference.
Always restores the selected 8096 service in finally.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
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


CONFIGS = {arm: HERE / f"{arm}.snapshot.json" for arm in ("full", "subset")}
ORDER = ("full", "subset") * 3
SELECTED = ROOT / "strata-a3b-iq3s-strata-optimized.json"
EXPECTED = {
    "engine": "a6020a08638e937963679a888084f21b7a8a78b94244695b778e6ce107738d46",
    "target": "66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66",
    "full": "f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a",
    "subset": "7a0bd6c1a1c8d5037262c52b50375d063e128a254f5f6348a0772543893512ae",
}
METRICS = ("http_response_first_byte_s", "ttft_any_content_or_reasoning_s",
           "engine_prompt_tokens_per_s", "engine_predicted_tokens_per_s",
           "gpu0_total_used_mib_observed_peak", "engine_rss_bytes_observed_peak",
           "mtp_offered_tokens", "mtp_accepted_tokens", "mtp_acceptance_rate")


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def strict_gate() -> dict:
    root = HERE / "strict"
    logs = {}
    states = {}
    for arm in CONFIGS:
        folder = root / arm
        log = (folder / "strict.log").read_text(encoding="utf-8", errors="replace")
        logs[arm] = {
            "classes": [line for line in log.splitlines() if line.startswith("MTP class limit=")],
            "proposals": [line for line in log.splitlines() if line.startswith("A3B MTP proposal trace:")],
            "sequential_pass": "PASS: independent sequential-prefix continuation" in log,
        }
        files = sorted((folder / "state").glob("*.bin"))
        states[arm] = {file.name: {"bytes": file.stat().st_size,
                                   "sha256": sha256_file(file)} for file in files}
    if (len(logs["full"]["classes"]) != 8 or len(logs["full"]["proposals"]) != 29 or
            not all(value["sequential_pass"] for value in logs.values()) or
            logs["full"] != logs["subset"] or len(states["full"]) != 16 or
            states["full"] != states["subset"]):
        raise RuntimeError("real-model strict full/subset gate is incomplete or unequal")
    return {"state_pairs_bitwise_equal": 16, "proposal_lines_equal": 29,
            "class_lines_equal": 8}


def preflight() -> dict:
    configs = {arm: json.loads(path.read_text(encoding="utf-8")) for arm, path in CONFIGS.items()}
    full, subset = configs["full"], configs["subset"]
    ignore = {"log", "args"}
    if {k: v for k, v in full.items() if k not in ignore} != {k: v for k, v in subset.items() if k not in ignore}:
        raise RuntimeError("configs differ outside MTP model and log")
    full_args, subset_args = full["args"].copy(), subset["args"].copy()
    for args in (full_args, subset_args):
        args[args.index("--mtp-model") + 1] = "<MTP>"
    if full_args != subset_args or full["log"] == subset["log"]:
        raise RuntimeError("configs differ in unexpected inference arguments or share a log")
    paths = {}
    for arm, path in CONFIGS.items():
        _, target, exe, _log, mtp = load_config(path)
        if not mtp:
            raise RuntimeError("missing MTP GGUF")
        paths[arm] = {"engine": sha256_file(exe), "target": sha256_file(target),
                      "mtp": sha256_file(mtp), "mtp_path": str(mtp),
                      "config_sha256": sha256_file(path)}
        if (paths[arm]["engine"] != EXPECTED["engine"] or
                paths[arm]["target"] != EXPECTED["target"] or
                paths[arm]["mtp"] != EXPECTED[arm]):
            raise RuntimeError(f"{arm} binary/model hash changed")
    return {"config_and_model": paths, "strict": strict_gate(), "order": ORDER,
            "request": {"prompt_tokens": 50000, "completion_tokens": 256,
                        "temperature": 0, "seed": 12345, "thinking": False},
            "selected_config_sha256": sha256_file(SELECTED)}


def trial_command(suite: Path, arm: str) -> list[str]:
    return [sys.executable, str(ROOT / "tools" / "bench_a3b_ab.py"), "run",
            "--suite", str(suite / arm), "--arm", arm,
            "--config", str(CONFIGS[arm]), "--prompt-tokens", "50000",
            "--max-tokens", "256", "--repeats", "1", "--temperature", "0",
            "--seed", "12345", "--sample-interval", "0.5", "--timeout", "900"]


def run_trial(suite: Path, arm: str, index: int, environment: dict[str, str]) -> dict:
    arm_suite = suite / arm
    if arm == "subset":
        template = suite / "full" / "request-template.json"
        if not template.exists():
            raise RuntimeError("full arm did not establish the prompt")
        target = arm_suite / "request-template.json"
        if not target.exists():
            arm_suite.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(template, target)
        elif target.read_bytes() != template.read_bytes():
            raise RuntimeError("full/subset prompt templates differ")
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
    common = ("binary_sha256", "model_sha256", "prompt_sha256", "cached_prompt_tokens")
    for key in common:
        if len({item[key] for items in records.values() for item in items}) != 1:
            raise RuntimeError(f"A/B shared field changed: {key}")
    if {item["mtp_sha256"] for item in records["full"]} != {EXPECTED["full"]} or \
       {item["mtp_sha256"] for item in records["subset"]} != {EXPECTED["subset"]}:
        raise RuntimeError("A/B MTP hashes differ from the planned pair")
    out = {"identity": state["preflight"], "order": ORDER,
           "arms": {arm: {"runs": len(items),
                            "output_sha256": sorted({item["output_sha256"] for item in items}),
                            "proposal_sha256": sorted({item.get("mtp_proposal_trace_sha256") for item in items}),
                            **{key + "_median": statistics.median(item[key] for item in items
                                   if item.get(key) is not None) for key in METRICS}}
                    for arm, items in records.items()}}
    out["subset_minus_full"] = {key: out["arms"]["subset"][key + "_median"] -
                                out["arms"]["full"][key + "_median"] for key in METRICS}
    out["final_output_identical"] = out["arms"]["full"]["output_sha256"] == out["arms"]["subset"]["output_sha256"]
    out["proposal_trace_identical"] = out["arms"]["full"]["proposal_sha256"] == out["arms"]["subset"]["proposal_sha256"]
    out["note"] = "One 50K Chinese repetition workload; three alternating trials per arm. Whole GPU 0 memory, not process VRAM."
    write_json(suite / "comparison.json", out)
    return out


def execute(suite: Path) -> dict:
    if lifecycle.port_open():
        raise RuntimeError("8096 is occupied; refusing to touch an unrelated service")
    if suite.exists():
        raise RuntimeError("use a fresh output suite")
    plan = preflight()
    suite.mkdir(parents=True)
    snapshot = suite / "selected.snapshot.json"
    snapshot.write_bytes(SELECTED.read_bytes())
    state = {"preflight": plan, "trials": [], "experiment_complete": False,
             "restored": None, "error": None}
    write_json(suite / "runner-state.json", state)
    environment = lifecycle.clean_environment()
    active = None
    primary_error = None
    try:
        for index, arm in enumerate(ORDER, 1):
            active, _model, _identity, _status = lifecycle.start_service(
                CONFIGS[arm], suite, f"{index:02d}-{arm}", environment, 900)
            try:
                result = run_trial(suite, arm, index, environment)
            finally:
                lifecycle.stop_owned(active)
            result.update(lifecycle.audit_proposal_after_stop(active, result))
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
