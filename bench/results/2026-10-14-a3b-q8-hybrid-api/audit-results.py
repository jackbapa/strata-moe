"""Audit saved hybrid API evidence without HTTP, GPU access, or engine changes."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from statistics import median

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = HERE / "measurements"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    results = read(DATA / "results.json")
    plan = read(DATA / "plan.json")
    config = read(DATA / "config.json")
    counters = read(DATA / "engine-counters.json")
    summary = read(DATA / "case-summary.json")
    log = (DATA / "engine.log").read_text(encoding="utf-8")
    assert len(results) == len(counters["requests"]) == 12
    assert plan["completed_requests_in_process_before_run"] == counters["prior_requests"] == 0
    assert len(re.findall(r"^\[strata\].*engine started:", log, re.M)) == 1
    engine_counts = re.findall(r"^A3B: prompt (\d+) tokens .*?output (\d+) tokens ", log, re.M)
    assert len(engine_counts) == 12
    assert sha(config["exe"]) == plan["actual_exe_sha256"]
    assert sha(DATA / "config.json") == sha(HERE / "config.json") == plan["config_sha256"]
    assert sha(DATA / "engine.log") == counters["log_sha256"]
    assert (HERE / "engine.log").read_bytes().startswith((DATA / "engine.log").read_bytes())
    assert sha(HERE / "run-latency.py") == plan["runner_sha256"]
    assert not {"--hybrid-doorbell", "--gpu-static-experts", "--gpu-dynamic-experts", "--gpu-only-experts"}.intersection(config["args"])
    grouped = defaultdict(list)
    audited_trials = []
    for result, count, (prompt, completion) in zip(results, counters["requests"], engine_counts):
        trial = result["trial"]
        assert trial == count["trial"]
        usage = result["usage"]
        assert int(prompt) == usage["prompt_tokens"] == result["count_endpoint_prompt_tokens"] == result["target_prompt_tokens"]
        assert int(completion) == usage["completion_tokens"] == result["timings"]["predicted_n"]
        assert usage["prompt_tokens_details"]["cached_tokens"] == result["timings"]["cache_n"] == 0
        assert result["received_done"] and result["count_endpoint_matches_actual_usage"]
        assert result["temperature"] == 0 and result["reasoning_characters"] == 0
        done = 0
        packet_usage = packet_timings = None
        finish = None
        text, reasoning, nonempty_times = [], [], []
        for line in (DATA / f"{trial}.events.jsonl").read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            data = "\n".join(row[5:].lstrip(" ") for row in event["sse"].splitlines() if row.startswith("data:"))
            if not data:
                continue
            if data == "[DONE]":
                done += 1
                continue
            packet = json.loads(data)
            assert not packet.get("error")
            if isinstance(packet.get("usage"), dict):
                packet_usage = packet["usage"]
            if isinstance(packet.get("timings"), dict):
                packet_timings = packet["timings"]
            any_text = False
            for choice in packet.get("choices") or []:
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or {}
                visible = delta.get("content")
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(visible, str) and visible:
                    text.append(visible)
                    any_text = True
                if isinstance(thought, str) and thought:
                    reasoning.append(thought)
                    any_text = True
            if any_text:
                nonempty_times.append(event["at_s"])
        assert done == 1 and packet_usage == usage and packet_timings == result["timings"]
        assert finish == result["finish_reason"]
        assert nonempty_times[0] == result["ttft_any_content_or_reasoning_s"]
        assert nonempty_times[-1] == result["last_any_content_or_reasoning_s"]
        actual_rate = (usage["completion_tokens"] - 1) / (nonempty_times[-1] - nonempty_times[0])
        assert abs(actual_rate - result["client_usage_decode_tokens_per_s"]) < 1e-9
        output = read(DATA / f"{trial}.output.json")
        assert output == {"content": "".join(text), "reasoning": "".join(reasoning)}
        assert count["cpu_pool_decode_jobs"] * 3 == count["decode_cpu_rows"] > 0
        assert count["decode_gpu_rows"] > 0 and count["hybrid_calls"] > 0
        assert count["pcie_fraction"] == .25
        assert count["decode_hits"] + count["decode_misses"] == count["decode_cpu_rows"] + count["decode_gpu_rows"]
        assert count["mtp_offered"] == result["timings"]["draft_n"]
        assert count["mtp_accepted"] == result["timings"]["draft_n_accepted"] <= count["mtp_offered"]
        assert count["cpu_pool_jobs_match_decode_cpu_rows"]
        row = {**result, **count, "output_text_sha256": hashlib.sha256(output["content"].encode()).hexdigest()}
        grouped[result["case_id"]].append(row)
        audited_trials.append({"trial": trial, "done_frames": done, "cached_tokens": 0,
                               "prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
                               "output_text_sha256": row["output_text_sha256"]})

    metrics = ["http_response_first_byte_s", "body_first_bytes_s", "ttft_any_content_or_reasoning_s",
               "engine_prompt_tokens_per_s", "engine_predicted_tokens_per_s", "client_usage_decode_tokens_per_s",
               "total_response_s", "prefill_upload_mib", "decode_upload_mib", "decode_gpu_rows", "decode_cpu_rows",
               "cpu_pool_decode_input_quant_ms", "cpu_pool_decode_gate_up_ms", "cpu_pool_decode_hidden_quant_ms",
               "cpu_pool_decode_down_ms", "cpu_pool_decode_jobs", "gpu_miss_rows_request", "gpu_miss_experts_request"]
    cases = {}
    for name, rows in grouped.items():
        assert len(rows) == 3
        medians = {key: median(row[key] for row in rows) for key in metrics}
        for key, value in summary[name]["overall"].items():
            if key.endswith("_median"):
                metric = key[:-7]
                assert abs(median(row[metric] for row in rows) - value) < 1e-9
        accepted = sum(row["mtp_accepted"] for row in rows)
        offered = sum(row["mtp_offered"] for row in rows)
        total_rows = sum(row["decode_cpu_rows"] + row["decode_gpu_rows"] for row in rows)
        medians["cpu_pool_decode_total_phase_ms"] = median(sum(row[key] for key in (
            "cpu_pool_decode_input_quant_ms", "cpu_pool_decode_gate_up_ms", "cpu_pool_decode_hidden_quant_ms", "cpu_pool_decode_down_ms")) for row in rows)
        cases[name] = {"runs": 3, "medians": medians,
                       "completion_tokens": [row["usage"]["completion_tokens"] for row in rows],
                       "distinct_output_text_hashes": len({row["output_text_sha256"] for row in rows}),
                       "accepted": accepted, "offered": offered, "pooled_acceptance": accepted / offered,
                       "pooled_decode_cpu_row_share": sum(row["decode_cpu_rows"] for row in rows) / total_rows,
                       "pooled_decode_cache_hit_share": sum(row["decode_hits"] for row in rows) / total_rows,
                       "adaptive_windows_by_k": {str(k): sum(row["mtp_adaptive_windows_by_k"][str(k)] for row in rows) for k in range(7)}}

    samples = [json.loads(line) for line in (DATA / "resource-samples.jsonl").read_text(encoding="utf-8").splitlines()]
    resources = read(DATA / "resource-summary.json")
    before, after = read(DATA / "resources-before.json"), read(DATA / "resources-final.json")
    assert resources["samples"] == len(samples) == 130 and resources["errors"] == []
    assert {sample["pid"] for sample in samples} == {plan["engine_pid"]}
    assert max(sample["memory"]["rss"] for sample in samples) == resources["sampled_rss_max_bytes"]
    assert max(sample["memory"]["peak_wset"] for sample in samples) == resources["os_lifetime_working_set_peak_bytes"]
    for gpu, maximum in resources["sampled_gpu_used_max_mib"].items():
        assert max(device["used_mib"] for sample in samples for device in sample["gpu"] if str(device["index"]) == gpu) == maximum
    model_provenance = read(ROOT / "models/a3b/Q8_0-provenance.json")
    mtp_provenance = read(ROOT / "models/a3b/mtp-provenance.json")
    model = ROOT / "models/a3b" / model_provenance["file"]
    mtp = ROOT / "models/a3b" / mtp_provenance["file"]
    assert model.stat().st_size == model_provenance["bytes"]
    assert f"model SHA {model_provenance['sha256']};" in log
    assert mtp.stat().st_size == mtp_provenance["bytes"] and sha(mtp) == mtp_provenance["sha256"]
    profile = Path(config["args"][config["args"].index("--expert-profile") + 1])
    baseline = HERE.parent / "2026-10-14-a3b-q8-global-api"
    heldout_plan = read(baseline / "heldout/plan.json")
    for current, previous in zip(plan["cases"][:2], heldout_plan["cases"]):
        assert current["messages"] == previous["messages"]
    baseline_cases = read(baseline / "heldout/case-summary.json")
    matrix = read(baseline / "matrix/case-summary.json")
    comparisons = {}
    for case_id in cases:
        previous = baseline_cases[case_id] if case_id.startswith("heldout-") else matrix[
            next(name for name in matrix if name.startswith(case_id))]
        old = previous["overall"]
        new = cases[case_id]["medians"]
        comparisons[case_id] = {"scope": "Different frozen executable/service/cache history; descriptive comparison, not an isolated causal A/B.",
                               "gpu_only_decode_tps": old["engine_predicted_tokens_per_s_median"],
                               "hybrid_decode_tps": new["engine_predicted_tokens_per_s"],
                               "decode_change_percent": 100 * (new["engine_predicted_tokens_per_s"] / old["engine_predicted_tokens_per_s_median"] - 1),
                               "gpu_only_ttft_s": old["ttft_any_content_or_reasoning_s_median"],
                               "hybrid_ttft_s": new["ttft_any_content_or_reasoning_s"],
                               "gpu_only_prefill_tps": old["engine_prompt_tokens_per_s_median"],
                               "hybrid_prefill_tps": new["engine_prompt_tokens_per_s"]}
    report = {"scope": "Offline audit of the completed 12-request ordinary hybrid cohort; no API/GPU calls. Not a doorbell result.",
              "assertions": {"requests": 12, "done_once_each": True, "api_count_endpoint_engine_counts_match": True,
                             "all_cached_tokens_zero": True, "raw_sse_text_usage_timings_match": True,
                             "all_cpu_pool_jobs_times_three_match_cpu_projection_rows": True,
                             "cpu_and_gpu_rows_positive_each_request": True, "case_medians_recomputed": True,
                             "resource_samples_recomputed": True},
              "cases": cases, "trials": audited_trials, "descriptive_gpu_only_comparisons": comparisons,
              "resources": {"sample_count": len(samples), "requested_interval_s": resources["interval_s"],
                            "before_rss_gib": before["memory"]["rss"] / 2**30,
                            "post_rss_gib": after["memory"]["rss"] / 2**30,
                            "sampled_rss_max_gib": resources["sampled_rss_max_bytes"] / 2**30,
                            "os_lifetime_peak_gib": resources["os_lifetime_working_set_peak_bytes"] / 2**30,
                            "sampled_system_used_max_gib": resources["sampled_system_used_max_bytes"] / 2**30,
                            "post_system_used_gib": after["system_memory"]["used"] / 2**30,
                            "post_system_available_gib": after["system_memory"]["available"] / 2**30,
                            "sampled_gpu_used_max_mib": resources["sampled_gpu_used_max_mib"],
                            "post_gpu": after["gpu"]},
              "identity": {"engine_pid": plan["engine_pid"], "exe_sha256": plan["actual_exe_sha256"],
                           "config_sha256": plan["config_sha256"], "engine_log_sha256": counters["log_sha256"],
                           "model": {**model_provenance, "verification": "Executed engine startup verified the complete model SHA; not reread by this offline audit."},
                           "mtp": {**mtp_provenance, "verification": "Complete current file SHA recomputed by this audit."},
                           "expert_profile_sha256": sha(profile)}}
    (HERE / "audit-results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    files = {}
    for path in sorted(HERE.rglob("*")):
        if path.is_file() and path.name not in {"README.md", "artifact-manifest.json"} and path != HERE / "engine.log" and "__pycache__" not in path.parts:
            files[path.relative_to(HERE).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    manifest = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "scope": "Completed ordinary hybrid API cohort, preparation/reproduction/audit files; mutable README, live appendable root engine.log and manifest itself excluded. Captured measurements/engine.log is included.",
                "identity": report["identity"], "files": files}
    (HERE / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"assertions": report["assertions"], "cases": cases, "resources": report["resources"], "manifest_files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
