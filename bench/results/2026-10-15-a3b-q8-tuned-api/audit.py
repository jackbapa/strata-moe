"""Audit completed tuned-Q8 experiment files; never accesses a service/GPU."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_helpers import extract_counters, validate_result
from bench_a3b_latency import summarize_cases

HERE = Path(__file__).resolve().parent
CASES = ("heldout-python-bfs", "heldout-chinese-transactions", "prompt32-output256", "prompt2048-output256")


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_sse(folder, row):
    validate_result(row)
    records = [json.loads(line) for line in (folder / f"{row['trial']}.events.jsonl").read_text(encoding="utf-8").splitlines()]
    content, reasoning, text_times = [], [], []
    usage = timings = None
    done = 0
    for event in records:
        data = "\n".join(line[5:].lstrip(" ") for line in event["sse"].splitlines() if line.startswith("data:"))
        if not data:
            continue
        if data == "[DONE]":
            done += 1
            continue
        item = json.loads(data)
        if item.get("error"):
            raise RuntimeError(f"SSE error in {row['trial']}")
        usage = item.get("usage") if isinstance(item.get("usage"), dict) else usage
        timings = item.get("timings") if isinstance(item.get("timings"), dict) else timings
        any_text = False
        for choice in item.get("choices") or []:
            delta = choice.get("delta") or {}
            text, thought = delta.get("content"), delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(text, str) and text:
                content.append(text); any_text = True
            if isinstance(thought, str) and thought:
                reasoning.append(thought); any_text = True
        if any_text:
            text_times.append(event["at_s"])
    if done != 1 or usage != row["usage"] or timings != row["timings"] or not text_times:
        raise RuntimeError(f"SSE completion/usage/timings mismatch in {row['trial']}")
    actual_text = {"content": "".join(content), "reasoning": "".join(reasoning)}
    if actual_text != read(folder / f"{row['trial']}.output.json"):
        raise RuntimeError("saved output differs from SSE text")
    rate = (usage["completion_tokens"] - 1) / (text_times[-1] - text_times[0])
    if not math.isclose(rate, row["client_usage_decode_tokens_per_s"], rel_tol=1e-10) or not math.isclose(text_times[0], row["ttft_any_content_or_reasoning_s"], abs_tol=1e-9):
        raise RuntimeError("SSE timestamps do not reproduce TTFT/client output rate")
    normalized_sse = "\n\n".join(event["sse"] for event in records) + "\n\n"
    if (folder / f"{row['trial']}.sse").read_text(encoding="utf-8") != normalized_sse:
        raise RuntimeError("normalized SSE copy differs from saved frames")
    return digest(folder / f"{row['trial']}.output.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=HERE / "measurements")
    args = parser.parse_args()
    folder = args.results_root.resolve()
    plan, results = read(folder / "plan.json"), read(folder / "results.json")
    expected_order = [(case, repeat) for case in CASES for repeat in (1, 2, 3)]
    if [(row["case_id"], row["repeat"]) for row in results] != expected_order or plan["phase"] != "all":
        raise RuntimeError("report requires the complete ordered twelve-request matrix")
    if digest(folder / "config.json") != plan["config_sha256"] or plan["actual_exe_sha256"].lower() != plan["expected_exe_sha256"].lower():
        raise RuntimeError("config/executable fingerprint mismatch")
    output_hashes = {row["trial"]: audit_sse(folder, row) for row in results}
    summary = {case: next(iter(summarize_cases([row for row in results if row["case_id"] == case]).values())) for case in CASES}
    if summary != read(folder / "case-summary.json"):
        raise RuntimeError("case medians/first/subsequent runs do not match all saved trials")
    counters = extract_counters((folder / "engine.log").read_bytes(), results, plan["completed_requests_in_process_before_run"])
    if counters != read(folder / "engine-counters.json"):
        raise RuntimeError("saved counters differ from re-parsed engine log")
    optional = plan.get("optional_features", {})
    for row in counters["requests"]:
        for feature, field in (("prefill_cpu_share", "prefill_share_calls"), ("mtp_fused_rounds", "mtp_fused_rounds")):
            if optional.get(feature, False) != (field in row):
                raise RuntimeError(f"optional feature/counter presence differs: {feature}")
    samples = [json.loads(line) for line in (folder / "resource-samples.jsonl").read_text(encoding="utf-8").splitlines()]
    resources = read(folder / "resource-summary.json")
    if not samples or any(sample["pid"] != plan["engine_pid"] for sample in samples):
        raise RuntimeError("resource sample process mismatch")
    rss_max = max(sample["memory"]["rss"] for sample in samples)
    gpu_max = {str(index): max(gpu["used_mib"] for sample in samples for gpu in sample["gpu"] if gpu["index"] == index)
               for index in {gpu["index"] for sample in samples for gpu in sample["gpu"]}}
    if resources["samples"] != len(samples) or resources["sampled_rss_max_bytes"] != rss_max or resources["sampled_gpu_used_max_mib"] != gpu_max:
        raise RuntimeError("resource maxima do not match saved periodic samples")
    rows = []
    for case in CASES:
        selected = [row for row in results if row["case_id"] == case]
        selected_counters = [row for row in counters["requests"] if row["case_id"] == case]
        offered = sum(row["mtp_offered"] for row in selected_counters)
        accepted = sum(row["mtp_accepted"] for row in selected_counters)
        rows.append({"case_id": case, "trials": 3,
                     "actual_prompt_tokens": sorted({row["usage"]["prompt_tokens"] for row in selected}),
                     "actual_completion_tokens": [row["usage"]["completion_tokens"] for row in selected],
                     "pooled_mtp_accepted": accepted, "pooled_mtp_offered": offered,
                     "pooled_mtp_acceptance": accepted / offered if offered else None,
                     "distinct_output_files": len({output_hashes[row["trial"]] for row in selected}),
                     "cpu_decode_projection_rows": sum(row["decode_cpu_projection_rows"] for row in selected_counters),
                     "gpu_decode_projection_rows": sum(row["decode_gpu_projection_rows"] for row in selected_counters),
                     **summary[case]["overall"]})
    report = {"audited_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", "requests": 12,
              "actual_exe_sha256": plan["actual_exe_sha256"], "execution_env": plan["engine_execution_env"],
              "optional_features": optional,
              "cases": rows, "resources": resources, "counter_assertions": counters["assertions"],
              "limitations": ["SSE deltas can contain MTP bursts; output rate uses token usage and first/last nonempty-text times, not event count.",
                              "Resource sampling starts after model readiness; sampled GPU maximum is not a continuous instantaneous peak.",
                              "CPU pool phases are decode-only; doorbell callback/mailbox fields include the whole request.",
                              "Only two natural heldout tasks, and no broad model-quality evaluation.",
                              "An old/new executable comparison cannot isolate the native expert reuse environment setting."]}
    (folder / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    table = ["| Case | Actual input | Actual output (3 trials) | TTFB ms | TTFT s | Engine output t/s | Client output t/s | Prefill t/s |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        table.append(f"| {row['case_id']} | {','.join(map(str,row['actual_prompt_tokens']))} | {'/'.join(map(str,row['actual_completion_tokens']))} | {row['http_response_first_byte_s_median']*1000:.2f} | {row['ttft_visible_content_s_median']:.3f} | {row['engine_predicted_tokens_per_s_median']:.2f} | {row['client_usage_decode_tokens_per_s_median']:.2f} | {row['engine_prompt_tokens_per_s_median']:.2f} |")
    table += ["", "All three trials are included in each median. See case-summary.json for first/subsequent values.",
              f"Sampled RSS max: {rss_max/2**30:.4f} GiB; sampled GPU maxima: {gpu_max} MiB. These are sampled values."]
    (folder / "audit-table.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    names = ["plan.json", "results.json", "config.json", "case-summary.json", "engine-counters.json", "engine.log",
             "resources-before.json", "resources-final.json", "resource-summary.json", "resource-samples.jsonl"]
    names += [row["trial"] + suffix for row in results for suffix in (".events.jsonl", ".output.json", ".sse")]
    manifest = {name: {"bytes": (folder/name).stat().st_size, "sha256": digest(folder/name)} for name in sorted(names)}
    (folder / "evidence-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("PASS tuned-Q8 file audit: twelve requests, SSE-derived TTFT/output rate, DONE/usage/MTP/log counters, optional prefill/fused counters, all-trial medians, resource samples and evidence hashes")
    print("\n".join(table))


if __name__ == "__main__":
    main()
