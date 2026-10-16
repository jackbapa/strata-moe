"""File-only audit of the completed six-request temperature0.7 supplement."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from audit import audit_sse
from audit_helpers import extract_counters, save_sse
from bench_a3b_latency import summarize, summarize_cases

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=HERE / "measurements-temp07")
    parser.add_argument("--service-cohort", type=Path, default=HERE / "measurements-share")
    args = parser.parse_args()
    folder, service = args.results_root.resolve(), args.service_cohort.resolve()
    plan, results = read(folder / "plan.json"), read(folder / "results.json")
    if plan["temperature"] != .7 or plan["thinking"] or plan["cache_policy"] != "resident" or plan["repeats"] != 3:
        raise RuntimeError("supplement plan differs from the stated sampling/cache matrix")
    if [(r["target_prompt_tokens"], r["repeat"]) for r in results] != [(n, rep) for n in (32, 2048) for rep in (1, 2, 3)]:
        raise RuntimeError("supplement requires short/long with all three trials")
    context = read(service / "plan.json")
    config = read(service / "config.json")
    if config["sampling"] != {"temperature": .7, "top_p": .8, "top_k": 20}:
        raise RuntimeError("service sampling defaults differ from reported web defaults")
    if not context["optional_features"]["prefill_cpu_share"] or context["optional_features"]["mtp_fused_rounds"]:
        raise RuntimeError("service context is not the selected share/unfused cohort")
    if summarize(results) != read(folder / "summary.json") or summarize_cases(results) != read(folder / "case-summary.json"):
        raise RuntimeError("saved supplement summaries do not include all trials")
    rows, outputs = [], {}
    for row in results:
        case = next(c for c in plan["cases"] if c["target_prompt_tokens"] == row["target_prompt_tokens"])
        body = {"model": plan["model"], "messages": case["messages"], "max_tokens": 256,
                "temperature": .7, "seed": 12345, "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if row["request_sha256"] != digest:
            raise RuntimeError("request fingerprint does not reproduce temperature/seed/template")
        if row["usage"]["prompt_tokens"] != row["target_prompt_tokens"] or row["usage"]["completion_tokens"] != 256 or not row["resident_process"] or row["model_reset"] is not None:
            raise RuntimeError("unexpected supplement tokens or model reset")
        save_sse(folder, row["trial"])
        outputs[row["trial"]] = audit_sse(folder, row)
        rows.append({**row, "case_id": f"prompt{row['target_prompt_tokens']}-output256"})
    # The generic runner did not collect an engine fingerprint or RAM sampler.
    # Root ran it immediately after the closed share matrix on the same process.
    # Capture the log separately; never rewrite the closed share engine snapshot.
    snapshot = Path(config["log"]).read_bytes()
    prior = context["completed_requests_in_process_before_run"] + len(read(service / "results.json"))
    counters = extract_counters(snapshot, rows, prior)
    (folder / "engine.log").write_bytes(snapshot)
    (folder / "engine-counters.json").write_text(json.dumps(counters, indent=2) + "\n", encoding="utf-8")
    service_context = {key: context[key] for key in ("actual_exe_sha256", "config_sha256", "engine_pid", "engine_execution_env", "optional_features")}
    service_context.update(service_cohort_plan=str(service / "plan.json"), service_cohort_plan_sha256=sha(service / "plan.json"),
                           service_context_scope="Previously recorded share-service identity; generic supplement did not re-query process identity or resource peaks.",
                           sampling_defaults=config["sampling"], verified_request_seed=12345, prior_requests=prior)
    (folder / "service-context.json").write_text(json.dumps(service_context, indent=2) + "\n", encoding="utf-8")
    report = {"status": "PASS", "audited_utc": datetime.now(timezone.utc).isoformat(), "requests": 6,
              "sampling": {"temperature": .7, "top_p": .8, "top_k": 20, "seed": 12345, "thinking": False},
              "cases": read(folder / "case-summary.json"), "output_hashes": outputs,
              "counter_assertions": counters["assertions"],
              "limitations": ["Separate six-request subset after the greedy share matrix on the same resident service; not pooled into greedy medians.",
                              "Service identity comes from the preceding closed share plan; generic runner did not independently fingerprint the process.",
                              "No additional RAM/VRAM sampler in this subset; report resource peaks from the separately identified share matrix."]}
    (folder / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    names = ["plan.json", "results.json", "summary.json", "case-summary.json", "engine.log", "engine-counters.json", "service-context.json"]
    names += [r["trial"] + suffix for r in results for suffix in (".events.jsonl", ".output.json", ".sse")]
    manifest = {n: {"bytes": (folder / n).stat().st_size, "sha256": sha(folder / n)} for n in sorted(names)}
    (folder / "evidence-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("PASS six-request supplement: request SHA, SSE/timing/usage, all-trial medians, selected engine counters, separate evidence hashes")
    for case, summary in report["cases"].items():
        s = summary["overall"]
        print(case, "TTFB_ms", round(1000 * s["http_response_first_byte_s_median"], 2), "TTFT_s", round(s["ttft_visible_content_s_median"], 3),
              "engine_tps", s["engine_predicted_tokens_per_s_median"], "client_tps", round(s["client_usage_decode_tokens_per_s_median"], 2), "prefill_tps", s["engine_prompt_tokens_per_s_median"])


if __name__ == "__main__":
    main()
