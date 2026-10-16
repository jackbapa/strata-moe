"""Summarize six same-binary 50K/256 service requests without causal claims."""

import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent / "50k-256"
ARMS = ("selected", "k2-sync", "k2-async")
FIELDS = (
    "http_response_first_byte_s",
    "ttft_any_content_or_reasoning_s",
    "engine_prompt_tokens_per_s",
    "engine_predicted_tokens_per_s",
    "gpu0_total_used_mib_observed_peak",
    "engine_private_bytes_observed_peak",
    "mtp_offered_tokens",
    "mtp_accepted_tokens",
    "mtp_device_chain_k2_calls",
    "mtp_device_chain_k2_fallbacks",
)


def main():
    audit = json.loads((HERE / "proposal-audit.json").read_text())
    data = {}
    all_runs = []
    for arm in ARMS:
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(
            (HERE / "arms" / arm).glob("run[0-9][0-9][0-9].json"))]
        assert len(runs) == 2, (arm, len(runs))
        all_runs.extend(runs)
        metrics = {field: [run.get(field) for run in runs] for field in FIELDS}
        data[arm] = {
            "runs": 2,
            "metrics": metrics,
            "median": {field: statistics.median(values) if all(x is not None for x in values)
                       else None for field, values in metrics.items()},
            "output_sha256": [run["output_sha256"] for run in runs],
            "proposal_trace_sha256_recovered": [item["trace_sha256"] for item in audit["runs"][arm]],
            "proposal_windows_recovered": [item["trace_windows"] for item in audit["runs"][arm]],
        }
    for key in ("binary_sha256", "model_sha256", "mtp_sha256", "prompt_sha256",
                "cached_prompt_tokens", "engine_log_prompt_tokens", "engine_log_output_tokens"):
        assert len({run[key] for run in all_runs}) == 1, key
    assert all(run["engine_log_prompt_tokens"] == 50000 and
               run["engine_log_output_tokens"] == 256 and
               run["cached_prompt_tokens"] == 0 for run in all_runs)
    result = {
        "runs_total": len(all_runs),
        "binary_sha256": all_runs[0]["binary_sha256"],
        "model_sha256": all_runs[0]["model_sha256"],
        "mtp_sha256": all_runs[0]["mtp_sha256"],
        "prompt_sha256": all_runs[0]["prompt_sha256"],
        "prompt_tokens": 50000,
        "output_tokens": 256,
        "cached_prompt_tokens": 0,
        "arms": data,
        "all_outputs_identical": len({run["output_sha256"] for run in all_runs}) == 1,
        "all_proposals_identical": len({item["trace_sha256"] for arm in ARMS
                                        for item in audit["runs"][arm]}) == 1,
        "median_difference": {
            "k2_async_minus_k2_sync": {field: data["k2-async"]["median"][field] -
                                         data["k2-sync"]["median"][field]
                                         for field in FIELDS if data["k2-async"]["median"][field] is not None
                                         and data["k2-sync"]["median"][field] is not None},
            "k2_async_minus_selected": {field: data["k2-async"]["median"][field] -
                                         data["selected"]["median"][field]
                                         for field in FIELDS if data["k2-async"]["median"][field] is not None
                                         and data["selected"]["median"][field] is not None},
        },
        "limitations": [
            "Two independent model restarts per arm; these medians are exploratory.",
            "GPU memory is total observed GPU 0 usage, not per-process attribution.",
            "The original benchmark parser missed proposal traces; proposal-audit.json recovers them from raw logs.",
            "K2 synchronous run001 has a different proposal stream despite identical final output.",
        ],
    }
    (HERE / "three-arm-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    compact = {arm: {k: [round(x, 3) if isinstance(x, float) else x for x in v]
                     for k, v in data[arm]["metrics"].items() if k in FIELDS[:5]}
               for arm in ARMS}
    print(json.dumps({"runs_total": result["runs_total"],
                      "all_outputs_identical": result["all_outputs_identical"],
                      "all_proposals_identical": result["all_proposals_identical"],
                      "metrics": compact}, indent=2))


if __name__ == "__main__":
    main()
