"""Read-only alignment of this run's 12 supplemental + 18 matrix requests."""
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools"))
from bench_a3b import adaptive_counters, cache_decay_counters


def main():
    raw = (HERE / "engine.log").read_bytes()
    blocks = re.split(r"(?m)^A3B: prompt ", raw.decode("utf-8", "replace"))[1:]
    supplemental = json.loads((HERE / "supplemental/results.json").read_text(encoding="utf-8"))
    matrix = json.loads((HERE / "matrix/results.json").read_text(encoding="utf-8"))
    assert len(supplemental) == 12 and len(matrix) == 18
    inputs = [("supplemental", row) for row in supplemental] + [("matrix", row) for row in matrix]
    assert len(blocks) >= len(inputs)
    records, groups = [], {}
    for index, ((source, row), block) in enumerate(zip(inputs, blocks)):
        count = re.match(r"(\d+) tokens .*?; output (\d+) tokens", block)
        assert (int(count[1]), int(count[2])) == (row["usage"]["prompt_tokens"], row["usage"]["completion_tokens"])
        assert row["received_done"] and row["count_endpoint_matches_actual_usage"]
        assert row["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and row["reasoning_characters"] == 0
        transfer = re.search(r"A3B transfer: prefill ([\d.]+) MiB; decode ([\d.]+) MiB; decode GPU rows (\d+) CPU rows (\d+); decode hit (\d+) miss (\d+)", block)
        case = row.get("case_id", f"prompt{row['target_prompt_tokens']}-output{row['requested_max_tokens']}")
        record = {"request_index_1based": index + 1, "trial": row["trial"], "source": source, "case": case,
                  "prompt_tokens": row["usage"]["prompt_tokens"], "output_tokens": row["usage"]["completion_tokens"],
                  "mtp_accepted_tokens": row["timings"]["draft_n_accepted"], "mtp_offered_tokens": row["timings"]["draft_n"],
                  "prefill_upload_mib": float(transfer[1]), "decode_upload_mib": float(transfer[2]),
                  "decode_gpu_rows": int(transfer[3]), "decode_cpu_rows": int(transfer[4]),
                  "decode_cache_hits": int(transfer[5]), "decode_cache_misses": int(transfer[6]),
                  "decode_hit_fraction": int(transfer[5]) / (int(transfer[5]) + int(transfer[6])),
                  **adaptive_counters(block), **cache_decay_counters(block)}
        assert record["decode_cpu_rows"] == 0
        output = json.loads((HERE / source / (row["trial"] + ".output.json")).read_text(encoding="utf-8"))
        record["output_text_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        records.append(record)
        groups.setdefault(case, []).append((row, record))
    summaries = {}
    for case, entries in groups.items():
        rows, counters = zip(*entries)
        accepted = sum(row["timings"]["draft_n_accepted"] for row in rows)
        offered = sum(row["timings"]["draft_n"] for row in rows)
        summaries[case] = {"runs": len(rows), "accepted_tokens": accepted, "offered_tokens": offered,
                           "pooled_acceptance_fraction": accepted / offered if offered else None,
                           "distinct_output_texts": len({c["output_text_sha256"] for c in counters}),
                           **{key + "_median": statistics.median(c[key] for c in counters) for key in
                              ("prefill_upload_mib", "decode_upload_mib", "decode_hit_fraction", "prefill_cache_decay_sweeps", "decode_cache_decay_sweeps")},
                           "adaptive_windows_by_k_total": {str(k): sum(c["mtp_adaptive_windows_by_k"][str(k)] for c in counters) for k in range(7)}}
    result = {"scope": "First 30 completed sequential records of engine.log, matched prompt/output counts to API results; later requests, if present, excluded.",
              "source_log_sha256_at_read": hashlib.sha256(raw).hexdigest(), "requests": records, "cases": summaries}
    (HERE / "engine-counters.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
