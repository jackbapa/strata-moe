"""Align six held-out API requests with the latest appended engine process."""
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3] / "tools"))
from bench_a3b import adaptive_counters, cache_decay_counters

raw = (HERE / "engine.log").read_bytes()
log = raw.decode("utf-8", "replace")
process = log[log.rfind("[strata]"):]
blocks = re.split(r"(?m)^A3B: prompt ", process)[1:]
requests = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
assert len(blocks) == len(requests) == 6
records, groups = [], {}
for index, (request, block) in enumerate(zip(requests, blocks)):
    counts = re.match(r"(\d+) tokens .*?; output (\d+) tokens", block)
    assert (int(counts[1]), int(counts[2])) == (request["usage"]["prompt_tokens"], request["usage"]["completion_tokens"])
    assert request["received_done"] and request["count_endpoint_matches_actual_usage"]
    assert request["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 and request["reasoning_characters"] == 0
    transfer = re.search(r"A3B transfer: prefill ([\d.]+) MiB; decode ([\d.]+) MiB; decode GPU rows (\d+) CPU rows (\d+); decode hit (\d+) miss (\d+)", block)
    cpu_total = re.search(r"A3B cache: .*? GPU rows (\d+) CPU rows (\d+)", block)
    assert int(transfer[4]) == int(cpu_total[2]) == 0
    output = json.loads((HERE / (request["trial"] + ".output.json")).read_text(encoding="utf-8"))
    record = {"request_index_1based": index + 1, "trial": request["trial"], "case_id": request["case_id"],
              "prefill_upload_mib": float(transfer[1]), "decode_upload_mib": float(transfer[2]),
              "decode_gpu_rows": int(transfer[3]), "decode_cpu_rows": int(transfer[4]),
              "decode_hits": int(transfer[5]), "decode_misses": int(transfer[6]),
              "decode_hit_fraction": int(transfer[5]) / (int(transfer[5]) + int(transfer[6])),
              "mtp_accepted": request["timings"]["draft_n_accepted"], "mtp_offered": request["timings"]["draft_n"],
              "output_text_sha256": hashlib.sha256(json.dumps(output, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
              **adaptive_counters(block), **cache_decay_counters(block)}
    records.append(record)
    groups.setdefault(request["case_id"], []).append(record)
summary = {}
for case, rows in groups.items():
    accepted, offered = sum(x["mtp_accepted"] for x in rows), sum(x["mtp_offered"] for x in rows)
    summary[case] = {"runs": len(rows), "mtp_accepted": accepted, "mtp_offered": offered,
                     "pooled_acceptance": accepted / offered, "distinct_outputs": len({x["output_text_sha256"] for x in rows}),
                     **{key + "_median": statistics.median(x[key] for x in rows) for key in
                        ("prefill_upload_mib", "decode_upload_mib", "decode_hit_fraction", "decode_cache_decay_sweeps")},
                     "adaptive_windows_by_k_total": {str(k): sum(x["mtp_adaptive_windows_by_k"][str(k)] for x in rows) for k in range(7)}}
result = {"scope": "Latest process block only; copied engine.log also contains an older initial-baseline process, deliberately excluded.",
          "source_sha256": hashlib.sha256(raw).hexdigest(), "engine_start_line": process.splitlines()[0],
          "requests": records, "cases": summary}
(HERE / "engine-counters.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
