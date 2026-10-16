"""Pure-file validation for the doorbell API experiment; no HTTP/GPU calls."""
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from bench_a3b import adaptive_counters, cache_decay_counters


def latest_process_log(text):
    starts = list(re.finditer(r"(?m)^\[strata\].*engine started:", text))
    return text[starts[-1].start():] if starts else text


def save_sse(directory, trial):
    # The shared socket reader retains each received frame and its arrival
    # time. This additional .sse copy uses LF and frame separators for review.
    frames = [json.loads(line)["sse"] for line in (directory / f"{trial}.events.jsonl").read_text(encoding="utf-8").splitlines()]
    (directory / f"{trial}.sse").write_text("\n\n".join(frames) + "\n\n", encoding="utf-8", newline="\n")


def validate_result(result):
    usage, timings = result.get("usage") or {}, result.get("timings") or {}
    if not result.get("received_done") or not result.get("count_endpoint_matches_actual_usage"):
        raise RuntimeError("missing SSE DONE or tokenizer/usage count mismatch")
    if timings.get("cache_n") != 0 or timings.get("prompt_n") != usage.get("prompt_tokens") or timings.get("predicted_n") != usage.get("completion_tokens"):
        raise RuntimeError("SSE usage differs from engine DONE-derived counts or unexpectedly reused a prompt")
    if usage.get("total_tokens") != usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0):
        raise RuntimeError("SSE usage total token count mismatch")
    for field in ("http_response_first_byte_s", "ttft_any_content_or_reasoning_s", "ttft_visible_content_s",
                  "engine_prompt_tokens_per_s", "engine_predicted_tokens_per_s", "client_usage_decode_tokens_per_s"):
        value = result.get(field)
        if value is None or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"missing/non-finite measured {field}")
    if result["http_response_first_byte_s"] > result["ttft_any_content_or_reasoning_s"] or result.get("reasoning_characters"):
        raise RuntimeError("invalid first-byte/first-token order or unexpected reasoning output")


def one(pattern, block, label):
    matches = list(re.finditer(pattern, block))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {label} counter line, found {len(matches)}")
    return matches[0]


def extract_counters(log_bytes, results, prior_requests):
    text = latest_process_log(log_bytes.decode("utf-8", "replace"))
    # A3B static precedes DONE; all remaining counters follow A3B: prompt.
    # Splitting at the latter would associate static facts with the next trial.
    blocks = re.split(r"(?m)(?=^A3B static:)", text)[1:]
    selected = blocks[prior_requests:prior_requests + len(results)]
    if len(selected) != len(results):
        raise RuntimeError("missing completed doorbell request blocks")
    records = []
    for result, block in zip(results, selected):
        validate_result(result)
        counts = one(r"A3B: prompt (\d+) tokens ([\d.]+)s [\d.]+ tok/s; output (\d+) tokens ([\d.]+)s [\d.]+ tok/s", block, "performance")
        if (int(counts[1]), int(counts[3])) != (result["usage"]["prompt_tokens"], result["usage"]["completion_tokens"]):
            raise RuntimeError("API/engine request order or token counts differ")
        for seconds, field in ((float(counts[2]), "prompt_ms"), (float(counts[4]), "predicted_ms")):
            if abs(seconds * 1000 - result["timings"][field]) > .6:
                raise RuntimeError("printed engine time differs from DONE-derived SSE timing beyond rounding")
        static = one(r"A3B static: calls (\d+); mapped host ([\d.]+) MiB; shared GPU scratch ([\d.]+) MiB; prefill route reads (\d+) decode route reads (\d+)", block, "static")
        doorbell = one(r"A3B doorbell: calls (\d+) CPU requests (\d+) callback ([\d.]+) ms; mailbox to host (\d+) bytes to GPU (\d+) bytes", block, "doorbell")
        transfer = one(r"A3B transfer: prefill ([\d.]+) MiB; decode ([\d.]+) MiB; decode GPU rows (\d+) CPU rows (\d+); decode hit (\d+) miss (\d+)", block, "transfer")
        total = one(r"A3B cache: ([\d.]+) MiB; GPU rows (\d+) CPU rows (\d+); hit (\d+) miss (\d+); uploaded ([\d.]+) MiB", block, "cache")
        routes = one(r"A3B backend: route reads (\d+); native layout calls (\d+); single-column calls (\d+)", block, "backend")
        pool = one(r"A3B CPU pool: calls (\d+) jobs (\d+) unique inputs (\d+) experts (\d+); input quant ([\d.]+) ms gate/up ([\d.]+) ms hidden quant ([\d.]+) ms down ([\d.]+) ms", block, "CPU pool")
        mtp = one(r"A3B MTP: windows (\d+) offered (\d+) accepted (\d+); draft ([\d.]+) ms catchup ([\d.]+) ms", block, "MTP")
        gpu_rows, cpu_rows = int(transfer[3]), int(transfer[4])
        if int(static[1]) <= 0 or int(static[1]) != int(doorbell[1]) or int(static[5]) != 0:
            raise RuntimeError("doorbell/static execution mismatch or decode routing readback")
        if gpu_rows <= 0 or cpu_rows <= 0 or gpu_rows % 3 or cpu_rows % 3:
            raise RuntimeError("measured decode did not exercise both CPU/GPU projection rows")
        if float(transfer[2]) != 0 or int(transfer[5]) != gpu_rows or int(transfer[6]) != cpu_rows:
            raise RuntimeError("decode uploaded expert weights or fixed-map hit/miss ownership differs")
        if int(pool[1]) <= 0 or int(pool[2]) * 3 != cpu_rows or int(doorbell[2]) < int(pool[1]):
            raise RuntimeError("CPU pool job/call counts do not match actual decode rows")
        if int(doorbell[4]) <= 0 or int(doorbell[5]) < cpu_rows // 3 * 2048 * 4:
            raise RuntimeError("doorbell mailbox traffic does not cover CPU result rows")
        if (int(mtp[2]), int(mtp[3])) != (result["timings"].get("draft_n"), result["timings"].get("draft_n_accepted")) or int(mtp[3]) > int(mtp[2]):
            raise RuntimeError("MTP counters differ from engine DONE-derived SSE counters")
        raw_done = re.findall(r"(?m)^DONE (\d+) (\d+) ([\d.]+) ([\d.]+) (\S+) (\d+) (\d+) (\d+)", block)
        if raw_done and (len(raw_done) != 1 or tuple(map(int, (raw_done[0][0], raw_done[0][1], raw_done[0][5], raw_done[0][6]))) !=
                         (int(counts[3]), int(counts[1]), int(mtp[3]), int(mtp[2]))):
            raise RuntimeError("raw engine DONE counts mismatch")
        decay = cache_decay_counters(block)
        if not decay or decay["decode_cache_decay_sweeps"] != 0:
            raise RuntimeError("fixed doorbell cache decay counter missing/nonzero")
        record = {"trial": result["trial"], "case_id": result["case_id"],
                  "gpu_static_calls_request": int(static[1]), "doorbell_calls_request": int(doorbell[1]),
                  "doorbell_cpu_requests_request": int(doorbell[2]), "doorbell_callback_ms_request": float(doorbell[3]),
                  "mailbox_to_host_bytes_request": int(doorbell[4]), "mailbox_to_gpu_bytes_request": int(doorbell[5]),
                  "mapped_extra_host_mib": float(static[2]), "shared_gpu_scratch_mib": float(static[3]),
                  "prefill_route_reads": int(static[4]), "decode_route_reads": int(static[5]),
                  "prefill_upload_mib": float(transfer[1]), "decode_upload_mib": float(transfer[2]),
                  "decode_gpu_projection_rows": gpu_rows, "decode_cpu_projection_rows": cpu_rows,
                  "decode_hits": int(transfer[5]), "decode_misses": int(transfer[6]),
                  "cache_mib": float(total[1]), "gpu_projection_rows_process_total": int(total[2]),
                  "cpu_projection_rows_process_total": int(total[3]), "route_reads_process_total": int(routes[1]),
                  "cpu_pool_decode_calls": int(pool[1]), "cpu_pool_decode_jobs": int(pool[2]),
                  "cpu_pool_decode_unique_inputs": int(pool[3]), "cpu_pool_decode_unique_experts": int(pool[4]),
                  "cpu_pool_decode_input_quant_ms": float(pool[5]), "cpu_pool_decode_gate_up_ms": float(pool[6]),
                  "cpu_pool_decode_hidden_quant_ms": float(pool[7]), "cpu_pool_decode_down_ms": float(pool[8]),
                  "mtp_windows": int(mtp[1]), "mtp_offered": int(mtp[2]), "mtp_accepted": int(mtp[3]),
                  "mtp_acceptance": int(mtp[3]) / int(mtp[2]) if int(mtp[2]) else None,
                  "mtp_draft_ms": float(mtp[4]), "mtp_catchup_ms": float(mtp[5]),
                  "raw_engine_DONE_logged": bool(raw_done), **adaptive_counters(block), **decay}
        hybrid = re.search(r"A3B hybrid: calls (\d+); PCIe fraction ([\d.]+); GPU miss rows (\d+) experts (\d+); input allocations (\d+) bytes (\d+) fallbacks (\d+)", block)
        if hybrid:
            if float(hybrid[2]) != 0:
                raise RuntimeError("doorbell experiment enabled a PCIe miss fraction")
            record.update(ordinary_hybrid_calls_request=int(hybrid[1]), pcie_fraction=float(hybrid[2]),
                          ordinary_gpu_miss_rows_request=int(hybrid[3]), ordinary_gpu_miss_experts_request=int(hybrid[4]),
                          ordinary_hybrid_input_allocations_process_total=int(hybrid[5]), ordinary_hybrid_input_bytes=int(hybrid[6]),
                          ordinary_hybrid_input_fallbacks_process_total=int(hybrid[7]))
        records.append(record)
    return {"log_sha256": hashlib.sha256(log_bytes).hexdigest(), "prior_requests": prior_requests,
            "requests": records,
            "assertions": "doorbell/static calls positive and equal; both actual CPU/GPU decode projection rows positive; pool jobs*3=CPU rows; decode route reads=0; decode expert weight upload=0.0MiB; counts/times/MTP match engine DONE-derived SSE; fixed-map hit/miss ownership; no decode decay",
            "scope": "Pool phases/rows are decode deltas. Doorbell/mailbox/static calls are whole-request deltas. Cumulative process values are labeled. Engine stdout DONE is normally consumed by the server; its exported SSE timings/usage are checked, and a literal DONE is additionally checked if present in the log."}
