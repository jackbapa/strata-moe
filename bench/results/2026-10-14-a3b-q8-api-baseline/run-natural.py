"""Six natural-task resident SSE requests for the initial Q8 baseline only."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from bench_a3b_latency import Client, stream_request
from bench_a3b_supplemental import NATURAL_PROMPTS, natural_case, case_report

out = Path(__file__).resolve().parent / "natural"
out.mkdir(parents=True, exist_ok=True)
client = Client("http://127.0.0.1:8096", 600)
model = client.json("/v1/models")["data"][0]["id"]
assert model == "qwen3.6-35b-a3b-q8_0", model
cases = [natural_case(client, model, name, text, 512) for name, text in NATURAL_PROMPTS.items()]
plan = {"label": "initial Q8 GPU-only baseline; not final tuning", "model": model,
        "cache_policy": "resident", "repeats": 3, "thinking": False, "cases": cases,
        "measurement_code": "bench_a3b_latency.stream_request", "reload_or_unload": False}
(out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
results = []
for case in cases:
    for repeat in range(1, 4):
        trial = f"{case['case_id']}-repeat{repeat:02d}"
        body = {"model": model, "messages": case["messages"], "max_tokens": 512,
                "temperature": 0, "seed": 12345, "stream": True,
                "stream_options": {"include_usage": True},
                "chat_template_kwargs": {"enable_thinking": False}}
        result = stream_request(client, body, out, trial)
        result.update({"trial": trial, "case_id": case["case_id"], "repeat": repeat,
                       "cache_state": "resident-case-first" if repeat == 1 else "resident-repeat",
                       "temperature": 0, "target_prompt_tokens": case["target_prompt_tokens"],
                       "count_endpoint_prompt_tokens": case["count_endpoint_prompt_tokens"],
                       "requested_max_tokens": 512, "resident_process": True})
        result["count_endpoint_matches_actual_usage"] = result["usage"]["prompt_tokens"] == case["count_endpoint_prompt_tokens"]
        results.append(result)
        (out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (out / "case-summary.json").write_text(json.dumps(case_report(results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "rate_note"}, ensure_ascii=False), flush=True)
