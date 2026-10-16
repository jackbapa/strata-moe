"""Resident SSE checks on natural tasks and nonzero-temperature sampling.

Uses exactly the socket/SSE measurement code from bench_a3b_latency.py.
No model unloading, warm-up exclusion, or idle VRAM release occurs here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench_a3b_latency import Client, prepare_prompt, stream_request, summarize_cases


NATURAL_PROMPTS = {
    "natural-code": """Write a complete Python implementation of a fixed-capacity LRU cache.
Use collections.OrderedDict. The constructor must reject nonpositive capacity.
get(key) should return None for a missing key and promote a hit to most recently used.
put(key, value) must update existing keys, promote them, and evict the least recently used
key when needed. Include concise assertions covering eviction, updating an existing key,
and promotion after get. Briefly explain the time and space complexity. Output the code
first, followed by the explanation.""",
    "natural-chinese": """请用中文给刚接触网络开发的同学解释：浏览器输入一个 HTTPS 网址后，
从域名解析、建立连接、验证证书到收到网页，依次发生了什么？解释 DNS 缓存、TLS
和 HTTP 各自解决的问题，给出一个贴近实际的小例子，并说明“连接成功”为什么不等于
“网页内容已经下载完成”。用大约四百字连贯说明，术语第一次出现时给出简单解释。""",
}


def natural_case(client, model, case_id, text, max_tokens):
    messages = [{"role": "user", "content": text}]
    count = int(client.json("/v1/messages/count_tokens", {
        "model": model, "messages": messages, "thinking": {"type": "disabled"}})["input_tokens"])
    return {"case_id": case_id, "description": "Natural task; prompt length is measured, not padded.",
            "messages": messages, "target_prompt_tokens": count,
            "count_endpoint_prompt_tokens": count, "temperature": 0.0, "max_tokens": max_tokens}


def case_report(results):
    groups = {}
    for result in results:
        groups.setdefault(result["case_id"], []).append(result)
    return {case_id: next(iter(summarize_cases(items).values())) for case_id, items in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--natural-max-tokens", type=int, default=512)
    parser.add_argument("--sample-max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--describe-only", action="store_true", help="Print tasks without contacting the service")
    args = parser.parse_args()
    if min(args.repeats, args.natural_max_tokens, args.sample_max_tokens) < 1:
        parser.error("repeat and output counts must be positive")
    if args.describe_only:
        print(json.dumps({"natural_prompts": NATURAL_PROMPTS,
                          "sampling_subset": {"prompt_tokens": [32, 2048], "temperature": 0.7,
                                              "max_tokens": args.sample_max_tokens},
                          "repeats": args.repeats}, ensure_ascii=False, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    client = Client(args.url or f"http://127.0.0.1:{args.port}", args.timeout)
    model = client.json("/v1/models")["data"][0]["id"]
    cases = [natural_case(client, model, name, text, args.natural_max_tokens)
             for name, text in NATURAL_PROMPTS.items()]
    for target in (32, 2048):
        case = prepare_prompt(client, model, target, False)
        case.update({"case_id": f"sampling07-prompt{target}", "temperature": 0.7,
                     "max_tokens": args.sample_max_tokens,
                     "description": "Same synthetic prompt construction as the main matrix; nonzero-temperature target sampling."})
        cases.append(case)
    metadata = {"model": model, "url": client.url, "cache_policy": "resident",
                "repeats": args.repeats, "seed": args.seed, "thinking": False, "cases": cases,
                "measurement_code": "bench_a3b_latency.stream_request",
                "notes": "The service stays resident. Every measured trial contributes to the overall median. Natural prompts are a workload check, not a scored quality benchmark. Same seed is recorded but bit-identical sampled repeats are not assumed."}
    (args.out / "plan.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    results = []
    for case in cases:
        for repeat in range(1, args.repeats + 1):
            trial = f"{case['case_id']}-repeat{repeat:02d}"
            body = {"model": model, "messages": case["messages"], "max_tokens": case["max_tokens"],
                    "temperature": case["temperature"], "seed": args.seed, "stream": True,
                    "stream_options": {"include_usage": True},
                    "chat_template_kwargs": {"enable_thinking": False}}
            if case["temperature"]:
                body.update({"top_p": 0.95, "top_k": 20})
            result = stream_request(client, body, args.out, trial)
            result.update({"trial": trial, "case_id": case["case_id"], "repeat": repeat,
                           "cache_state": "resident-case-first" if repeat == 1 else "resident-repeat",
                           "temperature": case["temperature"], "target_prompt_tokens": case["target_prompt_tokens"],
                           "count_endpoint_prompt_tokens": case["count_endpoint_prompt_tokens"],
                           "requested_max_tokens": case["max_tokens"], "resident_process": True})
            result["count_endpoint_matches_actual_usage"] = result["usage"]["prompt_tokens"] == case["count_endpoint_prompt_tokens"]
            results.append(result)
            (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (args.out / "case-summary.json").write_text(json.dumps(case_report(results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({key: value for key, value in result.items() if key != "rate_note"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
