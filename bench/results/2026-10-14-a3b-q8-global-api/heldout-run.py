"""Six new-prompt requests on the separately restarted production service.

Prepared only; run after root grants its GPU window. Does not reload/unload.
Uses the same socket/SSE measurements as the 30-request candidate matrix.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from bench_a3b_latency import Client, stream_request, summarize_cases
from bench_a3b_supplemental import natural_case

PROMPTS = {
    "heldout-python-bfs": """Write a complete Python function shortest_path(graph, start, goal) for an unweighted directed graph represented by a dictionary mapping each node to a list of neighbors.
Use breadth-first search with collections.deque and a parent map. Return a list of nodes describing a shortest path, or None if no path exists. Nodes may be arbitrary hashable values. Treat a missing adjacency entry as a node with no outgoing edges. Handle start == goal, cycles, and nodes reached more than once.
Include runnable assertions for a shortest path, an unreachable goal, start == goal, and a cyclic graph. Put all code in one Python code block, followed by a brief explanation of time and space complexity. Keep the implementation concise enough to finish.""",
    "heldout-chinese-transactions": """请用中文给初学者解释数据库事务隔离。先简要说明事务和提交，再分别解释脏读、不可重复读、幻读的区别。
用两个并发事务操作银行账户余额及查询“余额大于100元的账户列表”的小例子，写清楚谁先读、谁更新或插入、谁提交，以及另一个事务观察到什么。
最后概括读已提交、可重复读、可串行化通常要避免哪些现象，注明具体数据库的实现可能不同。约四百字，连贯说明，避免只列术语。""",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8096")
    parser.add_argument("--out", type=Path, default=HERE / "heldout")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--describe-only", action="store_true", help="Print prompts; no service access")
    args = parser.parse_args()
    if min(args.repeats, args.max_tokens) < 1:
        parser.error("repeat and output counts must be positive")
    if args.describe_only:
        print(json.dumps({"prompts": PROMPTS, "repeats": args.repeats, "max_tokens": args.max_tokens,
                          "history": "Separate service process after the 30-request run; no request reloads"}, ensure_ascii=False, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    if any((args.out / name).exists() for name in ("results.json", "plan.json", "case-summary.json")):
        raise FileExistsError("use a new output directory; existing held-out run is preserved")
    client = Client(args.url, args.timeout)
    health = client.json("/health")
    if not health.get("loaded"):
        raise RuntimeError("service must already be ready; this runner does not reload it")
    model = client.json("/v1/models")["data"][0]["id"]
    cases = [natural_case(client, model, name, text, args.max_tokens) for name, text in PROMPTS.items()]
    training = json.loads((HERE.parent / "2026-10-14-a3b-q8-training/training-cases.json").read_text(encoding="utf-8"))
    assert all(text not in [entry[1] for entry in training] for text in PROMPTS.values())
    config_path = ROOT / "strata-a3b-q8.json"
    plan = {"model": model, "url": args.url, "health_before": health, "cases": cases,
            "repeats": args.repeats, "max_tokens": args.max_tokens, "seed": args.seed,
            "temperature": 0, "thinking": False,
            "process_history": "Separately restarted production service after the 30-request candidate run; first BFS trial is its first measured request; later requests share that process.",
            "measurement_code": "bench_a3b_latency.stream_request/summarize_cases",
            "production_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "scope": "Two new prompt texts absent from the five profile training prompts; limited workload check, not a broad quality/generalization benchmark. Do not pool with the earlier process's 30 requests."}
    (args.out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    results = []
    for case_index, case in enumerate(cases):
        for repeat in range(1, args.repeats + 1):
            trial = f"{case['case_id']}-repeat{repeat:02d}"
            request = {"model": model, "messages": case["messages"], "max_tokens": args.max_tokens,
                       "temperature": 0, "seed": args.seed, "stream": True,
                       "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False}}
            result = stream_request(client, request, args.out, trial)
            result.update({"trial": trial, "case_id": case["case_id"], "repeat": repeat,
                           "cache_state": "fresh-service-first-measured-request" if case_index == 0 and repeat == 1 else
                                          ("resident-new-case" if repeat == 1 else "resident-repeat"),
                           "temperature": 0, "target_prompt_tokens": case["target_prompt_tokens"],
                           "count_endpoint_prompt_tokens": case["count_endpoint_prompt_tokens"],
                           "requested_max_tokens": args.max_tokens, "resident_process": True,
                           "process_cohort": "heldout-separate-production-start"})
            result["count_endpoint_matches_actual_usage"] = result["usage"]["prompt_tokens"] == case["count_endpoint_prompt_tokens"]
            results.append(result)
            summary = {}
            for name in PROMPTS:
                rows = [row for row in results if row["case_id"] == name]
                if rows:
                    summary[name] = next(iter(summarize_cases(rows).values()))
            (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (args.out / "case-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({key: value for key, value in result.items() if key != "rate_note"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
