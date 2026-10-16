"""Measure real streaming API latency without counting role/progress events as tokens.

Default runs retain the resident engine and label its initial cache state unknown.
Explicit cold/hot policies unload and reload only the selected server's model.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import statistics
import time
from urllib.parse import urlsplit


class _ObservedRaw(io.RawIOBase):
    def __init__(self, raw, observation):
        super().__init__()
        self.raw, self.observation = raw, observation

    def readable(self):
        return True

    def readinto(self, data):
        n = self.raw.readinto(data)
        if n and self.observation.get("first_response_byte_at") is None:
            self.observation["first_response_byte_at"] = time.perf_counter()
        return n

    def fileno(self):
        return self.raw.fileno()

    def close(self):
        if not self.closed:
            self.raw.close()
        super().close()


class _SocketProxy:
    def __init__(self, sock, observation):
        self.sock, self.observation = sock, observation

    def makefile(self, mode):
        # Observe the first successful socket read, before status/header parsing.
        raw = self.sock.makefile(mode, buffering=0)
        return io.BufferedReader(_ObservedRaw(raw, self.observation))


class _TimedResponse(http.client.HTTPResponse):
    def __init__(self, observation, sock, *args, **kwargs):
        super().__init__(_SocketProxy(sock, observation), *args, **kwargs)


class Client:
    def __init__(self, url, timeout):
        self.url = url.rstrip("/")
        self.parts = urlsplit(self.url)
        if self.parts.scheme not in ("http", "https") or not self.parts.hostname:
            raise ValueError("--url must be an http(s) URL")
        self.timeout = timeout

    def request(self, path, body=None, timed=False):
        cls = http.client.HTTPSConnection if self.parts.scheme == "https" else http.client.HTTPConnection
        connection = cls(self.parts.hostname, self.parts.port, timeout=self.timeout)
        observation = {}
        if timed:
            connection.response_class = partial(_TimedResponse, observation)
        headers = {"Content-Type": "application/json", "Origin": self.url,
                   "Accept": "text/event-stream" if timed else "application/json"}
        if key := os.environ.get("STRATA_API_KEY"):
            headers["Authorization"] = "Bearer " + key
        payload = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        observation["request_started_at"] = time.perf_counter()
        connection.request("POST" if body is not None else "GET", (self.parts.path.rstrip("/") + path),
                           body=payload, headers=headers)
        response = connection.getresponse()
        observation["headers_received_at"] = time.perf_counter()
        if response.status != 200:
            detail = response.read().decode("utf-8", "replace")[:2000]
            connection.close()
            raise RuntimeError(f"HTTP {response.status} at {path}: {detail}")
        return connection, response, observation

    def json(self, path, body=None):
        connection, response, _ = self.request(path, body)
        try:
            return json.loads(response.read())
        finally:
            response.close()
            connection.close()

    def reload(self, model):
        began = time.perf_counter()
        unloaded = self.json("/v1/unload", {"model": model})
        if unloaded.get("status") == "busy":
            raise RuntimeError("server is busy; cold-cache reset did not occur")
        loaded = self.json("/v1/load", {"model": model})
        return {"unload": unloaded, "load": loaded, "elapsed_s": time.perf_counter() - began}


def prepare_prompt(client, model, target, thinking):
    instruction = "详细解释MoE，写满千字。"
    material = ("模型由路由器、专家与共享网络组成。路由器根据输入选择专家，再合并专家的结果。" * (target + 1))
    counts = {}

    def content(n):
        return ("资料：" + material[:n] + "\n" if n else "") + instruction

    def count(n):
        if n not in counts:
            body = {"model": model, "messages": [{"role": "user", "content": content(n)}],
                    "thinking": {"type": "enabled" if thinking else "disabled"}}
            counts[n] = int(client.json("/v1/messages/count_tokens", body)["input_tokens"])
        return counts[n]

    lo, hi = 0, min(len(material), target * 4)
    while count(hi) < target and hi < len(material):
        hi = min(len(material), hi * 2)
    while lo < hi:
        mid = (lo + hi) // 2
        if count(mid) < target:
            lo = mid + 1
        else:
            hi = mid
    for n in range(max(0, lo - 12), min(len(material), lo + 12) + 1):
        count(n)
    best = min(counts, key=lambda n: (abs(counts[n] - target), n))
    return {"target_prompt_tokens": target, "count_endpoint_prompt_tokens": counts[best],
            "messages": [{"role": "user", "content": content(best)}],
            "count_source": "/v1/messages/count_tokens; actual OpenAI usage is authoritative"}


def stream_request(client, body, out, trial, on_first_model_delta=None):
    connection, response, observation = client.request("/v1/chat/completions", body, timed=True)
    started = observation["request_started_at"]
    relative = lambda moment: None if moment is None else moment - started
    usage = timings = None
    first_body = first_event = first_any = first_visible = first_reasoning = last_any = last_visible = None
    visible, reasoning, buffer = [], [], b""
    finish = None
    event_count = data_events = nonempty_events = 0
    received_done = False

    def frame(raw, now, event_file):
        nonlocal usage, timings, first_event, first_any, first_visible, first_reasoning
        nonlocal last_any, last_visible, finish, data_events, nonempty_events, received_done
        if first_event is None:
            first_event = now
        data = b"\n".join(line[5:].lstrip(b" ") for line in raw.splitlines() if line.startswith(b"data:"))
        record = {"at_s": relative(now), "sse": raw.decode("utf-8", "replace")}
        if not data:
            record["kind"] = "comment_or_empty"
        elif data == b"[DONE]":
            received_done = True
            record["kind"] = "done"
        else:
            item = json.loads(data)
            if item.get("error"):
                raise RuntimeError(f"SSE error: {item['error']}")
            record["kind"] = "data"
            data_events += 1
            if isinstance(item.get("usage"), dict):
                usage = item["usage"]
            if isinstance(item.get("timings"), dict):
                timings = item["timings"]
            any_text = False
            for choice in item.get("choices") or []:
                if choice.get("finish_reason") is not None:
                    finish = choice["finish_reason"]
                delta = choice.get("delta") or {}
                text = delta.get("content")
                thought = delta.get("reasoning_content") or delta.get("reasoning")
                if isinstance(text, str) and text:
                    visible.append(text)
                    first_visible = first_visible if first_visible is not None else now
                    last_visible = now
                    any_text = True
                if isinstance(thought, str) and thought:
                    reasoning.append(thought)
                    first_reasoning = first_reasoning if first_reasoning is not None else now
                    any_text = True
            if any_text:
                if first_any is None:
                    first_any = now
                    if on_first_model_delta is not None:
                        on_first_model_delta(now)
                last_any = now
                nonempty_events += 1
        event_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    try:
        with (out / (trial + ".events.jsonl")).open("w", encoding="utf-8") as event_file:
            while True:
                # read1 returns currently available body bytes; no iter_lines(512) buffering.
                chunk = response.read1(65536)
                now = time.perf_counter()
                if not chunk:
                    break
                if first_body is None:
                    first_body = now
                buffer += chunk
                buffer = buffer.replace(b"\r\n", b"\n")
                while b"\n\n" in buffer:
                    raw, buffer = buffer.split(b"\n\n", 1)
                    event_count += 1
                    frame(raw, now, event_file)
                if received_done:
                    break
            if buffer.strip():
                frame(buffer, time.perf_counter(), event_file)
    finally:
        response.close()
        connection.close()
    completed = time.perf_counter()
    if not received_done or usage is None:
        raise RuntimeError("stream lacks [DONE] or usage; cannot compute token rate")
    count = int(usage.get("completion_tokens") or 0)
    interval = last_any - first_any if first_any is not None and last_any is not None else 0
    visible_interval = last_visible - first_visible if first_visible is not None and last_visible is not None else 0
    result = {"http_response_first_byte_s": relative(observation.get("first_response_byte_at")),
              "http_headers_received_s": relative(observation["headers_received_at"]),
              "body_first_bytes_s": relative(first_body), "first_sse_event_s": relative(first_event),
              "ttft_any_content_or_reasoning_s": relative(first_any),
              "ttft_visible_content_s": relative(first_visible), "ttft_reasoning_s": relative(first_reasoning),
              "last_any_content_or_reasoning_s": relative(last_any), "last_visible_content_s": relative(last_visible),
              "total_response_s": completed - started,
              "client_usage_decode_tokens_per_s": (count - 1) / interval if count > 1 and interval > 0 else None,
              "client_visible_usage_decode_tokens_per_s": (count - 1) / visible_interval if not reasoning and count > 1 and visible_interval > 0 else None,
              "client_completion_tokens_per_total_second": count / (completed - started),
              "engine_predicted_tokens_per_s": (timings or {}).get("predicted_per_second"),
              "engine_prompt_tokens_per_s": (timings or {}).get("prompt_per_second"),
              "usage": usage, "timings": timings, "finish_reason": finish, "received_done": received_done,
              "sse_frames": event_count, "data_events": data_events, "nonempty_text_events": nonempty_events,
              "visible_characters": sum(map(len, visible)), "reasoning_characters": sum(map(len, reasoning)),
              "request_sha256": hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
              "rate_note": "Client rate is (API completion_tokens - 1)/(last nonempty delta - first nonempty delta). SSE deltas need not equal single model tokens; MTP can emit bursts, and usage may include hidden/stop tokens. Visible-only rate is null if reasoning was emitted."}
    (out / (trial + ".output.json")).write_text(json.dumps({"content": "".join(visible), "reasoning": "".join(reasoning)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


METRIC_FIELDS = ["http_response_first_byte_s", "http_headers_received_s", "body_first_bytes_s",
                 "ttft_any_content_or_reasoning_s", "ttft_visible_content_s", "engine_prompt_tokens_per_s",
                 "engine_predicted_tokens_per_s", "client_usage_decode_tokens_per_s", "total_response_s"]


def metric_summary(items):
    return {"runs": len(items), **{field + "_median": statistics.median(values)
            for field in METRIC_FIELDS if (values := [x[field] for x in items if x.get(field) is not None])}}


def summarize(results):
    groups = {}
    for result in results:
        key = f"prompt{result['target_prompt_tokens']}-output{result['requested_max_tokens']}-{result['cache_state']}"
        groups.setdefault(key, []).append(result)
    return {key: metric_summary(items) for key, items in groups.items()}


def summarize_cases(results):
    """Keep all resident trials together, with the first and later trials visible.

    No warm-up inference is silently excluded. Three resident repeats therefore
    mean three values in overall and only two values in subsequent_runs.
    """
    groups = {}
    for result in results:
        key = f"prompt{result['target_prompt_tokens']}-output{result['requested_max_tokens']}"
        groups.setdefault(key, []).append(result)
    report = {}
    for key, items in groups.items():
        trials = []
        for item in items:
            timings = item.get("timings") or {}
            offered, accepted = timings.get("draft_n"), timings.get("draft_n_accepted")
            trials.append({"trial": item["trial"], "cache_state": item["cache_state"],
                           "actual_prompt_tokens": item["usage"].get("prompt_tokens"),
                           "actual_completion_tokens": item["usage"].get("completion_tokens"),
                           "finish_reason": item["finish_reason"],
                           "draft_tokens_offered": offered, "draft_tokens_accepted": accepted,
                           "draft_acceptance_rate": accepted / offered if offered else None,
                           **{field: item.get(field) for field in METRIC_FIELDS}})
        report[key] = {"overall": metric_summary(items), "first_run": trials[0],
                       "subsequent_runs": metric_summary(items[1:]), "trials": trials,
                       "all_prompt_counts_match": all(x["count_endpoint_matches_actual_usage"] for x in items),
                       "note": "Overall includes every measured trial. First_run is shown separately; subsequent_runs is the remaining trials, with its actual run count. Resident cache labels describe request order and do not imply a reset or cold disk."}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Full server URL; overrides --port")
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prompt-tokens", type=int, nargs="+", default=[32, 512, 2048])
    parser.add_argument("--max-tokens", type=int, nargs="+", default=[256, 512])
    parser.add_argument("--cache-policy", choices=["resident", "reload-case", "cold-hot-pairs"], default="resident",
                        help="resident retains unknown initial cache; reload-case reloads once per case; cold-hot-pairs reloads before each repeated cold/hot pair")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--label", default="a3b-api")
    parser.add_argument("--prepare-only", action="store_true", help="Count tokens and save cases without model inference or reload")
    args = parser.parse_args()
    if args.repeats < 1 or min(args.prompt_tokens + args.max_tokens) < 1:
        parser.error("repeat and token counts must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    client = Client(args.url or f"http://127.0.0.1:{args.port}", args.timeout)
    models = client.json("/v1/models")
    model = models["data"][0]["id"]
    cases = [prepare_prompt(client, model, n, args.thinking) for n in args.prompt_tokens]
    metadata = {"label": args.label, "url": client.url, "model": model, "cache_policy": args.cache_policy,
                "repeats": args.repeats, "thinking": args.thinking, "temperature": args.temperature,
                "max_tokens": args.max_tokens, "cases": cases,
                "timing_definitions": {"http_response_first_byte": "first successful underlying socket read, including status and headers",
                    "http_headers_received": "getresponse finished parsing HTTP headers",
                    "body_first_bytes": "first nonempty HTTPResponse.read1 body read",
                    "ttft_any": "first SSE delta with nonempty content or reasoning text, excluding role/progress/comments",
                    "ttft_visible": "first SSE delta with nonempty visible content",
                    "engine": "final SSE timings and usage from the engine, excluding model load as defined by the server",
                    "client_decode": "(completion_tokens - 1)/(last nonempty any-text delta - first nonempty any-text delta), subject to MTP/detokenizer burst granularity"}}
    (args.out / "plan.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.prepare_only:
        print(json.dumps({"prepared_cases": [{k: v for k, v in c.items() if k != "messages"} for c in cases]}, ensure_ascii=False), flush=True)
        return
    results = []
    for case in cases:
        for max_tokens in args.max_tokens:
            reset = client.reload(model) if args.cache_policy == "reload-case" else None
            for repeat in range(1, args.repeats + 1):
                states = ["resident-first-unknown" if repeat == 1 else "resident-repeat"]
                if args.cache_policy == "reload-case":
                    states = ["cold-experts-after-reload" if repeat == 1 else "warm-experts-repeat"]
                elif args.cache_policy == "cold-hot-pairs":
                    reset = client.reload(model)
                    states = ["cold-experts-after-reload", "warm-experts-repeat"]
                for state in states:
                    trial = f"prompt{case['target_prompt_tokens']}-output{max_tokens}-repeat{repeat:02d}-{state}"
                    body = {"model": model, "messages": case["messages"], "max_tokens": max_tokens,
                            "temperature": args.temperature, "seed": args.seed, "stream": True,
                            "stream_options": {"include_usage": True},
                            "chat_template_kwargs": {"enable_thinking": args.thinking}}
                    result = stream_request(client, body, args.out, trial)
                    result.update({"trial": trial, "repeat": repeat, "cache_state": state,
                                   "target_prompt_tokens": case["target_prompt_tokens"],
                                   "count_endpoint_prompt_tokens": case["count_endpoint_prompt_tokens"],
                                   "requested_max_tokens": max_tokens, "model_reset": reset if state.startswith("cold") else None,
                                   "resident_process": True})
                    result["count_endpoint_matches_actual_usage"] = result["usage"]["prompt_tokens"] == case["count_endpoint_prompt_tokens"]
                    results.append(result)
                    (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    (args.out / "summary.json").write_text(json.dumps(summarize(results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    (args.out / "case-summary.json").write_text(json.dumps(summarize_cases(results), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    print(json.dumps({k: v for k, v in result.items() if k not in ("model_reset", "rate_note")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
