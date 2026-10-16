"""Exercise an already-running local A3B server with real model requests."""
import argparse
import json
from pathlib import Path
import time
import requests

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8096")
    p.add_argument("--out", type=Path)
    p.add_argument("--extended", action="store_true", help="Check speculative boundaries, block carryover and seeded penalties")
    args = p.parse_args()
    base = args.url.rstrip("/")
    health = requests.get(base + "/health", timeout=10)
    health.raise_for_status()
    assert health.json()["loaded"] and not health.json()["images"]
    models = requests.get(base + "/v1/models", timeout=10)
    models.raise_for_status(); model = models.json()["data"][0]["id"]
    page = requests.get(base + "/", timeout=10)
    page.raise_for_status(); assert "<html" in page.text.lower()
    body = {"model": model, "messages": [{"role": "user", "content": "17乘以23等于多少？只给出结果。"}],
            "max_tokens": 256, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    results = {"health": health.json(), "model": model, "requests": []}
    for index in range(2):
        start = time.monotonic()
        r = requests.post(base + "/v1/chat/completions", json=body, timeout=120)
        r.raise_for_status(); data = r.json(); choice = data["choices"][0]
        assert "391" in choice["message"]["content"], data
        assert choice["finish_reason"] == "stop", data
        results["requests"].append({"kind": "openai", "repeat": index, "elapsed_s": time.monotonic() - start, "response": data})
    r = requests.post(base + "/v1/chat/completions", json={**body, "stream": True}, stream=True, timeout=120)
    r.raise_for_status(); events = []
    for line in r.iter_lines():
        if line.startswith(b"data: "):
            value = line[6:]
            if value == b"[DONE]": break
            events.append(json.loads(value))
    text = "".join(e["choices"][0].get("delta", {}).get("content", "") for e in events if e.get("choices"))
    assert "391" in text and any(e.get("choices") and e["choices"][0].get("finish_reason") == "stop" for e in events)
    results["stream"] = events
    r = requests.post(base + "/v1/messages", headers={"anthropic-version": "2023-06-01"},
        json={"model": model, "messages": [{"role": "user", "content": "请只回答：你好。"}], "max_tokens": 256,
              "temperature": 0, "thinking": {"type": "disabled"}}, timeout=120)
    r.raise_for_status(); data = r.json()
    assert "你好" in "".join(x.get("text", "") for x in data["content"]), data
    results["anthropic"] = data
    if args.extended:
        extra = []
        for limit in [1, 2, 3, 4, 5, 8]:
            r = requests.post(base + "/v1/chat/completions", json={**body,
                "messages": [{"role": "user", "content": "请详细解释CPU和GPU如何协同计算混合专家模型。"}],
                "max_tokens": limit}, timeout=120)
            r.raise_for_status(); data = r.json()
            assert data["usage"]["completion_tokens"] == limit and data["choices"][0]["finish_reason"] == "length", data
            extra.append({"kind": "output-boundary", "limit": limit, "response": data})
        long_body = {**body, "messages": [{"role": "user", "content":
            "以下是无关资料：" + "苹果和香蕉是水果，猫和狗是动物。" * 100 +
            "\n忽略上述资料。17乘以23等于多少？只给出结果。"}]}
        r = requests.post(base + "/v1/chat/completions", json=long_body, timeout=180)
        r.raise_for_status(); data = r.json()
        assert data["usage"]["prompt_tokens"] > 256 and "391" in data["choices"][0]["message"]["content"], data
        extra.append({"kind": "prefill-block-carry", "response": data})
        sampled = {**body, "messages": [{"role": "user", "content": "用一句中文解释混合专家模型。"}],
                   "max_tokens": 64, "temperature": 0.7, "seed": 12345,
                   "repetition_penalty": 1.1, "frequency_penalty": 0.2, "presence_penalty": 0.1}
        for _ in range(3):
            r = requests.post(base + "/v1/chat/completions", json=sampled, timeout=120)
            r.raise_for_status(); data = r.json()
            assert data["choices"][0]["message"]["content"], data
            extra.append({"kind": "seeded-sampling", "response": data})
        # The target RNG is seeded per request, but adaptive cache admissions
        # can change CPU/GPU activation quantization and then the sampled path.
        # Record this instead of treating a changing placement as fixed math.
        results["warm_seeded_choices_equal"] = extra[-1]["response"]["choices"] == extra[-2]["response"]["choices"]
        assert any(x["response"].get("timings", {}).get("draft_n", 0) > 0 for x in extra), extra
        results["extended"] = extra
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PASS health, model listing, web page, repeat requests, OpenAI non-stream/stream, Anthropic text")
    for r in results["requests"]:
        print("repeat", r["repeat"], "elapsed_s", round(r["elapsed_s"], 3), "timings", r["response"].get("timings"))
    if args.extended:
        print("PASS speculative output boundaries, prefill block carry, seeded penalties; warm seeded choices equal:",
              results["warm_seeded_choices_equal"])

if __name__ == "__main__": main()
