"""One deterministic 50K-ish middle-needle retrieval on the selected IQ3_S service."""
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / "q4-trial1" / "plan.json").read_text(encoding="utf-8"))
text = plan["cases"][0]["messages"][0]["content"]
text = text.rsplit("\n", 1)[0]
needle = "特别记录：秘密编号是海蓝-73192。"
middle = len(text) // 2
text = text[:middle] + needle + text[middle:] + "\n请只输出资料中的秘密编号，不要解释。"

def post(path, body):
    request = urllib.request.Request("http://127.0.0.1:8096" + path,
                                     data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)

messages = [{"role": "user", "content": text}]
count = post("/v1/messages/count_tokens", {"model": "qwen3.6-35b-a3b-ud-iq3_s", "messages": messages,
                                            "thinking": {"type": "disabled"}})
start = time.perf_counter()
answer = post("/v1/chat/completions", {"model": "qwen3.6-35b-a3b-ud-iq3_s", "messages": messages,
                                       "max_tokens": 128, "temperature": 0, "seed": 12345,
                                       "stream": False, "chat_template_kwargs": {"enable_thinking": False}})
elapsed = time.perf_counter() - start
message = answer["choices"][0]["message"]
content = message.get("content") or ""
result = {"count_endpoint": count, "usage": answer.get("usage"), "elapsed_s": elapsed,
          "needle": "海蓝-73192", "answer": content, "reasoning": message.get("reasoning_content"),
          "passed": "海蓝-73192" in content}
(ROOT / "needle-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
if not result["passed"]:
    raise SystemExit(1)

