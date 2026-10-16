"""32K x 3, long outputs; common-window rates from actual engine token counters."""
from pathlib import Path
import argparse
import concurrent.futures
import datetime
import json
import threading
import time
import tuning_common as b

BASE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, default=BASE / "steady32-output1024")
parser.add_argument("--config", type=Path, default=b.ROOT / "strata-iq3_s.json")
parser.add_argument("--reuse-inputs", type=Path)
parser.add_argument("--expected-resident", choices=("on", "off"), default="off")
parser.add_argument("--no-restore", action="store_true")
options = parser.parse_args()
OUT = options.output_dir
OUT.mkdir(exist_ok=True)
b.OUT = OUT
b.STOP = threading.Event()
b.ABORT = threading.Event()
b.START = datetime.datetime.now().astimezone()
b.STATE = {"phase": "calibration", "active": False, "last_progress": time.monotonic(), "signature": None}
health = b.http_json(b.URL, "/health")
slots = b.http_json(b.URL, "/slots")
assert health["max_context"] == 262144 and len(slots) == 3
assert all(s["n_ctx"] == 262144 and not s["is_processing"] for s in slots)
config = json.loads(options.config.read_text())
assert ("--resident-experts" in config["args"]) == (options.expected_resident == "on")
b.RESULT = {"status": "preparing", "stages": [], "config": config, "health": health, "slots": slots,
            "output_cap": 1024, "input_target": 32768, "context_limit": 262144}

instruction = ("\n\nWrite an extensive numbered catalog of 500 concrete observations inspired by the words above. "
               "Each numbered entry must have at least two complete sentences. Start directly with entry 1. "
               "Keep writing entries without a summary, conclusion, or early stopping.")
prompts = []
for tag in "ABC":
    if options.reuse_inputs:
        prompt = (options.reuse_inputs / f"input-{tag}.txt").read_text()
        n = b.count(prompt)
        assert abs(n - 32768) <= 1, n
        prompts.append({"tag": tag, "prompt": prompt, "input_tokens": n})
        (OUT / f"input-{tag}.txt").write_text(prompt)
        print(f"REUSED IDENTICAL INPUT {tag}: {n} tokens", flush=True)
        continue
    original = (BASE / "inputs" / f"input32768-{tag}.txt").read_text()
    words = original.split("Here is a list of words:\n", 1)[1].split("\n\nSummarize", 1)[0].split()
    prefix = f"Independent steady output benchmark {tag} {int(time.time())}.\nHere is a list of words:\n"
    for _ in range(20):
        prompt = prefix + " ".join(words) + instruction
        n = b.count(prompt)
        if abs(n - 32768) <= 1:
            break
        if n > 32768:
            words = words[:-(n - 32768)]
        else:
            words.extend(["stone"] * (32768 - n))
    assert abs(n - 32768) <= 1, n
    prompts.append({"tag": tag, "prompt": prompt, "input_tokens": n})
    (OUT / f"input-{tag}.txt").write_text(prompt)
    print(f"CALIBRATED {tag}: {n} tokens", flush=True)

samples = []
sample_stop = threading.Event()

def sample():
    while not sample_stop.is_set():
        try:
            start = time.perf_counter()
            live = b.http_json(b.URL, "/metrics", timeout=2)["live"]
            row = {"t": (start + time.perf_counter()) / 2, "live": live}
            samples.append(row)
            b.append("token-counters.jsonl", row)
        except Exception as e:
            b.append("counter-errors.jsonl", {"t": time.perf_counter(), "error": repr(e)})
        sample_stop.wait(0.25)

barrier = threading.Barrier(3)
def one(p):
    barrier.wait()
    start = time.perf_counter()
    r = b.chat(b.URL, "", [{"role": "user", "content": p["prompt"]}], 1024, timeout=600)
    r.update(tag=p["tag"], client_start=start, client_end=time.perf_counter())
    b.append("requests.jsonl", r)
    return r

watcher = threading.Thread(target=b.monitor, daemon=True)
counter = threading.Thread(target=sample, daemon=True)
watcher.start()
counter.start()
try:
    b.STATE.update(phase="32kx3-output1024", active=True, last_progress=time.monotonic(), signature=None)
    print("TEST 32K x 3, output cap 1024 each", flush=True)
    begin = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        requests = list(pool.map(one, prompts))
    wall = time.perf_counter() - begin
    b.STATE["active"] = False
    assert not b.ABORT.is_set(), b.RESULT.get("abort_reason")
    assert all(not r.get("err") for r in requests), [r.get("err") for r in requests]
    assert all(r["usage"]["completion_tokens"] == 1024 for r in requests)
    assert all(r["usage"]["prompt_tokens_details"]["cached_tokens"] == 0 for r in requests)
    common = [s for s in samples if len(s["live"].get("slots", [])) == 3
              and all(q["state"] == "decoding" for q in s["live"]["slots"])]
    assert len(common) > 20, "insufficient common decoding window"
    # Exclude admission transients and the final second near request completion.
    interior = [s for s in common if common[0]["t"] + 2 <= s["t"] <= common[-1]["t"] - 1]
    first, last = interior[0], interior[-1]
    span = last["t"] - first["t"]
    assert span >= 10, span
    def counts(s):
        return {q["slot"]: q["generated"] for q in s["live"]["slots"]}
    left, right = counts(first), counts(last)
    rates = {str(k): (right[k] - left[k]) / span for k in left}
    summary = {"status": "complete", "wall_s": wall, "output_tokens": sum(r["usage"]["completion_tokens"] for r in requests),
               "end_to_end_output_tok_s": sum(r["usage"]["completion_tokens"] for r in requests) / wall,
               "common_window_s": span, "common_start_since_submission_s": first["t"] - begin,
               "common_end_since_submission_s": last["t"] - begin,
               "common_start_tokens": left, "common_end_tokens": right,
               "common_per_slot_tok_s": rates, "common_total_tok_s": sum(rates.values()),
               "requests": requests, "method": "actual engine generated-token counter differences over one shared interval; excludes all prompt processing"}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    b.RESULT.update(status="complete", summary=summary)
    print(json.dumps({k:v for k,v in summary.items() if k != "requests"}, ensure_ascii=False), flush=True)
except Exception as e:
    b.RESULT.update(status="failed", error=repr(e))
    print("FAILED: " + repr(e), flush=True)
finally:
    b.STATE["active"] = False
    sample_stop.set()
    b.STOP.set()
    counter.join(timeout=5)
    watcher.join(timeout=12)
    b.save()
    if b.ABORT.is_set() and not options.no_restore:
        # Restore the same authorized main configuration if the watchdog stopped it.
        source = (BASE / "tune.py").read_text()
        ns = {"__file__": str(BASE / "tune.py")}
        exec(compile(source.split("\ntry:\n", 1)[0], "restore_definitions", "exec"), ns)
        ns["stop_current"]()
        ns["start"](config, b.ROOT / "strata-iq3_s.json", "steady32-restore-server.log")
        ns["verify"]()
