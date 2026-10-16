"""Near-64K concurrent input test against the existing local Strata service."""
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

import psutil

ROOT = Path(r"G:\Strata")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from strata_bench import chat, http_json
from ab_engine import long_prompt

URL = "http://127.0.0.1:8080"
START = datetime.datetime.now().astimezone()
STOP = threading.Event()
ABORT = threading.Event()
LOCK = threading.Lock()
STATE = {"phase": "calibration", "active": False, "last_progress": time.monotonic(), "signature": None}
RESULT = {"started": START.isoformat(), "driver": "617.42", "stages": [], "status": "calibrating",
          "input_target": 65400, "output_cap": 128, "context_limit": 65536,
          "context_slack": 8, "synthetic_word_list": True}
owner = next((c.pid for c in psutil.net_connections(kind="tcp")
              if c.status == psutil.CONN_LISTEN and c.laddr.port == 8080 and c.laddr.ip == "127.0.0.1"), None)
SERVER = psutil.Process(owner) if owner else None
if SERVER is not None:
    assert any("Strata" in a and "server.py" in a for a in SERVER.cmdline()), SERVER.cmdline()
RESULT["server_pid"] = owner


def save():
    with LOCK:
        (OUT / "results.json").write_text(json.dumps(RESULT, ensure_ascii=False, indent=2), encoding="utf-8")


def append(name, obj):
    with LOCK, (OUT / name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def stop_server():
    try:
        processes = SERVER.children(recursive=True) + [SERVER]
        for p in processes:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(processes, timeout=4)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


def abort(reason):
    if ABORT.is_set():
        return
    ABORT.set()
    RESULT.update(status="aborted", abort_reason=reason)
    append("alerts.jsonl", {"time": time.time(), "reason": reason})
    save()
    print("ABORT " + reason, flush=True)
    stop_server()


def count(prompt):
    data = {"model": "strata", "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"}, "max_tokens": 128}
    req = urllib.request.Request(URL + "/v1/messages/count_tokens", data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)["input_tokens"]


def calibrate(tag, target):
    lo, hi = 1, target * 2
    words = int(target * 3800 / 4037)
    best = None
    for attempt in range(24):
        text = f"Fresh long-context trial {tag}.\n" + long_prompt(words)
        n = count(text)
        append("calibration.jsonl", {"tag": tag, "words": words, "input_tokens": n})
        if n <= target:
            best = (text, n, words)
            lo = words + 1
        else:
            hi = words - 1
        if n <= target and target - n <= 2:
            break
        if hi < lo:
            break
        words = (lo + hi) // 2
    assert best and target - best[1] <= 3, (tag, target, best[1] if best else None)
    (OUT / (tag + ".txt")).write_text(best[0], encoding="utf-8")
    print(f"CALIBRATED {tag}: {best[1]} input tokens", flush=True)
    return {"tag": tag, "prompt": best[0], "input_tokens": best[1], "words": best[2]}


def monitor():
    start_text = START.strftime("%Y-%m-%dT%H:%M:%S")
    event_cmd = ("Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='nvlddmkm';"
                 f"StartTime=[datetime]'{start_text}'" + "} -ErrorAction SilentlyContinue | "
                 "ForEach-Object { [pscustomobject]@{id=$_.Id;record=$_.RecordId;"
                 "time=$_.TimeCreated.ToString('o');xml=$_.ToXml()} } | ConvertTo-Json -Compress")
    last_event = 0
    while not STOP.is_set():
        now = time.time()
        vm = psutil.virtual_memory()
        entry = {"time": now, "phase": STATE["phase"], "available_ram_gib": vm.available / 2**30}
        if vm.available < 8 * 2**30:
            append("telemetry.jsonl", entry)
            abort("available RAM below 8 GiB")
            return
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if q.returncode:
                raise RuntimeError(q.stderr)
            cards = []
            for line in q.stdout.strip().splitlines():
                x = [v.strip() for v in line.split(",")]
                cards.append({"index": int(x[0]), "used_mib": int(x[1]), "total_mib": int(x[2]),
                              "temperature_c": int(x[3]), "utilization": x[4], "power_w": x[5]})
            entry["gpus"] = cards
            if any(g["temperature_c"] >= 90 for g in cards):
                abort("GPU temperature reached 90 C")
                return
            if any(g["total_mib"] - g["used_mib"] < 256 for g in cards):
                abort("free VRAM below 256 MiB")
                return
        except Exception as e:
            append("telemetry.jsonl", {**entry, "gpu_query_error": repr(e)})
            abort("GPU query failed or timed out: " + repr(e))
            return
        if now - last_event > 4:
            last_event = now
            try:
                q = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", event_cmd],
                                   capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=6)
                if q.stdout.strip():
                    events = json.loads(q.stdout)
                    events = events if isinstance(events, list) else [events]
                    append("events.jsonl", {"time": now, "events": events})
                    if any(e["id"] in (13, 14, 153) for e in events):
                        abort("new NVIDIA driver error")
                        return
            except Exception as e:
                entry["event_query_error"] = repr(e)
        try:
            live = http_json(URL, "/metrics", timeout=2).get("live", {})
            entry["live"] = live
            progress = {k: live.get(k) for k in ("state", "phase", "prompt_read", "prompt_total", "generated")}
            progress["slots"] = [{k: s.get(k) for k in ("slot", "state", "prompt_tokens", "generated", "held_tokens")}
                                 for s in live.get("slots", [])]
            signature = json.dumps(progress, sort_keys=True)
            if signature != STATE["signature"]:
                STATE.update(signature=signature, last_progress=time.monotonic())
            RESULT["max_observed_running"] = max(RESULT.get("max_observed_running", 0), live.get("running") or 0)
        except Exception as e:
            entry["metrics_error"] = repr(e)
        append("telemetry.jsonl", entry)
        if STATE["active"] and time.monotonic() - STATE["last_progress"] > 45:
            abort("no reported prompt/decode progress for 45 seconds")
            return
        STOP.wait(1)


def run(name, prompts):
    if ABORT.is_set():
        raise RuntimeError(RESULT.get("abort_reason"))
    STATE.update(phase=name, active=True, last_progress=time.monotonic(), signature=None)
    RESULT["status"] = name
    save()
    print(f"TEST {name}, {len(prompts)} requests", flush=True)
    barrier = threading.Barrier(len(prompts))

    def one(p):
        barrier.wait()
        r = chat(URL, "", [{"role": "user", "content": p["prompt"]}], 128, timeout=900)
        append("requests.jsonl", {"stage": name, "tag": p["tag"], "calibrated_input_tokens": p["input_tokens"], **r})
        return {"tag": p["tag"], "calibrated_input_tokens": p["input_tokens"], **r}

    begin = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        requests = list(pool.map(one, prompts))
    wall = time.perf_counter() - begin
    STATE["active"] = False
    stage = {"name": name, "requests": requests, "wall_s": wall}
    RESULT["stages"].append(stage)
    save()
    if ABORT.is_set() or any(r.get("err") for r in requests):
        raise RuntimeError("request failed or watchdog aborted: " + str([r.get("err") for r in requests]))
    assert all((r.get("usage") or {}).get("prompt_tokens", 0) >= 16000 for r in requests)
    assert all(r.get("usage") and r.get("timings") for r in requests)
    print(f"DONE {name}: {wall:.1f} s; "
          f"decode {[r['timings'].get('predicted_per_second') for r in requests]}; "
          f"TTFT {[round(r['first'], 1) if r.get('first') is not None else None for r in requests]}", flush=True)


