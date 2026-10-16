"""Bounded retest of the current three-stage / three-slot configuration.

Write telemetry and results as they arrive. Stop this server on a fresh NVIDIA
event, low RAM/VRAM, a failed GPU query, or 30 seconds without engine-log progress
during a request. These guards cannot guarantee that a kernel crash is prevented.
"""
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

import psutil

ROOT = Path(r"G:\Strata")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tools"))
from strata_bench import chat, http_json
from ab_engine import PROMPTS, long_prompt

URL = "http://127.0.0.1:8080"
CONFIG = json.loads((OUT / "config.json").read_text(encoding="utf-8-sig"))
CONFIG["log"] = str(OUT / "engine.log")
(OUT / "run-config.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
started = datetime.datetime.now().astimezone()
stop_monitor = threading.Event()
aborted = threading.Event()
state = {"phase": "loading", "active_since": None}
results = {"started": started.isoformat(), "driver": "617.42", "config": CONFIG,
           "stages": [], "status": "loading"}
write_lock = threading.Lock()
server = None


def checkpoint():
    with write_lock:
        (OUT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def journal(name, entry):
    with write_lock, (OUT / name).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def stop_server():
    if server is None:
        return
    try:
        parent = psutil.Process(server.pid)
        children = parent.children(recursive=True)
        for p in children:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        parent.terminate()
        _, alive = psutil.wait_procs(children + [parent], timeout=4)
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass


def abort(reason):
    if aborted.is_set():
        return
    aborted.set()
    results["status"] = "aborted"
    results["abort_reason"] = reason
    print("ABORT: " + reason, flush=True)
    journal("alerts.jsonl", {"time": time.time(), "reason": reason})
    checkpoint()
    stop_server()


def monitor():
    start_iso = started.strftime("%Y-%m-%dT%H:%M:%S")
    event_cmd = ("Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='nvlddmkm';"
                 f"StartTime=[datetime]'{start_iso}'" + "} -ErrorAction SilentlyContinue | "
                 "ForEach-Object { [pscustomobject]@{id=$_.Id;record=$_.RecordId;"
                 "time=$_.TimeCreated.ToString('o');xml=$_.ToXml()} } | ConvertTo-Json -Compress")
    last_event_poll = 0
    max_running = 0
    while not stop_monitor.is_set():
        now = time.time()
        vm = psutil.virtual_memory()
        entry = {"time": now, "phase": state["phase"], "available_ram_gib": vm.available / 2**30}
        if vm.available < 8 * 2**30:
            journal("telemetry.jsonl", entry)
            abort("available RAM below 8 GiB")
            return
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            if q.returncode:
                raise RuntimeError(q.stderr.strip())
            cards = []
            for line in q.stdout.strip().splitlines():
                x = [v.strip() for v in line.split(",")]
                cards.append({"index": int(x[0]), "used_mib": int(x[1]), "total_mib": int(x[2]),
                              "temperature_c": int(x[3]), "utilization": x[4], "power_w": x[5]})
            entry["gpus"] = cards
            if any(x["temperature_c"] >= 90 for x in cards):
                abort("GPU temperature reached 90 C")
                return
            if state["phase"] != "loading" and any(x["total_mib"] - x["used_mib"] < 256 for x in cards):
                abort("GPU free VRAM below 256 MiB")
                return
        except Exception as e:
            entry["gpu_query_error"] = repr(e)
            journal("telemetry.jsonl", entry)
            abort("GPU query failed or timed out: " + repr(e))
            return
        if now - last_event_poll > 4:
            last_event_poll = now
            try:
                ev = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", event_cmd],
                                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=6)
                if ev.stdout.strip():
                    events = json.loads(ev.stdout)
                    events = events if isinstance(events, list) else [events]
                    journal("events.jsonl", {"time": now, "events": events})
                    if any(e["id"] in (13, 14, 153) for e in events):
                        abort("new nvlddmkm error event")
                        return
            except Exception as e:
                entry["event_query_error"] = repr(e)
        if state["phase"] != "loading":
            try:
                metrics = http_json(URL, "/metrics", timeout=2)
                live = metrics.get("live", {})
                entry["live"] = {k: live.get(k) for k in ("state", "running", "parallel", "queued", "phase")}
                max_running = max(max_running, live.get("running") or 0)
                results["max_observed_running"] = max_running
            except Exception as e:
                entry["metrics_error"] = repr(e)
        journal("telemetry.jsonl", entry)
        active = state["active_since"]
        log = OUT / "engine.log"
        if active is not None and now - active > 30 and log.exists() and now - log.stat().st_mtime > 30:
            abort("engine log made no progress for 30 seconds during a request")
            return
        stop_monitor.wait(1)


def check_abort():
    if aborted.is_set():
        raise RuntimeError(results.get("abort_reason", "aborted"))
    if server.poll() is not None:
        raise RuntimeError(f"server exited: {server.returncode}")


def request_set(name, prompts, max_tokens):
    check_abort()
    state.update(phase=name, active_since=time.time())
    print(f"TEST {name}, {len(prompts)} clients", flush=True)
    barrier = threading.Barrier(len(prompts))

    def one(item):
        i, prompt = item
        barrier.wait()
        r = chat(URL, "", [{"role": "user", "content": prompt}], max_tokens, timeout=90)
        journal("requests.jsonl", {"stage": name, "client": i, **r})
        return r

    begin = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        requests = list(pool.map(one, enumerate(prompts)))
    wall = time.perf_counter() - begin
    state["active_since"] = None
    rates = [(r.get("timings") or {}).get("predicted_per_second") for r in requests]
    rates = [r for r in rates if r is not None]
    generated = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in requests)
    stage = {"name": name, "wall_s": wall, "aggregate_wall_tok_s": generated / wall,
             "decode_median": statistics.median(rates) if rates else None,
             "requests": requests}
    results["stages"].append(stage)
    checkpoint()
    print(f"DONE {name}: decode {stage['decode_median']}, wall {stage['aggregate_wall_tok_s']:.2f} tok/s", flush=True)
    if any(r.get("err") or not r.get("usage") or not r.get("timings") for r in requests):
        raise RuntimeError(f"request failed in {name}")
    check_abort()
    return stage


try:
    with (OUT / "server.log").open("w", encoding="utf-8") as log:
        server = subprocess.Popen([sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata",
                                   "--config", str(OUT / "run-config.json"), "--host", "127.0.0.1", "--port", "8080"],
                                  cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1"), stdout=log, stderr=subprocess.STDOUT)
        results["server_pid"] = server.pid
        checkpoint()
        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        deadline = time.monotonic() + 480
        while time.monotonic() < deadline:
            check_abort()
            try:
                health = http_json(URL, "/health", timeout=2)
                if health.get("loaded"):
                    results["health"] = health
                    break
            except Exception:
                pass
            time.sleep(2)
        else:
            raise TimeoutError("model startup exceeded 480 seconds")
        state["phase"] = "ready"
        results["status"] = "testing"
        results["v1_status"] = http_json(URL, "/v1/status", timeout=5)
        if results["v1_status"].get("concurrency", {}).get("serving") != 3:
            raise RuntimeError("engine did not expose three concurrent slots")
        checkpoint()
        print("READY: 3 concurrent slots", flush=True)
        request_set("warmup", ["Say hello in five words."], 32)
        for round_no in range(3):
            request_set(f"short3-{round_no}", [f"Fresh trial {round_no} item {i}.\n{p}" for i, p in enumerate(PROMPTS)], 256)
        request_set("long4k-solo", ["Solo fresh input.\n" + long_prompt(3800)], 128)
        for clients in (2, 3):
            request_set(f"long4k-{clients}clients", [f"Fresh batch {clients} item {i}.\n" + long_prompt(3800)
                                                     for i in range(clients)], 128)
        for round_no in range(3):
            request_set(f"long4k-repeat-{round_no}", [f"Fresh repeated batch {round_no} item {i}.\n" + long_prompt(3800)
                                                     for i in range(3)], 128)
        state["phase"] = "idle-check"
        time.sleep(5)
        check_abort()
        results["health_end"] = http_json(URL, "/health", timeout=5)
        results["status"] = "passed-no-reproduction"
        results["finished"] = datetime.datetime.now().astimezone().isoformat()
        results["service_left_running"] = True
        checkpoint()
        print("PASSED: no reproduction; server left running", flush=True)
except Exception as e:
    results["error"] = repr(e)
    abort(repr(e))
    results["service_left_running"] = False
    checkpoint()
finally:
    stop_monitor.set()
    if "watcher" in globals():
        watcher.join(timeout=12)
    checkpoint()
