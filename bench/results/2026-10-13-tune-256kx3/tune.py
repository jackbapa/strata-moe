"""Compare bounded configurations at independent 256K context per slot."""
import copy
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import sys

import psutil
import tuning_common as b

ROOT = Path(r"G:\Strata")
OUT = Path(__file__).resolve().parent
BASE = json.loads((OUT / "base-config.json").read_text(encoding="utf-8-sig"))
ALL = {"started": datetime.datetime.now().astimezone().isoformat(), "context_per_slot": 262144,
       "slots": 3, "driver": "617.42", "profiles": [], "status": "calibration"}
RESUME = "--resume" in sys.argv
if RESUME:
    ALL = json.loads((OUT / "comparison.json").read_text(encoding="utf-8"))
    ALL["status"] = "resuming"
    ALL.pop("error", None)
HANDLES = []


def save_all():
    (OUT / "comparison.json").write_text(json.dumps(ALL, ensure_ascii=False, indent=2), encoding="utf-8")


def config(name, chunk, groups):
    c = copy.deepcopy(BASE)
    c["args"][c["args"].index("--prefill") + 1] = str(chunk)
    c["args"][c["args"].index("--max-context") + 1] = "262144"
    c["args"][c["args"].index("--batch-groups") + 1] = str(groups)
    c["parallel"] = 3
    c["log"] = str(OUT / name / "engine.log")
    return c


def idle_check():
    live = b.http_json(b.URL, "/metrics", timeout=5)["live"]
    if live.get("running") or live.get("queued"):
        raise RuntimeError("service is busy; not restarting")


def stop_current():
    owner = next((c.pid for c in psutil.net_connections(kind="tcp")
                  if c.status == psutil.CONN_LISTEN and c.laddr.port == 8080), None)
    if owner is None:
        if any(p.info["name"].lower() in ("strata.exe", "strata-vision.exe") for p in psutil.process_iter(["name"])):
            raise RuntimeError("orphaned engines remain; not starting another model")
        return
    idle_check()
    p = psutil.Process(owner)
    assert any("Strata" in a and "server.py" in a for a in p.cmdline())
    children = [(c.pid, c.create_time()) for c in p.children(recursive=True)
                if c.name().lower() in ("strata.exe", "strata-vision.exe")]
    subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                    f"Stop-Process -Id {owner} -Force"], check=True, capture_output=True)
    end = time.monotonic() + 10
    while time.monotonic() < end:
        if not any(p.info["name"] in ("strata.exe", "strata-vision.exe")
                   for p in psutil.process_iter(["name"])):
            return
        time.sleep(0.5)
    survivors = []
    for pid, created in children:
        try:
            child = psutil.Process(pid)
            if child.create_time() == created:
                survivors.append(pid)
        except psutil.NoSuchProcess:
            pass
    if survivors:
        subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                        "Stop-Process -Id " + ",".join(map(str, survivors)) + " -Force"],
                       check=True, capture_output=True)
    time.sleep(3)


def start(c, path, log_name):
    path.write_text(json.dumps(c, indent=2), encoding="utf-8")
    log = (OUT / log_name).open("w", encoding="utf-8")
    HANDLES.append(log)
    p = subprocess.Popen([sys.executable, str(ROOT / "serve/server.py"), "--engine", "strata",
                          "--config", str(path), "--host", "127.0.0.1", "--port", "8080"],
                         cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1"), stdout=log, stderr=subprocess.STDOUT)
    b.SERVER = psutil.Process(p.pid)
    end = time.monotonic() + 480
    while time.monotonic() < end:
        if p.poll() is not None:
            raise RuntimeError(f"server exited {p.returncode}")
        try:
            if b.http_json(b.URL, "/health", timeout=2).get("loaded"):
                return p
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("startup timeout")


def verify():
    health = b.http_json(b.URL, "/health", timeout=5)
    slots = b.http_json(b.URL, "/slots", timeout=5)
    assert health["max_context"] == 262144
    assert len(slots) == 3 and all(s["n_ctx"] == 262144 for s in slots), slots
    return {"health": health, "slots": slots}


def cool():
    # Idle cooling between input sizes reduces thermal/order differences.
    b.STATE["phase"] = "cooling"
    print("COOLING before next burst", flush=True)
    for _ in range(18):
        data = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True)
        peak = max(int(x) for x in data.splitlines())
        if peak <= 55:
            return
        time.sleep(5)


def digest(stage):
    requests = stage["requests"]
    return {"name": stage["name"], "wall_s": stage["wall_s"],
            "input_tokens": [r["usage"]["prompt_tokens"] for r in requests],
            "cached_tokens": [(r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens") for r in requests],
            "output_tokens": [r["usage"]["completion_tokens"] for r in requests],
            "ttft_s": sorted(r["first"] for r in requests),
            "completion_s": sorted(r["total"] for r in requests),
            "decode_tok_s": [r["timings"]["predicted_per_second"] for r in requests],
            "prompt_tok_s": [r["timings"]["prompt_per_second"] for r in requests]}


def profile(name, chunk, groups=3, adopt=False):
    print(f"PROFILE {name} chunk={chunk} groups={groups}", flush=True)
    d = OUT / name
    d.mkdir(exist_ok=True)
    c = config(name, chunk, groups)
    if not adopt:
        stop_current()
        start(c, d / "config.json", name + ".server.log")
    else:
        (d / "config.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    b.OUT = d
    b.STOP = threading.Event()
    b.ABORT = threading.Event()
    b.STATE = {"phase": "ready", "active": False, "last_progress": time.monotonic(), "signature": None}
    b.RESULT = {"started": datetime.datetime.now().astimezone().isoformat(), "status": "ready", "stages": [],
                "context_limit": 262144, "config": c, **verify()}
    watcher = threading.Thread(target=b.monitor, daemon=True)
    watcher.start()
    item = {"name": name, "chunk": chunk, "groups": groups, "status": "testing"}
    ALL["profiles"].append(item)
    save_all()
    try:
        warm = b.chat(b.URL, "", [{"role": "user", "content": "Write three short sentences about sorting arrays."}], 64, timeout=60)
        if warm.get("err"):
            raise RuntimeError(warm["err"])
        cool()
        b.run("32kx3", DATA[32768])
        item["32k"] = digest(b.RESULT["stages"][-1])
        save_all()
        cool()
        b.run("64kx3", DATA[65536])
        item["64k"] = digest(b.RESULT["stages"][-1])
        item["score_s"] = item["32k"]["wall_s"] + item["64k"]["wall_s"]
        item["status"] = "passed"
        b.RESULT.update(status="passed", finished=datetime.datetime.now().astimezone().isoformat())
        print(f"PROFILE DONE {name}: 32K {item['32k']['wall_s']:.1f}s, 64K {item['64k']['wall_s']:.1f}s", flush=True)
    except Exception as e:
        item.update(status="failed", error=repr(e))
        b.RESULT.update(status="failed", error=repr(e))
        raise
    finally:
        b.STOP.set()
        watcher.join(timeout=12)
        b.save()
        if (d / "telemetry.jsonl").exists():
            samples = [json.loads(x) for x in (d / "telemetry.jsonl").read_text().splitlines()]
            cards = [g for s in samples for g in s.get("gpus", [])]
            item["min_available_ram_gib"] = min(s["available_ram_gib"] for s in samples)
            item["max_temperature_c"] = max(g["temperature_c"] for g in cards)
            item["min_gpu_free_mib"] = min(g["total_mib"] - g["used_mib"] for g in cards)
        save_all()
    return item


try:
    b.OUT = OUT / "inputs"
    b.OUT.mkdir(exist_ok=True)
    if RESUME:
        DATA = {int(size): [{**p, "prompt": (b.OUT / (p["tag"] + ".txt")).read_text(encoding="utf-8")}
                           for p in data] for size, data in ALL["inputs"].items()}
    else:
        verify()
        DATA = {size: [b.calibrate(f"input{size}-{letter}", size) for letter in "ABC"] for size in (32768, 65536)}
    ALL["inputs"] = {str(size): [{k: v for k, v in p.items() if k != "prompt"} for p in data] for size, data in DATA.items()}
    save_all()
    for chunk in (2048, 4096, 8192):
        name = f"chunk{chunk}-g3"
        if any(p["name"] == name and p["status"] == "passed" for p in ALL["profiles"]):
            continue
        if RESUME and any(p["name"] == name and p["status"] == "failed" for p in ALL["profiles"]):
            print(f"SKIP failed profile {name}; preserve its recorded failure", flush=True)
            continue
        profile(name, chunk, adopt=not RESUME and chunk == 2048)
    candidate = min((p for p in ALL["profiles"] if p["status"] == "passed"), key=lambda p: p["score_s"])
    profile(f"chunk{candidate['chunk']}-g1", candidate["chunk"], groups=1)
    winner = min((p for p in ALL["profiles"] if p["status"] == "passed"), key=lambda p: p["score_s"])
    final = config("selected", winner["chunk"], winner["groups"])
    final["log"] = str(ROOT / "strata-iq3_s.log")
    ALL["selected"] = winner["name"]
    ALL["selection_metric"] = "sum of one 32Kx3 and one 64Kx3 burst completion time; output cap 128"
    (ROOT / "strata-iq3_s.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    stop_current()
    start(final, ROOT / "strata-iq3_s.json", "selected-server.log")
    ALL["final_verified"] = verify()
    ALL.update(status="complete", service_left_running=True, finished=datetime.datetime.now().astimezone().isoformat())
    save_all()
    print(f"COMPLETE selected {winner['name']}; three independent 256K slots active", flush=True)
except Exception as e:
    ALL.update(status="failed", error=repr(e))
    save_all()
    print("TUNING FAILED: " + repr(e), flush=True)
