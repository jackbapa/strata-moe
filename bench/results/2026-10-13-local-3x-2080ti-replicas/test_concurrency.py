"""Local API comparison on one installed engine; no downloads or source changes."""
import concurrent.futures
import copy
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

BASE = json.loads((OUT / "original-config.json").read_text(encoding="utf-8-sig"))
URL = "http://127.0.0.1:8080"


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def config(name, gpu, resident=False, replicas=None, parallel=1):
    c = copy.deepcopy(BASE)
    c["gpu"] = gpu
    c["parallel"] = parallel
    c.pop("replicas", None)
    c.pop("layer_split", None)
    if replicas:
        c["replicas"] = replicas
    c["args"][c["args"].index("--max-context") + 1] = "65536"
    if resident:
        c["args"].append("--resident-experts")
    c["log"] = str(OUT / (name + ".engine.log"))
    return c


def stop(p):
    try:
        children = psutil.Process(p.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
    for child in children:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    p.terminate()
    psutil.wait_procs(children, timeout=10)
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()


def readiness(p, timeout=600):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if p.poll() is not None:
            raise RuntimeError(f"server exited: {p.returncode}")
        try:
            h = http_json(URL, "/health", timeout=2)
            if h.get("loaded"):
                return h
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError("model load timed out")


def burst(prompts, tokens):
    barrier = threading.Barrier(len(prompts))

    def one(prompt):
        barrier.wait()
        return chat(URL, "", [{"role": "user", "content": prompt}], tokens, timeout=600)

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        requests = list(pool.map(one, prompts))
    wall = time.perf_counter() - start
    generated = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in requests)
    return {"wall_s": wall, "output_tokens": generated,
            "aggregate_wall_tok_s": generated / wall, "requests": requests}


def summary(requests):
    rates = [(r.get("timings") or {}).get("predicted_per_second") for r in requests]
    rates = [r for r in rates if r is not None]
    ttft = [r["first"] for r in requests if r.get("first") is not None]
    return {"decode_median": statistics.median(rates) if rates else None,
            "decode_range": [min(rates), max(rates)] if rates else None,
            "ttft_median_s": statistics.median(ttft) if ttft else None,
            "errors": [r["err"] for r in requests if r.get("err")],
            "requests": len(requests)}


arms = [
    ("single-full", config("single-full", 0)),
    ("single-resident", config("single-resident", 0, resident=True)),
    ("replicas3-resident", config("replicas3-resident", [0, 1, 2], resident=True, replicas=3)),
    ("split3-batch3", config("split3-batch3", [0, 1, 2], parallel=3)),
]
RESULT_PATH = OUT / "results.json"
LAUNCHER = ROOT / "serve/server.py"
if "--tuned" in sys.argv:
    tuned = config("replicas3-budget34-affinity", [0, 1, 2], resident=True, replicas=3)
    tuned["args"] += ["--resident-budget-gib", "34"]
    arms = [("replicas3-budget34-affinity", tuned)]
    RESULT_PATH = OUT / "tuned-results.json"
    LAUNCHER = OUT / "start-replicas-local.py"
results = {"date_local": "2026-10-13", "context": 65536, "arms": {}}
save(RESULT_PATH, results)
for name, cfg in arms:
    path = OUT / (name + ".json")
    save(path, cfg)
    print(f"START {name}", flush=True)
    env = dict(os.environ, STRATA_REPLICA_START="serial", PYTHONUNBUFFERED="1")
    with (OUT / (name + ".server.log")).open("w", encoding="utf-8") as log:
        p = subprocess.Popen([sys.executable, str(LAUNCHER), "--engine", "strata",
                              "--config", str(path), "--host", "127.0.0.1", "--port", "8080"],
                             cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        arm = {"config": str(path), "pid": p.pid}
        results["arms"][name] = arm
        samples = []
        done = threading.Event()

        def sample():
            while not done.is_set():
                vm = psutil.virtual_memory()
                samples.append({"time": time.time(), "available_gib": vm.available / 2**30,
                                "used_gib": (vm.total - vm.available) / 2**30})
                done.wait(1)

        monitor = threading.Thread(target=sample, daemon=True)
        monitor.start()
        try:
            arm["health"] = readiness(p)
            print(f"READY {name}; available RAM {psutil.virtual_memory().available / 2**30:.1f} GiB", flush=True)
            arm["warmup"] = burst(["Warm up: " + x for x in PROMPTS] if "3" in name else ["Say hello."], 32)
            if arm["warmup"].get("requests") and any(r.get("err") for r in arm["warmup"]["requests"]):
                raise RuntimeError("warmup failed")
            arm["solo"] = []
            for round_no in range(3):
                for i, prompt in enumerate(PROMPTS):
                    r = chat(URL, "", [{"role": "user", "content": f"Test {round_no}-{i}.\n{prompt}"}],
                             192, timeout=600)
                    arm["solo"].append(r)
                print(f"SOLO {name} round {round_no + 1}: {summary(arm['solo'])}", flush=True)
                save(RESULT_PATH, results)
            arm["solo_summary"] = summary(arm["solo"])
            if "3" in name:
                arm["bursts"] = []
                for round_no in range(3):
                    b = burst([f"Concurrent round {round_no}, item {i}.\n{x}" for i, x in enumerate(PROMPTS)], 256)
                    arm["bursts"].append(b)
                    print(f"BURST {name} {round_no + 1}: {b['aggregate_wall_tok_s']:.2f} wall tok/s; "
                          f"{summary(b['requests'])}", flush=True)
                    save(RESULT_PATH, results)
                arm["burst_summary"] = summary([r for b in arm["bursts"] for r in b["requests"]])
                arm["burst_wall_median"] = statistics.median(b["aggregate_wall_tok_s"] for b in arm["bursts"])
                arm["long_burst"] = burst([f"Long item {i}.\n" + long_prompt(3800) for i in range(3)], 128)
                print(f"LONG {name}: {arm['long_burst']['aggregate_wall_tok_s']:.2f} wall tok/s; "
                      f"{summary(arm['long_burst']['requests'])}", flush=True)
            else:
                arm["long_solo"] = chat(URL, "", [{"role": "user", "content": long_prompt(3800)}], 128, timeout=600)
            arm["metrics"] = http_json(URL, "/metrics", timeout=10)
            arm["status"] = http_json(URL, "/v1/status", timeout=10)
            arm["gpu_end"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                                       "--format=csv,noheader"], text=True)
        except Exception as e:
            arm["error"] = repr(e)
            print(f"ERROR {name}: {e}", flush=True)
        finally:
            done.set()
            monitor.join(timeout=3)
            arm["memory_min_available_gib"] = min((s["available_gib"] for s in samples), default=None)
            save(OUT / (name + ".memory.json"), samples)
            stop(p)
            save(RESULT_PATH, results)
            print(f"STOP {name}", flush=True)
            time.sleep(3)
print("COMPARISON COMPLETE", flush=True)
