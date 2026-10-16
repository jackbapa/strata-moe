"""Fresh-restart A/B of resident-experts, using identical 32K inputs and 1024 output cap."""
from pathlib import Path
import argparse
import copy
import datetime
import json
import os
import statistics
import subprocess
import sys
import time

BASE = Path(__file__).resolve().parent
ns = {"__file__": str(BASE / "tune.py")}
source = (BASE / "tune.py").read_text()
exec(compile(source.split("\ntry:\n", 1)[0], "harness_definitions", "exec"), ns)
b = ns["b"]
parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, default=BASE / "resident-ab-32k")
parser.add_argument("--layer-split")
options = parser.parse_args()
OUT = options.output_dir
OUT.mkdir(exist_ok=True)
original = json.loads((b.ROOT / "strata-iq3_s.json").read_text())
(OUT / "original-config.json").write_text(json.dumps(original, indent=2))
if options.layer_split:
    original["layer_split"] = options.layer_split
comparison = {"started": datetime.datetime.now().astimezone().isoformat(), "profiles": [],
              "status": "running", "input_tokens_per_request": 32768, "output_tokens_per_request": 1024,
              "context_per_slot": 262144, "driver": "617.42", "order": ["off", "on"],
              "layer_split": original["layer_split"],
              "method": "identical inputs, fresh server per arm, same warmup and cooling rule; one burst per arm"}
def save():
    (OUT / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2))

def arm(mode):
    d = OUT / mode
    d.mkdir(exist_ok=True)
    c = copy.deepcopy(original)
    c["args"] = [x for x in c["args"] if x != "--resident-experts"]
    if mode == "on":
        c["args"].append("--resident-experts")
    c["log"] = str(d / "engine.log")
    ns["stop_current"]()
    print(f"START {mode}: fresh model process", flush=True)
    start_begin = time.perf_counter()
    ns["start"](c, d / "config.json", f"{OUT.name}-{mode}.server.log")
    startup_s = time.perf_counter() - start_begin
    verification = ns["verify"]()
    warm = b.chat(b.URL, "", [{"role": "user", "content": "Write three short sentences about sorting arrays."}], 64, timeout=60)
    assert not warm.get("err"), warm
    cooling = []
    for _ in range(12):
        values = subprocess.check_output(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"], text=True)
        temps = [int(x) for x in values.splitlines()]
        cooling.append({"time": time.time(), "temperatures_c": temps})
        if max(temps) <= 55:
            break
        time.sleep(5)
    (d / "cooling.json").write_text(json.dumps(cooling, indent=2))
    print(f"TEST {mode}: start temperatures {temps}; 32K x 3, 1024 outputs each", flush=True)
    with (d / "progress.log").open("w") as log:
        run = subprocess.run([sys.executable, str(BASE / "steady32.py"), "--output-dir", str(d),
                              "--config", str(d / "config.json"), "--reuse-inputs", str(BASE / "steady32-output1024"),
                              "--expected-resident", mode, "--no-restore"], stdout=log, stderr=subprocess.STDOUT,
                             cwd=b.ROOT, timeout=600)
    assert run.returncode == 0, run.returncode
    results = json.loads((d / "results.json").read_text())
    assert results["status"] == "complete", results.get("error")
    summary = json.loads((d / "summary.json").read_text())
    samples = [json.loads(x) for x in (d / "telemetry.jsonl").read_text().splitlines()]
    cards = [g for s in samples for g in s.get("gpus", [])]
    item = {"mode": mode, "status": "passed", "startup_s": startup_s, "common_total_tok_s": summary["common_total_tok_s"],
            "common_per_slot_tok_s": summary["common_per_slot_tok_s"], "common_window_s": summary["common_window_s"],
            "wall_s": summary["wall_s"], "end_to_end_output_tok_s": summary["end_to_end_output_tok_s"],
            "ttft_s": sorted(r["first"] for r in summary["requests"]),
            "available_ram_start_gib": samples[0]["available_ram_gib"],
            "available_ram_min_gib": min(s["available_ram_gib"] for s in samples),
            "max_temperature_c": max(g["temperature_c"] for g in cards),
            "min_gpu_free_mib": min(g["total_mib"] - g["used_mib"] for g in cards),
            "start_temperatures_c": temps, "verified": verification}
    comparison["profiles"].append(item)
    save()
    print("DONE " + json.dumps({k:v for k,v in item.items() if k != "verified"}, ensure_ascii=False), flush=True)
    return item

chosen = copy.deepcopy(original)
try:
    save()
    off = arm("off")
    on = arm("on")
    speed_ratio = on["common_total_tok_s"] / off["common_total_tok_s"]
    wall_ratio = on["wall_s"] / off["wall_s"]
    comparison.update(on_vs_off_decode_ratio=speed_ratio, on_vs_off_wall_ratio=wall_ratio,
                      available_ram_gain_gib=on["available_ram_min_gib"] - off["available_ram_min_gib"])
    # User's goal is performance; accept the RAM-saving mode only with no material regression.
    enable = speed_ratio >= 0.97 and wall_ratio <= 1.03
    chosen["args"] = [x for x in chosen["args"] if x != "--resident-experts"]
    if enable:
        chosen["args"].append("--resident-experts")
    comparison["selected"] = "on" if enable else "off"
    comparison["selection_rule"] = "enable only if simultaneous decode is within 3% and full wall time within 3% of off; otherwise prioritize speed"
except Exception as e:
    comparison.update(status="test_failed", error=repr(e), selected="original configuration")
    print("TEST FAILED " + repr(e), flush=True)
finally:
    save()
    print("RESTORE FINAL: " + comparison.get("selected", "original"), flush=True)
    chosen["log"] = str(b.ROOT / "strata-iq3_s.log")
    ns["stop_current"]()
    ns["start"](chosen, b.ROOT / "strata-iq3_s.json", f"{OUT.name}-final.server.log")
    validation = ns["verify"]()
    with __import__("concurrent.futures").futures.ThreadPoolExecutor(max_workers=3) as pool:
        smoke = list(pool.map(lambda i: b.chat(b.URL, "", [{"role": "user", "content": f"Reply exactly READY {i}."}], 16, timeout=60), range(3)))
    assert all(not r.get("err") for r in smoke), smoke
    comparison.update(final_verified=validation, final_config=chosen, smoke=smoke, service_left_running=True,
                      finished=datetime.datetime.now().astimezone().isoformat())
    if "error" not in comparison:
        comparison["status"] = "complete"
    save()
    print("COMPLETE: " + comparison["selected"] + "; 3 independent 256K slots verified", flush=True)
