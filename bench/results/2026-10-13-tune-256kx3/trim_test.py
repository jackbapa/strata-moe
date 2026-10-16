"""Final bounded A/B trial; restore a verified running main service afterwards."""
from pathlib import Path
import concurrent.futures
import json

# Load only the existing harness definitions, before its top-level experiment.
source = Path(__file__).with_name("tune.py").read_text(encoding="utf-8")
exec(compile(source.split("\ntry:\n", 1)[0], "tune_definitions", "exec"))

previous = json.loads((ROOT / "strata-iq3_s.json").read_text(encoding="utf-8"))
comparison = json.loads((OUT / "comparison.json").read_text(encoding="utf-8"))
reference = next(p for p in comparison["profiles"] if p["name"] == comparison["selected"])
trial = json.loads((OUT / "trim15-32/config.json").read_text(encoding="utf-8"))
DATA = {int(size): [{**p, "prompt": (OUT / "inputs" / (p["tag"] + ".txt")).read_text(encoding="utf-8")}
                   for p in data] for size, data in comparison["inputs"].items()}
ALL = {"profiles": [], "status": "testing", "reference": reference}

def save_all():
    (OUT / "trim-comparison.json").write_text(json.dumps(ALL, ensure_ascii=False, indent=2), encoding="utf-8")

def config(name, chunk, groups):
    return copy.deepcopy(trial)

chosen = previous
try:
    item = profile("trim15-32", 8192, 3)
    gain = 1 - item["score_s"] / reference["score_s"]
    ALL["gain_fraction"] = gain
    # Require a meaningful observed improvement to adopt this additional flag.
    if gain >= 0.05:
        chosen = copy.deepcopy(trial)
        ALL["selected"] = "trim15-32"
    else:
        ALL["selected"] = reference["name"]
except Exception as e:
    ALL.update(trial_error=repr(e), selected=reference["name"])
    print("TRIAL FAILED; restoring previous measured winner: " + repr(e), flush=True)
finally:
    b.STOP.set()
    chosen["log"] = str(ROOT / "strata-iq3_s.log")
    stop_current()
    start(chosen, ROOT / "strata-iq3_s.json", "final-server.log")
    validation = verify()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        smoke = list(pool.map(lambda i: b.chat(b.URL, "", [{"role": "user", "content": f"Reply exactly READY {i}."}], 16, timeout=60), range(3)))
    assert all(not r.get("err") for r in smoke), smoke
    validation.update(config=chosen, smoke=smoke, resident_experts_enabled="--resident-experts" in chosen["args"])
    (OUT / "final-validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    ALL.update(status="complete", final_verified=validation, service_left_running=True)
    save_all()
    print("COMPLETE: " + ALL["selected"] + "; three independent 256K slots; resident-experts OFF", flush=True)
