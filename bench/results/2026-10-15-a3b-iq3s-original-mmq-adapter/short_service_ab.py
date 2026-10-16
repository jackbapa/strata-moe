"""Same-build A3B IQ3_S service-path check for the opt-in Strata MMQ hook.

This intentionally uses the target model without MTP so a first-token logits
comparison isolates the prefill expert path. Run after building strata-moe.exe.
"""

from array import array
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXE = ROOT / "build-a3b" / "bin" / "strata-moe.exe"
MODEL = ROOT / "models" / "a3b" / "Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
PROMPT = HERE / "short_prompt.txt"
PROMPT.write_text(("Explain why a small routed expert model can still spend time moving quantized weights.\n" * 48), encoding="utf-8")


def run(name: str, opt_in: bool) -> dict:
    argv = [
        str(EXE), "--model", str(MODEL), "--ctx", "4096", "--batch", "512",
        "--device", "0", "--vram-mib", "0", "--predict", "1",
        "--native-gpu", "--raw", "--prompt-file", str(PROMPT),
        "--fused-moe", "--strata-native-experts", "--logits", str(HERE / f"{name}.logits"),
    ]
    if opt_in:
        argv.append("--strata-prefill-mmq")
    start = time.perf_counter()
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, errors="replace")
    wall = time.perf_counter() - start
    (HERE / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
    (HERE / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit {result.returncode}; see {name}.stderr.txt")
    pattern = r"A3B Strata original prefill MMQ: CUDA calls (\d+) rows (\d+); request prefill calls (\d+) rows (\d+); decode calls (\d+); graph builds (\d+) fallback graph builds (\d+)"
    match = re.search(pattern, result.stderr)
    return {
        "argv": argv,
        "wall_seconds_including_model_load": wall,
        "prefill_counter": list(map(int, match.groups())) if match else None,
        "prompt_log": re.search(r"A3B: prompt [^\n]+", result.stderr).group(0),
        "stdout": result.stdout,
    }


def logits(name: str) -> array:
    data = array("f")
    with (HERE / f"{name}.logits").open("rb") as file:
        data.fromfile(file, (HERE / f"{name}.logits").stat().st_size // 4)
    return data


baseline = run("baseline", False)
strata = run("strata_prefill", True)
a, b = logits("baseline"), logits("strata_prefill")
if len(a) != len(b):
    raise RuntimeError(f"logit vector size mismatch: {len(a)} != {len(b)}")
different = sum(x != y for x, y in zip(a, b))
max_abs = max(abs(x - y) for x, y in zip(a, b))
result = {
    "model": str(MODEL),
    "baseline": baseline,
    "strata_prefill": strata,
    "logit_count": len(a),
    "logit_different": different,
    "logit_max_abs": max_abs,
    "baseline_logits_sha256": hashlib.sha256((HERE / "baseline.logits").read_bytes()).hexdigest(),
    "strata_logits_sha256": hashlib.sha256((HERE / "strata_prefill.logits").read_bytes()).hexdigest(),
}
(HERE / "short_service_ab.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({k: v for k, v in result.items() if k not in ("baseline", "strata_prefill")}, indent=2))
print("baseline:", baseline["prompt_log"])
print("strata:", strata["prompt_log"])
print("actual prefill CUDA counters:", strata["prefill_counter"])
if not strata["prefill_counter"] or strata["prefill_counter"][2] <= 0:
    raise RuntimeError("Strata original MMQ service path had no actual prefill CUDA calls")
if different:
    raise RuntimeError(f"Strata prefill changed {different} first-token logits")
if any(not math.isfinite(x) for x in b):
    raise RuntimeError("non-finite Strata prefill logits")
