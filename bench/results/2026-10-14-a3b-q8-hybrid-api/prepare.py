"""Prepare the frozen CPU/GPU hybrid API experiment, without starting it."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPECTED_EXE_SHA256 = "a506677ca04b1c6f643f6c127cba61cd45c84f5d5ad6ddc7a324b8e17fe0c81a"


def main():
    target = HERE / "config.json"
    if target.exists() or (HERE / "preparation.json").exists():
        raise FileExistsError("prepared evidence already exists; retain it instead of overwriting")
    source_path = ROOT / "strata-a3b-q8.json"
    source = source_path.read_bytes()
    config = json.loads(source)
    arguments = config["args"]
    assert "--gpu-only-experts" in arguments and "--mtp-adaptive" in arguments
    assert arguments[arguments.index("--mtp-draft") + 1] == "6"
    assert arguments[arguments.index("--cache-decay-interval") + 1] == "4"
    assert arguments[arguments.index("--cache-reserve-mib") + 1] == "384"
    for excluded in ("--hybrid-experts", "--cpu-expert-pool", "--threads", "--pcie-frac",
                     "--gpu-static-experts", "--gpu-dynamic-experts"):
        assert excluded not in arguments, excluded
    arguments.remove("--gpu-only-experts")
    arguments += ["--hybrid-experts", "--cpu-expert-pool", "--threads", "8", "--pcie-frac", "0.25"]
    executable = ROOT / "build-a3b-hybrid-compatible/bin/strata-moe.exe"
    executable_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
    if executable_sha != EXPECTED_EXE_SHA256:
        raise ValueError("frozen hybrid executable SHA differs from the measured build")
    config["exe"], config["log"] = str(executable), str(HERE / "engine.log")
    config["host"], config["port"], config["idle_unload_s"] = "127.0.0.1", 8096, 0
    target.write_bytes((json.dumps(config, indent=2) + "\n").encode())
    preparation = {"scope": "Prepared only; no model/service/GPU request and no production config change.",
                   "source_config": str(source_path), "source_config_sha256": hashlib.sha256(source).hexdigest(),
                   "config_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                   "frozen_exe_sha256": executable_sha,
                   "changes": ["frozen hybrid executable and independent engine log", "hybrid experts with Q8 CPU pool",
                               "threads 8", "PCIe fraction 0.25"],
                   "preserved": "official Q8 target/MTP, strict shared weights, context4096/batch512, MTP6adaptive, profile/decay4/reserve384, exact dense/attention, async, existing env and sampling defaults"}
    (HERE / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n", encoding="utf-8")
    assert source_path.read_bytes() == source
    print(json.dumps(preparation, indent=2))


if __name__ == "__main__":
    main()
