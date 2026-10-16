"""Prepare isolated fixed/dynamic GPU experiments; never launches an engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def set_value(arguments, key, value):
    if key in arguments:
        arguments[arguments.index(key) + 1] = str(value)
    else:
        arguments.extend([key, str(value)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "strata-a3b-q8.json")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    targets = [args.out / "config-static.json", args.out / "config-dynamic.json", args.out / "preparation.json"]
    if any(target.exists() for target in targets):
        raise FileExistsError("experiment configs already exist; retain them or prepare a new directory")
    frozen = args.source
    source = frozen.read_bytes()
    config = json.loads(source)
    arguments = config["args"]
    assert "--gpu-only-experts" in arguments and "--mtp-model" in arguments
    assert arguments[arguments.index("--device") + 1] == "0"
    profile = ROOT / "models/a3b/profiles/qwen3.6-q8-broad-v1.profile"
    if not profile.is_file():
        raise FileNotFoundError(profile)
    arguments[:] = [argument for argument in arguments if argument not in ("--gpu-static-experts", "--gpu-dynamic-experts")]
    arguments.append("--gpu-static-experts")
    set_value(arguments, "--cache-decay-interval", 0)
    set_value(arguments, "--cache-reserve-mib", 640)
    set_value(arguments, "--expert-profile", profile)
    config["exe"] = str(ROOT / "build-a3b/bin" / ("strata-moe.exe" if os.name == "nt" else "strata-moe"))
    config["host"], config["port"], config["idle_unload_s"] = "127.0.0.1", 8096, 0
    config.setdefault("env", {})["CUDA_VISIBLE_DEVICES"] = "0"
    config["env"]["STRATA_A3B_EXPERT_Q8_REUSE"] = "0"
    args.out.mkdir(parents=True, exist_ok=True)
    configs = {}
    for mode in ("static", "dynamic"):
        isolated = json.loads(json.dumps(config))
        if mode == "dynamic":
            isolated["args"].append("--gpu-dynamic-experts")
        isolated["log"] = str(args.out / f"engine-{mode}.log")
        target = args.out / f"config-{mode}.json"
        target.write_bytes((json.dumps(isolated, indent=2) + "\n").encode())
        configs[mode] = {"path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    preparation = {"scope": "Prepared only; no service/model request, GPU work or production config change.",
                   "frozen_config": str(frozen), "frozen_config_sha256": hashlib.sha256(source).hexdigest(),
                   "experiment_configs": configs,
                   "profile_sha256": hashlib.sha256(profile.read_bytes()).hexdigest(),
                   "changes": ["build-a3b prototype executable", "isolated per-mode engine logs", "GPU-static opt-in; dynamic opt-in in second config only",
                               "cache decay 0", "reserve 640 MiB plus main's separate static scratch reservation"],
                   "measurements": "Run run-latency.py only after root starts and verifies the isolated service; it hashes the actual executable then."}
    (args.out / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n", encoding="utf-8")
    assert frozen.read_bytes() == source
    print(json.dumps(preparation, indent=2))


if __name__ == "__main__":
    main()
