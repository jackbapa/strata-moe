"""Run the same A3B K2 executable and fixture with three input-staging modes."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXE = ROOT / "build-a3b/bin/strata-moe-speculative-test.exe"
MODEL = ROOT / "models/a3b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
MTP = ROOT / "models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0.gguf"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("off", "async", "forced"))
    parser.add_argument("--nsys", action="store_true")
    args = parser.parse_args()
    for path in (EXE, MODEL, MTP):
        if not path.is_file():
            raise FileNotFoundError(path)
    arm_dir = HERE / args.arm
    arm_dir.mkdir(exist_ok=True)
    state_dir = arm_dir / ("nsys-state" if args.nsys else "state")
    state_dir.mkdir(exist_ok=True)
    env = dict(os.environ)
    for key in ("STRATA_A3B_MTP_ASYNC_INPUT",
                "STRATA_A3B_MTP_ASYNC_INPUT_FORCE_STEP2_FALLBACK",
                "STRATA_A3B_NVTX"):
        env.pop(key, None)
    env["STRATA_A3B_MTP_DEVICE_CHAIN_K2"] = "1"
    env["STRATA_A3B_MTP_PROPOSAL_TRACE"] = "1"
    if args.arm != "off":
        env["STRATA_A3B_MTP_ASYNC_INPUT"] = "1"
    if args.arm == "forced":
        env["STRATA_A3B_MTP_ASYNC_INPUT_FORCE_STEP2_FALLBACK"] = "1"
    if args.nsys:
        env["STRATA_A3B_NVTX"] = "1"
    cmd = [str(EXE), "--model", str(MODEL), "--threads", "16",
           "--device", "0", "--native-gpu", "--draft-count", "2",
           "--fused-moe", "--native-dense-exact", "--native-attention-exact",
           "--gpu-embedding", "--nextn-hidden", "--mtp-model", str(MTP),
           "--gpu-sampling", "--defer-mtp-catchup", "--draft-ubatch", "32",
           "--assert-sequential", "--draft-state-dir", str(state_dir)]
    if args.nsys:
        nsys = Path(r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.2\target-windows-x64\nsys.exe")
        if not nsys.is_file():
            raise FileNotFoundError(nsys)
        cmd = [str(nsys), "profile", "--trace=cuda,nvtx", "--sample=none",
               "--cpuctxsw=none", "--force-overwrite=true", "--output",
               str(arm_dir / "profile"), *cmd]
    metadata = {
        "arm": args.arm,
        "nsys": args.nsys,
        "binary_sha256": sha256(EXE),
        "command": cmd,
        "env": {k: env.get(k) for k in sorted(set(env) | {
            "STRATA_A3B_MTP_DEVICE_CHAIN_K2",
            "STRATA_A3B_MTP_ASYNC_INPUT",
            "STRATA_A3B_MTP_ASYNC_INPUT_FORCE_STEP2_FALLBACK",
            "STRATA_A3B_NVTX"})
                if k.startswith("GGML_CUDA_") or k.startswith("STRATA_A3B_MTP_") or
                k == "STRATA_A3B_NVTX"},
    }
    metadata_path = arm_dir / ("nsys-metadata.json" if args.nsys else "metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    begin = time.monotonic()
    with (arm_dir / ("nsys.log" if args.nsys else "strict.log")).open("wb") as log:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=1800)
    metadata["returncode"] = result.returncode
    metadata["wall_seconds"] = round(time.monotonic() - begin, 3)
    metadata["binary_sha256_after"] = sha256(EXE)
    metadata["state_files"] = len(list(state_dir.glob("*.bin")))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metadata[k] for k in
                      ("arm", "returncode", "wall_seconds", "binary_sha256",
                       "binary_sha256_after", "state_files")}, indent=2))
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
