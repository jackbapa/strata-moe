"""Run one A3B draft-vocabulary strict fixture arm, without managing services."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXE = ROOT / "build-a3b-draft-vocab-cuda/bin/Release/strata-moe-speculative-test.exe"
TARGET = ROOT / "models/a3b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
MTP = {
    "full": ROOT / "models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0.gguf",
    "subset": ROOT / "models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0-DV-ASCII-CJK-PUNCT.gguf",
}
FLAGS = {
    "STRATA_A3B_Q6_GROUPED_EXACT": "1",
    "STRATA_A3B_MTP_KV_ONLY_DIAGNOSTIC": "0",
    "STRATA_A3B_MTP_PROPOSAL_TRACE": "1",
    "STRATA_A3B_FUSED_GU": "1",
    "STRATA_A3B_DOWN_MODE": "1",
    "STRATA_A3B_Q6_REUSE_EXACT": "1",
    "STRATA_A3B_MTP_KV_ONLY": "1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=MTP)
    parser.add_argument("--device-chain", action="store_true",
                        help="enable dependent CUDA K2 chain for this strict arm")
    args = parser.parse_args()
    arm_dir = HERE / ("strict-device-chain" if args.device_chain else "strict") / args.arm
    if arm_dir.exists():
        raise FileExistsError(f"strict arm already exists: {arm_dir}")
    state = arm_dir / "state"
    state.mkdir(parents=True)
    command = [str(EXE), "--model", str(TARGET), "--device", "0", "--native-gpu",
               "--fused-moe", "--native-dense-exact", "--native-attention-exact",
               "--gpu-embedding", "--nextn-hidden", "--mtp-model", str(MTP[args.arm]),
               "--gpu-sampling", "--defer-mtp-catchup", "--draft-count", "2",
               "--draft-ubatch", "2", "--assert-sequential", "--draft-state-dir", str(state)]
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(("STRATA_A3B_", "GGML_CUDA_"))}
    environment.update(FLAGS)
    if args.device_chain:
        environment["STRATA_A3B_MTP_DEVICE_CHAIN_K2"] = "1"
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    beginning = time.monotonic()
    with (arm_dir / "strict.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, check=False, timeout=900,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    manifest = {
        "arm": args.arm,
        "command": command,
        "env": {**FLAGS, **({"STRATA_A3B_MTP_DEVICE_CHAIN_K2": "1"} if args.device_chain else {})},
        "cuda_visible_devices": "0",
        "elapsed_s": time.monotonic() - beginning,
        "exit_code": result.returncode,
        "fixture_sha256": sha256(EXE),
        "mtp_sha256": sha256(MTP[args.arm]),
        "target_sha256_expected_from_provenance": "66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66",
        "state_files": sorted(p.name for p in state.glob("*")),
    }
    (arm_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in
                      ("arm", "elapsed_s", "exit_code", "fixture_sha256", "mtp_sha256", "state_files")},
                     indent=2))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
