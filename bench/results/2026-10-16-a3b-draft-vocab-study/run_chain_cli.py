"""Short opt-in dependent K2 CUDA-chain check with the isolated A3B engine."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EXE = ROOT / "build-a3b-draft-vocab-cuda/bin/Release/strata-moe.exe"
TARGET = ROOT / "models/a3b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
MTP = ROOT / "models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0-DV-ASCII-CJK-PUNCT.gguf"
OUT = HERE / "chain-cli"


def main() -> int:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    args = [str(EXE), "--model", str(TARGET), "--ctx", "4096", "--batch", "2048",
            "--predict", "64", "--device", "0", "--vram-mib", "0",
            "--mtp-model", str(MTP), "--mtp-draft", "2", "--mtp-ubatch", "512",
            "--exact-verify", "--native-gpu", "--gpu-embedding", "--gpu-sampling",
            "--defer-mtp-catchup", "--fused-moe", "--native-dense-exact",
            "--strata-native-experts", "--prompt",
            "Explain in detail how GPU memory caches route MoE experts and list at least ten tradeoffs."]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_A3B_", "GGML_CUDA_"))}
    env.update({
        "STRATA_A3B_MTP_DEVICE_CHAIN_K2": "1",
        "STRATA_A3B_Q6_GROUPED_EXACT": "1",
        "STRATA_A3B_MTP_KV_ONLY_DIAGNOSTIC": "0",
        "STRATA_A3B_MTP_PROPOSAL_TRACE": "1",
        "STRATA_A3B_FUSED_GU": "1",
        "STRATA_A3B_DOWN_MODE": "1",
        "STRATA_A3B_Q6_REUSE_EXACT": "1",
        "STRATA_A3B_MTP_KV_ONLY": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    start = time.monotonic()
    with (OUT / "output.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(args, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                check=False, timeout=300,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    content = (OUT / "output.log").read_text(encoding="utf-8", errors="replace")
    chain = re.search(r"A3B MTP device K2 chain: calls (\d+) fallback (\d+);", content)
    details = {"command": args, "env": {k: v for k, v in env.items() if k.startswith("STRATA_A3B_")},
               "elapsed_s": time.monotonic() - start, "exit_code": result.returncode,
               "device_chain_calls": int(chain.group(1)) if chain else None,
               "device_chain_fallbacks": int(chain.group(2)) if chain else None}
    (OUT / "manifest.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: details[k] for k in ("elapsed_s", "exit_code", "device_chain_calls",
                                            "device_chain_fallbacks")}, indent=2))
    return 0 if result.returncode == 0 and details["device_chain_calls"] and \
                details["device_chain_fallbacks"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
