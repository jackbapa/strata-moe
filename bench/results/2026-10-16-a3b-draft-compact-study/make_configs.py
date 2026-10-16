"""Freeze two 50K/256 configs differing only in the compact GPU sampler flag."""
from __future__ import annotations

import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SELECTED = ROOT / "strata-a3b-iq3s-strata-optimized.json"
ENGINE = ROOT / "build-a3b-draft-compact-cuda/bin/Release/strata-moe.exe"
DRAFT = ROOT / "models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0-DV-ASCII-CJK-PUNCT-I32.gguf"


def main() -> None:
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    args = selected["args"]
    if args[args.index("--ctx") + 1] != "262144" or \
       args[args.index("--mtp-draft") + 1] != "2" or \
       "--strata-native-experts" not in args or \
       selected["env"].get("STRATA_A3B_MTP_DEVICE_CHAIN_K2") == "1":
        raise RuntimeError("selected service no longer has the intended 256K / IQ / ordinary K2 settings")
    for name, flag in (("i32_scatter", "0"), ("i32_compact", "1")):
        path = HERE / f"{name}.snapshot.json"
        if path.exists():
            raise FileExistsError(path)
        config = copy.deepcopy(selected)
        config["exe"] = str(ENGINE)
        config["args"][config["args"].index("--mtp-model") + 1] = str(DRAFT)
        config["env"]["STRATA_A3B_DRAFT_COMPACT_GPU"] = flag
        config["log"] = str(HERE / f"{name}.engine.log")
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
