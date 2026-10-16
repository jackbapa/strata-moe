"""Recover per-request MTP proposal traces from retained engine logs."""

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent / "50k-256"
ARMS = ("selected", "k2-sync", "k2-async")
PATTERN = re.compile(
    r"^A3B MTP proposal trace: window (\d+) pos (-?\d+) input (-?\d+) offered (\d+)((?: -?\d+)*)$",
    re.MULTILINE,
)


def sha(lines):
    return hashlib.sha256(("\n".join(lines) + "\n").encode("ascii")).hexdigest()


def groups_for(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    groups = []
    for match in PATTERN.finditer(text):
        window = int(match[1])
        tokens = match[5].split()
        if len(tokens) != int(match[4]):
            raise ValueError(f"malformed trace record at {path}: {match[0]}")
        if window == 1:
            groups.append([])
        if not groups or window != len(groups[-1]) + 1:
            raise ValueError(f"non-sequential trace at {path}: {match[0]}")
        groups[-1].append(match[0])
    return groups


def main():
    result = {}
    for arm in ARMS:
        groups = groups_for(HERE / "engines" / f"{arm}.log")
        runs = sorted((HERE / "arms" / arm).glob("run[0-9][0-9][0-9].json"))
        assert len(groups) == len(runs) == 2, (arm, len(groups), len(runs))
        result[arm] = []
        for path, lines in zip(runs, groups):
            item = json.loads(path.read_text(encoding="utf-8"))
            result[arm].append({
                "trial": path.stem,
                "trace_windows": len(lines),
                "trace_sha256": sha(lines),
                "output_sha256": item["output_sha256"],
                "offered": item["mtp_offered_tokens"],
                "accepted": item["mtp_accepted_tokens"],
                "k2_calls": item.get("mtp_device_chain_k2_calls"),
                "k2_fallbacks": item.get("mtp_device_chain_k2_fallbacks"),
            })
    ref = groups_for(HERE / "engines" / "selected.log")[0]
    first_differences = {}
    for arm in ARMS:
        groups = groups_for(HERE / "engines" / f"{arm}.log")
        for index, lines in enumerate(groups, 1):
            mismatch = next((i for i, (left, right) in enumerate(zip(ref, lines), 1)
                             if left != right), None)
            if mismatch is None and len(ref) != len(lines):
                mismatch = min(len(ref), len(lines)) + 1
            first_differences[f"{arm}/run{index:03d}"] = {
                "window": mismatch,
                "selected_run001": ref[mismatch - 1] if mismatch and mismatch <= len(ref) else None,
                "this_trial": lines[mismatch - 1] if mismatch and mismatch <= len(lines) else None,
            }
    output = {"source": "retained engine logs; split at window 1 after each service restart",
              "runs": result, "first_difference_vs_selected_run001": first_differences}
    (HERE / "proposal-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
