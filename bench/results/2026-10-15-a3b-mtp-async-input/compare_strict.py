"""Compare K2 sync, async, and forced-fallback strict diagnostic outputs."""

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("off", "forced", "async")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    states = {}
    traces = {}
    classes = {}
    for arm in ARMS:
        base = HERE / arm
        metadata = json.loads((base / "metadata.json").read_text())
        assert metadata["returncode"] == 0
        assert metadata["binary_sha256"] == metadata["binary_sha256_after"]
        states[arm] = {p.name: sha(p) for p in sorted((base / "state").glob("*.bin"))}
        log = (base / "strict.log").read_text(errors="replace")
        traces[arm] = [line for line in log.splitlines()
                       if line.startswith("A3B MTP proposal trace:")]
        classes[arm] = [tuple(m.groups()) for m in re.finditer(
            r"MTP class limit=(\d+) output=(\d+) offered=(\d+) accepted=(\d+)", log)]
        assert len(states[arm]) == 16, (arm, len(states[arm]))
        assert len(traces[arm]) == 29, (arm, len(traces[arm]))
        assert len(classes[arm]) == 8, (arm, len(classes[arm]))
        if arm == "forced":
            assert "second step rebuilding with ordinary inputs after pinned DMA" in log
        elif arm == "async":
            assert "pinned CUDA token, hidden, position, output and KV input staging active" in log
            assert "second step rebuilding with ordinary inputs after pinned DMA" not in log
    binary_shas = [json.loads((HERE / arm / "metadata.json").read_text())["binary_sha256"]
                   for arm in ARMS]
    report = {
        "arms": ARMS,
        "binary_sha256": binary_shas[0],
        "same_binary": len(set(binary_shas)) == 1,
        "states_per_arm": {a: len(states[a]) for a in ARMS},
        "state_bytes_identical": states["off"] == states["forced"] == states["async"],
        "proposal_lines_per_arm": {a: len(traces[a]) for a in ARMS},
        "proposals_identical": traces["off"] == traces["forced"] == traces["async"],
        "class_lines_per_arm": {a: len(classes[a]) for a in ARMS},
        "class_fields_identical": classes["off"] == classes["forced"] == classes["async"],
        "state_sha256": states["off"],
    }
    (HERE / "strict-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "state_sha256"}, indent=2))
    assert all(report[k] for k in ("same_binary", "state_bytes_identical",
                                   "proposals_identical", "class_fields_identical"))


if __name__ == "__main__":
    main()
