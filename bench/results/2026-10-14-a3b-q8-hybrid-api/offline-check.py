"""Check the frozen config and request-log parser without HTTP/GPU access."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("hybrid_latency_runner", HERE / "run-latency.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

source = json.loads((ROOT / "strata-a3b-q8.json").read_bytes())
config = json.loads((HERE / "config.json").read_bytes())
expected_args = list(source["args"])
expected_args.remove("--gpu-only-experts")
expected_args += ["--hybrid-experts", "--cpu-expert-pool", "--threads", "8", "--pcie-frac", "0.25"]
assert config["args"] == expected_args
assert config["env"] == source["env"] and config["sampling"] == source["sampling"]
assert config["idle_unload_s"] == 0 and config["host"] == "127.0.0.1" and config["port"] == 8096
assert hashlib.sha256(Path(config["exe"]).read_bytes()).hexdigest() == runner.EXPECTED_EXE_SHA256

# Protocol fixture, not a real inference result. The same prompt/output counts
# intentionally occur in every block, so prior-request indexing must be right.
def block(calls):
    return f"""A3B: prompt 32 tokens 0.500s 64.00 tok/s; output 256 tokens 5.000s 51.20 tok/s
A3B cache: 16384.0 MiB; GPU rows 180 CPU rows 60; hit 120 miss 120; uploaded 64.0 MiB
A3B backend: route reads 50; native layout calls 0; single-column calls 1
A3B transfer: prefill 10.0 MiB; decode 6.4 MiB; decode GPU rows 18 CPU rows 6; decode hit 12 miss 12
A3B hybrid: calls {calls}; PCIe fraction 0.2500; GPU miss rows 2 experts 2; input allocations 1 bytes 65536 fallbacks 0
A3B CPU pool: calls 1 jobs 2 unique inputs 1 experts 2; input quant 1.125 ms gate/up 2.250 ms hidden quant 0.125 ms down 3.500 ms
A3B cache decay: prefill 0; decode 5
A3B MTP: windows 3 offered 9 accepted 5; draft 2.0 ms catchup 1.0 ms
A3B MTP adaptive: K0=1/2.0ms K1=1/3.0ms K2=1/4.0ms
"""

results = [{"trial": f"repeat{i}", "case_id": "protocol-fixture", "usage": {"prompt_tokens": 32, "completion_tokens": 256}}
           for i in (1, 2)]
parsed = runner.extract_counters(("[strata] test-only engine started\n" + block(10) + block(20) + block(30)).encode(), results, 1)
assert [row["hybrid_calls"] for row in parsed["requests"]] == [20, 30]
assert parsed["requests"][0]["cpu_pool_decode_down_ms"] == 3.5
assert parsed["requests"][0]["cpu_pool_jobs_match_decode_cpu_rows"]
assert parsed["requests"][0]["mtp_acceptance"] == 5 / 9
try:
    runner.extract_counters(block(1).replace("A3B CPU pool:", "missing-pool:").encode(), results[:1], 0)
except RuntimeError:
    pass
else:
    raise AssertionError("missing CPU pool counters were accepted")
print("PASS offline: frozen config/args/hash, log request indexing, CPU pool phases, MTP acceptance and missing-counter rejection; no HTTP/GPU access")
