"""Check A3B protocol boundaries while retaining the Flash-Next serializer."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import StrataEngine
from serve.a3b_protocol import make_sampling_keys

legacy = StrataEngine.sampling_keys
native = make_sampling_keys(legacy)
boundary = {"temperature": 0.7, "top_p": 1.0, "top_k": 0, "seed": 0}
old = legacy(boundary)
new = native(boundary)
assert "top_p=" not in old and "top_k=64" in old and "seed=" not in old
assert all(value in new.split() for value in ("temperature=0.7", "top_p=1.0", "top_k=0", "seed=0"))
ordinary = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "seed": 12345,
            "repetition_penalty": 1.1, "frequency_penalty": 0.2, "presence_penalty": 0.1}
assert set(native(ordinary).split()) == set(legacy(ordinary).split())
assert "top_k=127" in native({"top_k": 127})
assert native({}) == legacy({})
assert legacy(boundary) == old
print("PASS A3B full top-p, unbounded/wide top-k, seed zero, penalties; legacy serializer unchanged")
