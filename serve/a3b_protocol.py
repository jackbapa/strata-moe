"""Sampling values supported by the independent A3B engine's token protocol."""
import re


def make_sampling_keys(legacy):
    def serialize(sampling):
        keys = legacy(sampling)
        values = {}
        top_p = sampling.get("top_p")
        if isinstance(top_p, (int, float)) and not isinstance(top_p, bool) and 0 < top_p <= 1:
            values["top_p"] = repr(float(top_p))
        top_k = sampling.get("top_k")
        if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k >= 0:
            values["top_k"] = str(top_k)
        seed = sampling.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed <= 0xFFFFFFFF:
            values["seed"] = str(seed)
        for key, value in values.items():
            keys = re.sub(r" " + key + r"=[^ ]+", "", keys)
            keys += f" {key}={value}"
        return keys
    return serialize
