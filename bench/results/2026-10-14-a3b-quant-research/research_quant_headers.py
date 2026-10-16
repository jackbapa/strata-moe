"""Read only GGUF header ranges and pinned Hugging Face LFS metadata."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sys

import requests

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import GGUFFile

OUT = Path(__file__).resolve().parent
REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"
REV = "a483e9e6cbd595906af30beda3187c2663a1118c"
EXPERT = re.compile(r"blk\.[0-9]+\.ffn_(gate|up|down|gate_up)_exps\.weight")


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def inspect(path):
    f = GGUFFile(path)
    total = sum(t.expected_bytes() for t in f.tensors)
    experts = [t for t in f.tensors if EXPERT.fullmatch(t.name)]
    expert_bytes = sum(t.expected_bytes() for t in experts)
    embd = sum(t.expected_bytes() for t in f.tensors if t.name == "token_embd.weight")
    types = {}
    for t in experts:
        s = types.setdefault(t.type_name, {"count": 0, "bytes": 0})
        s["count"] += 1
        s["bytes"] += t.expected_bytes()
    nonexpert_types = {}
    for t in f.tensors:
        if not EXPERT.fullmatch(t.name):
            s = nonexpert_types.setdefault(t.type_name, {"count": 0, "bytes": 0})
            s["count"] += 1
            s["bytes"] += t.expected_bytes()
    return {"data_start": f.data_start, "tensor_count": len(f.tensors), "tensor_bytes": total,
            "routed_tensor_bytes": expert_bytes, "nonexpert_tensor_bytes": total - expert_bytes,
            "token_embedding_bytes": embd, "gpu_weight_bytes_if_input_embedding_on_cpu": total - embd,
            "routed_types": types, "nonexpert_types": nonexpert_types, "all_types": f.by_type(),
            "tokenizer_hashes": {k: digest(v) for k, v in f.metadata.items() if k.startswith("tokenizer.")},
            "architecture_metadata": {k: v for k, v in f.metadata.items()
                                      if k.startswith("qwen35moe.") or k == "general.architecture"},
            "has_mtp_tensors": any(t.name.startswith("blk.40.") or "nextn" in t.name for t in f.tensors)}


def fetch(item):
    filename = item["rfilename"]
    path = OUT / (filename + ".header")
    if not path.exists():
        n = 12 * 1024 * 1024
        url = f"https://huggingface.co/{REPO}/resolve/{REV}/{filename}?download=true"
        r = requests.get(url, headers={"Range": f"bytes=0-{n-1}"}, timeout=120)
        r.raise_for_status()
        if r.status_code != 206 or len(r.content) != n or not r.headers.get("Content-Range", "").startswith(f"bytes 0-{n-1}/"):
            raise RuntimeError(f"invalid range response: {r.status_code}, {len(r.content)}, {r.headers.get('Content-Range')}")
        path.write_bytes(r.content)
    result = {"file": filename, "revision": REV, "file_bytes": item["size"],
              "lfs_sha256": item["lfs"]["sha256"], **inspect(path)}
    print(filename, result["file_bytes"], result["routed_types"], flush=True)
    return result


if __name__ == "__main__":
    api = f"https://huggingface.co/api/models/{REPO}/revision/{REV}?blobs=true"
    r = requests.get(api, timeout=45)
    r.raise_for_status()
    repo = r.json()
    items = [x for x in repo["siblings"] if any(q in x["rfilename"] for q in ["Q4_K_M.gguf", "Q4_K_S.gguf", "IQ4_XS.gguf"])]
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, items))
    results.append({"file": "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf", "file_bytes": 26456194016,
                    "revision": REV, "lfs_sha256": "c13ce26253ea334df472bd8fbd2d6da66d8a41195c17f6fcbf44c4d20ece0932",
                    **inspect(ROOT / "models/a3b/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf")})
    mtp = inspect(ROOT / "models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0.gguf")
    q5 = results[-1]
    for item in results:
        item["tokenizer_metadata_identical_to_q5"] = item["tokenizer_hashes"] == q5["tokenizer_hashes"]
        item["architecture_metadata_identical_to_q5"] = item["architecture_metadata"] == q5["architecture_metadata"]
        item["target_and_mtp_gpu_weight_bytes"] = item["gpu_weight_bytes_if_input_embedding_on_cpu"] + mtp["gpu_weight_bytes_if_input_embedding_on_cpu"]
    (OUT / "quant-header-results.json").write_text(json.dumps({"source_api": api, "repo_revision": repo["sha"], "models": results, "mtp": mtp}, indent=2) + "\n")
