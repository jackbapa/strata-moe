"""Read a pinned remote GGUF header and optionally hash only its shared tables.

Never downloads the complete model. Shared-table hashing reads about 1.01 GiB.
Run from the Strata-Moe root; no GPU or inference APIs are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from gguf_reader import GGUFFile

SHARED = ("token_embd.weight", "output.weight", "output_norm.weight")


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="ggml-org/Qwen3.6-35B-A3B-GGUF")
    parser.add_argument("--revision", default="baec3ebee244827cda0f4557eafa8b28f7545fa6")
    parser.add_argument("--filename", default="Qwen3.6-35B-A3B-Q8_0.gguf")
    parser.add_argument("--hash-shared", action="store_true")
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    label = args.repo.split("/")[0] + "-" + args.filename.removesuffix(".gguf")
    api = f"https://huggingface.co/api/models/{args.repo}/revision/{args.revision}?blobs=true"
    with urllib.request.urlopen(api, timeout=60) as response:
        metadata = json.load(response)
    assert metadata["sha"] == args.revision
    record = next(x for x in metadata["siblings"] if x["rfilename"] == args.filename)
    size = record["size"]
    url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{args.filename}?download=true"
    (folder / (label + "-api.json")).write_text(json.dumps({"api": api, "revision": metadata["sha"],
                                                           "file": record}, indent=2) + "\n", encoding="utf-8")

    def read_range(start, length):
        end = start + length - 1
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(req, timeout=90) as response:
                    expected = f"bytes {start}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError(f"range refused: status={response.status}, range={response.headers.get('Content-Range')}")
                    block = response.read(length + 1)
                if len(block) != length:
                    raise RuntimeError(f"short range: {len(block)} != {length}")
                return block
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(0.5 * (attempt + 1))
        raise AssertionError

    header = read_range(0, 16 * 1024 * 1024)
    remote = GGUFFile.__new__(GGUFFile)
    remote.path, remote.metadata, remote.tensors = Path(args.filename), {}, []
    remote.version, remote.alignment = 0, 32
    remote._parse(io.BytesIO(header))
    remote.data_start = remote._data_start
    assert remote.data_start <= len(header)
    mtp = GGUFFile(ROOT / "models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0.gguf")
    original = GGUFFile(ROOT / "models/a3b/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf")
    tokenizer = lambda model: {k: v for k, v in model.metadata.items() if k.startswith("tokenizer.")}
    experts = [t for t in remote.tensors if "_exps.weight" in t.name]
    tensors = {t.name: t for t in remote.tensors}
    mtp_tensors = {t.name: t for t in mtp.tensors}
    summary = {"repo": args.repo, "revision": args.revision, "filename": args.filename,
               "download_url": url, "api_url": api, "file_bytes": size,
               "published_lfs_sha256": record["lfs"]["sha256"], "full_file_downloaded_or_hashed": False,
               "header_range_bytes": len(header), "header_data_start": remote.data_start,
               "header_sha256": hashlib.sha256(header[:remote.data_start]).hexdigest(),
               "architecture": remote.metadata.get("general.architecture"),
               "geometry": {k: v for k, v in remote.metadata.items() if k.startswith("qwen35moe.")},
               "all_tensor_types": dict(Counter(t.type_name for t in remote.tensors)),
               "expert_tensor_types": dict(Counter(t.type_name for t in experts)),
               "expert_tensor_count": len(experts), "expert_bytes": sum(t.expected_bytes() for t in experts),
               "tokenizer_sha256": json_hash(tokenizer(remote)),
               "tokenizer_metadata_equal_to_mtp": tokenizer(remote) == tokenizer(mtp),
               "tokenizer_metadata_equal_to_existing_q5": tokenizer(remote) == tokenizer(original),
               "tensors": [{"name": t.name, "type": t.type_name, "shape": t.shape,
                            "offset": t.offset, "bytes": t.expected_bytes()} for t in remote.tensors]}
    summary["shared_tables"] = [{"name": name, "type": tensors[name].type_name,
                                 "shape": tensors[name].shape, "bytes": tensors[name].expected_bytes(),
                                 "shape_and_type_equal_to_mtp": (tensors[name].shape, tensors[name].type_id) ==
                                     (mtp_tensors[name].shape, mtp_tensors[name].type_id)} for name in SHARED]
    destination = folder / (label + "-audit.json")
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("tensors", "geometry")}, ensure_ascii=False), flush=True)
    if not args.hash_shared:
        return

    def local_hash(tensor):
        digest = hashlib.sha256()
        with mtp.path.open("rb") as source:
            source.seek(mtp.data_start + tensor.offset)
            remaining = tensor.expected_bytes()
            while remaining:
                data = source.read(min(16 * 1024 * 1024, remaining))
                if not data:
                    raise EOFError("local MTP tensor truncated")
                digest.update(data)
                remaining -= len(data)
        return digest.hexdigest()

    with ThreadPoolExecutor(max_workers=8) as pool:
        for table in summary["shared_tables"]:
            tensor = tensors[table["name"]]
            length, chunk = tensor.expected_bytes(), 16 * 1024 * 1024
            pieces = [(remote.data_start + tensor.offset + pos, min(chunk, length - pos))
                      for pos in range(0, length, chunk)]
            digest = hashlib.sha256()
            for block in pool.map(lambda args: read_range(*args), pieces):
                digest.update(block)
            table["remote_sha256"] = digest.hexdigest()
            table["local_mtp_sha256"] = local_hash(mtp_tensors[table["name"]])
            table["strict_payload_equal_to_mtp"] = table["remote_sha256"] == table["local_mtp_sha256"]
            destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(table, ensure_ascii=False), flush=True)
    summary["shared_payload_range_bytes"] = sum(t["bytes"] for t in summary["shared_tables"])
    summary["all_three_shared_tables_strictly_equal_to_mtp"] = all(
        x["shape_and_type_equal_to_mtp"] and x["strict_payload_equal_to_mtp"] for x in summary["shared_tables"])
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
