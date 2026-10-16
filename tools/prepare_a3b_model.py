"""Copy selected quantized GGUF tensors verbatim; never dequantize/requantize.

The default recipe preserves every Q4_K_S tensor except output.weight, which
is copied from the verified Q8-headed Q5_K_M model. The original files remain
unchanged. The resulting Q8 head can be shared exactly with the trained MTP.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct

from gguf_reader import GGUFFile


def replace_tensors(base_path, donor_path, output_path, names):
    base_path, donor_path, output_path = map(lambda p: Path(p).resolve(), (base_path, donor_path, output_path))
    if output_path in (base_path, donor_path) or output_path.exists():
        raise FileExistsError("output must be a new file, distinct from both input models")
    base, donor = GGUFFile(base_path), GGUFFile(donor_path)
    for key in ("general.architecture", "qwen35moe.block_count", "qwen35moe.embedding_length"):
        if base.metadata.get(key) != donor.metadata.get(key):
            raise ValueError(f"incompatible donor: {key}")
    tokenizer = {k: v for k, v in base.metadata.items() if k.startswith("tokenizer.")}
    other_tokenizer = {k: v for k, v in donor.metadata.items() if k.startswith("tokenizer.")}
    if tokenizer != other_tokenizer:
        raise ValueError("tokenizer metadata differs")
    available = {t.name: t for t in donor.tensors}
    originals = {t.name: t for t in base.tensors}
    for name in names:
        if name not in available or name not in originals or available[name].shape != originals[name].shape:
            raise ValueError(f"missing or incompatible tensor: {name}")
    # Retain metadata bytes exactly, including all scalar/array value types.
    header = bytearray(base_path.open("rb").read(base.data_start))
    fields = {}
    with base_path.open("rb") as f:
        f.seek(8)
        nt, nkv = struct.unpack("<QQ", f.read(16))
        for _ in range(nkv):
            base._str(f)
            base._value(f)
        for _ in range(nt):
            name = base._str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            f.seek(nd * 8, 1)
            fields[name] = f.tell()
            f.seek(12, 1)
    offset = 0
    entries = []
    for original in base.tensors:
        selected = available[original.name] if original.name in names else original
        nbytes = selected.expected_bytes()
        if nbytes is None:
            raise ValueError(f"unsupported block geometry: {selected.name} {selected.type_name}")
        offset = (offset + base.alignment - 1) // base.alignment * base.alignment
        struct.pack_into("<IQ", header, fields[original.name], selected.type_id, offset)
        entries.append((original, selected, offset, nbytes))
        offset += nbytes
    partial = output_path.with_suffix(output_path.suffix + ".building")
    if partial.exists():
        raise FileExistsError(f"unfinished output already exists: {partial}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    whole_hash = hashlib.sha256()
    tensor_hashes = {}
    with base_path.open("rb") as source, donor_path.open("rb") as replacement, partial.open("wb") as out:
        out.write(header)
        whole_hash.update(header)
        for index, (original, selected, offset, remaining) in enumerate(entries):
            padding = base.data_start + offset - out.tell()
            zeros = b"\0" * padding
            out.write(zeros)
            whole_hash.update(zeros)
            donated = original.name in names
            reader, metadata = (replacement, donor) if donated else (source, base)
            reader.seek(metadata.data_start + selected.offset)
            digest = hashlib.sha256()
            while remaining:
                chunk = reader.read(min(16 * 1024 * 1024, remaining))
                if not chunk:
                    raise EOFError(f"short tensor: {selected.name}")
                out.write(chunk)
                whole_hash.update(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
            tensor_hashes[original.name] = digest.hexdigest()
            if donated or index % 100 == 0:
                print(f"{index + 1}/{len(entries)} {selected.name} {selected.type_name}: raw bytes copied", flush=True)
    checked = GGUFFile(partial)
    if checked.metadata != base.metadata:
        raise RuntimeError("metadata changed during tensor transplant")
    for tensor, (_, selected, offset, nbytes) in zip(checked.tensors, entries):
        if (tensor.name, tensor.shape, tensor.type_id, tensor.offset, tensor.expected_bytes()) != (
            selected.name, selected.shape, selected.type_id, offset, nbytes):
            raise RuntimeError(f"output tensor table mismatch: {tensor.name}")
    partial.replace(output_path)
    record = {"recipe": "verbatim GGUF tensor transplant; no requantization",
              "base": str(base_path), "base_bytes": base_path.stat().st_size,
              "donor": str(donor_path), "donor_bytes": donor_path.stat().st_size,
              "output": str(output_path), "output_bytes": output_path.stat().st_size,
              "output_sha256": whole_hash.hexdigest(), "replaced": [
                  {"name": name, "before_type": originals[name].type_name,
                   "after_type": available[name].type_name, "sha256": tensor_hashes[name]} for name in names],
              "tensor_sha256": tensor_hashes}
    output_path.with_suffix(".recipe.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--donor", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tensor", action="append", default=None)
    args = p.parse_args()
    result = replace_tensors(args.base, args.donor, args.output, args.tensor or ["output.weight"])
    print(json.dumps({k: v for k, v in result.items() if k != "tensor_sha256"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
