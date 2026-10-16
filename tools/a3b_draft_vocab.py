"""Build an opt-in Qwen3.6-A3B MTP draft-head subset from its own GGUF tokenizer.

The derived MTP GGUF keeps every tensor byte-identical except output.weight,
whose selected Q8_0 rows are copied without requantization, and adds a
``d2t`` mapping (I64 by default, or I32 with ``--map-type i32``). The full
token_embd.weight and tokenizer stay intact.
The ordinary MTP GGUF remains the default; pass the derived file explicitly
as --mtp-model after building a binary with the QWEN35MOE d2t graph branch.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys
import unicodedata

from draft_vocab import GROUPS, SCRIPTS, covering, corpus_counts, outside, scripts_of, token_text
from gguf_reader import GGUFFile, GGUF_MAGIC
from strata_tokenizer import Tokenizer
from a3b import MTP_BYTES, MTP_FILE, MTP_SHA256

TOKENIZER_KEYS = (
    "tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens",
    "tokenizer.ggml.merges", "tokenizer.ggml.token_type",
)
ROW_BLOCK = 32
Q8_BLOCK_BYTES = 34
COPY_CHUNK = 16 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(COPY_CHUNK):
            digest.update(block)
    return digest.hexdigest()


def tokenizer_fingerprint(meta: dict) -> str:
    payload = json.dumps([meta.get(k) for k in TOKENIZER_KEYS],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_pair(target: GGUFFile, draft: GGUFFile) -> int:
    for name, model in (("target", target), ("draft", draft)):
        if model.metadata.get("general.architecture") != "qwen35moe":
            raise ValueError(f"{name}: expected qwen35moe GGUF")
    if target.metadata.get("qwen35moe.block_count") != 40 or \
       draft.metadata.get("qwen35moe.block_count") != 41 or \
       draft.metadata.get("qwen35moe.nextn_predict_layers") != 1 or \
       target.metadata.get("qwen35moe.embedding_length") != 2048 or \
       draft.metadata.get("qwen35moe.embedding_length") != 2048:
        raise ValueError("expected the matching 40-layer A3B target and 41-layer single-NextN draft")
    for key in TOKENIZER_KEYS:
        if target.metadata.get(key) != draft.metadata.get(key):
            raise ValueError(f"target/draft tokenizer mismatch: {key}")
    n_vocab = len(draft.metadata["tokenizer.ggml.tokens"])
    if not n_vocab or len(draft.metadata["tokenizer.ggml.token_type"]) != n_vocab:
        raise ValueError("incomplete draft tokenizer")
    tensors = {t.name: t for t in draft.tensors}
    output = tensors.get("output.weight")
    embed = tensors.get("token_embd.weight")
    if not output or output.type_name != "Q8_0" or output.shape != [2048, n_vocab]:
        raise ValueError("expected the matching 2048 x full-vocab Q8_0 MTP output.weight")
    if not embed or embed.shape != output.shape:
        raise ValueError("MTP token_embd.weight must retain every full-vocabulary row")
    if "blk.0.attn_norm.weight" in tensors or "blk.40.nextn.shared_head_head.weight" in tensors:
        raise ValueError("draft GGUF must be an MTP-only A3B file using output.weight")
    if "d2t" in tensors:
        raise ValueError("draft GGUF already has a vocabulary subset")
    return n_vocab


def required_ids(target: GGUFFile, draft: GGUFFile) -> set[int]:
    kinds = draft.metadata["tokenizer.ggml.token_type"]
    ids = {i for i, kind in enumerate(kinds) if kind != 1}
    for model in (target, draft):
        ids.update(int(v) for k, v in model.metadata.items()
                   if k.startswith("tokenizer.ggml.") and k.endswith("_token_id"))
    return ids


def punctuation_piece(text: str) -> bool:
    return bool(text) and all(ch.isspace() or unicodedata.category(ch).startswith("P") for ch in text)


def choose_ids(target: GGUFFile, draft: GGUFFile, *, ascii_all: bool, punctuation: bool,
               scripts: str, corpora: list[Path], coverage: float) -> tuple[list[int], dict]:
    n_vocab = check_pair(target, draft)
    if not (ascii_all or punctuation or scripts or corpora):
        raise ValueError("select at least one of --ascii, --punct, --add or --corpus")
    if not 0 < coverage <= 1:
        raise ValueError("--coverage must be in (0, 1]")
    want: set[str] = set()
    for part in filter(None, (x.strip() for x in scripts.split(","))):
        names = GROUPS.get(part, [part])
        if any(name not in SCRIPTS for name in names):
            raise ValueError(f"unknown script {part}")
        want.update(names)
    tok = Tokenizer.from_gguf(draft.path)
    chosen = required_ids(target, draft)
    if ascii_all or punctuation or want:
        for i in range(n_vocab):
            raw = tok.token_bytes(i)
            if ascii_all and raw and all(32 <= b < 127 or b in (9, 10, 13) for b in raw):
                chosen.add(i)
            if punctuation or want:
                text = token_text(tok, i)
                if text is not None:
                    if want and scripts_of(text) & want:
                        chosen.add(i)
                    if punctuation and punctuation_piece(text):
                        chosen.add(i)
    counts: Counter = corpus_counts(tok, corpora) if corpora else Counter()
    before_corpus = outside(counts, chosen) if counts else None
    if counts:
        chosen.update(covering(counts, coverage))
    ids = sorted(chosen)
    if len(ids) >= n_vocab:
        raise ValueError("selection did not reduce the draft head")
    details = {
        "source": "A3B GGUF tokenizer; no Flash-Next token IDs reused",
        "target": str(target.path.resolve()), "draft": str(draft.path.resolve()),
        "tokenizer_sha256": tokenizer_fingerprint(draft.metadata),
        "n_vocab": n_vocab, "selected_rows": len(ids),
        "required_special_rows": len(required_ids(target, draft)),
        "q8_output_bytes": len(ids) * output_row_bytes(draft),
        "full_q8_output_bytes": n_vocab * output_row_bytes(draft),
        "ascii": ascii_all, "punctuation": punctuation, "scripts": sorted(want),
        "corpora": [str(p.resolve()) for p in corpora], "coverage": coverage,
        "corpus_tokens": sum(counts.values()),
        "corpus_outside_before": before_corpus,
        "corpus_outside_after": outside(counts, ids) if counts else None,
    }
    return ids, details


def output_row_bytes(draft: GGUFFile) -> int:
    tensor = next(t for t in draft.tensors if t.name == "output.weight")
    assert tensor.shape[0] % ROW_BLOCK == 0
    return tensor.shape[0] // ROW_BLOCK * Q8_BLOCK_BYTES


def write_ids(path: Path, ids: list[int], details: dict) -> dict:
    if path.exists() or path.with_suffix(path.suffix + ".json").exists():
        raise FileExistsError(f"IDs or provenance already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = struct.pack(f"<{len(ids)}i", *ids)
    details = {**details, "ids_sha256": hashlib.sha256(raw).hexdigest()}
    path.write_bytes(raw)
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return details


def read_ids(path: Path, target: GGUFFile, draft: GGUFFile) -> list[int]:
    manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    raw = path.read_bytes()
    if not raw or len(raw) % 4 or manifest.get("ids_sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("missing, empty or damaged A3B draft ID file")
    if manifest.get("tokenizer_sha256") != tokenizer_fingerprint(draft.metadata) or \
       manifest.get("n_vocab") != check_pair(target, draft):
        raise ValueError("draft ID list was made for a different tokenizer/model pair")
    ids = list(struct.unpack(f"<{len(raw) // 4}i", raw))
    if ids != sorted(set(ids)) or ids[0] < 0 or ids[-1] >= manifest["n_vocab"]:
        raise ValueError("draft ID list must contain sorted, unique, in-range A3B IDs")
    if not required_ids(target, draft).issubset(ids):
        raise ValueError("draft ID list omits a special/end token")
    if len(ids) >= manifest["n_vocab"]:
        raise ValueError("draft ID list does not reduce the head")
    return ids


def _metadata_bytes(path: Path) -> bytearray:
    with path.open("rb") as fh:
        magic, version, _, n_kv = struct.unpack("<IIQQ", fh.read(24))
        if magic != GGUF_MAGIC or version != 3:
            raise ValueError("expected GGUF v3")
        parser = GGUFFile.__new__(GGUFFile)
        for _ in range(n_kv):
            parser._str(fh)
            parser._value(fh)
        end = fh.tell()
        fh.seek(0)
        return bytearray(fh.read(end))


def _tensor_entry(name: str, shape: list[int], type_id: int, offset: int) -> bytes:
    encoded = name.encode("utf-8")
    return (struct.pack("<Q", len(encoded)) + encoded +
            struct.pack("<I", len(shape)) + struct.pack(f"<{len(shape)}Q", *shape) +
            struct.pack("<IQ", type_id, offset))


def make_subset_gguf(draft: GGUFFile, ids: list[int], output_path: Path,
                     *, map_type: str = "i64") -> dict:
    """Copy selected Q8_0 output rows, every other tensor verbatim, and append d2t."""
    if map_type not in ("i32", "i64"):
        raise ValueError("d2t map type must be i32 or i64")
    full_rows = next(t.shape[1] for t in draft.tensors if t.name == "output.weight")
    if not ids or ids != sorted(set(ids)) or ids[0] < 0 or ids[-1] >= full_rows:
        raise ValueError("d2t IDs must be sorted, unique and inside the full vocabulary")
    if map_type == "i32" and (ids[0] < 0 or ids[-1] >= 2**31):
        raise ValueError("d2t token IDs do not fit I32")
    if output_path.exists() or output_path.resolve() == draft.path.resolve():
        raise FileExistsError("output must be a new GGUF path")
    partial = output_path.with_suffix(output_path.suffix + ".building")
    if partial.exists():
        raise FileExistsError(f"unfinished output already exists: {partial}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = _metadata_bytes(draft.path)
    struct.pack_into("<Q", prefix, 8, len(draft.tensors) + 1)
    row_bytes = output_row_bytes(draft)
    alignment = draft.alignment
    specs = []
    next_offset = 0
    table = bytearray()
    for tensor in draft.tensors:
        size = tensor.expected_bytes()
        if size is None:
            raise ValueError(f"unsupported tensor byte geometry: {tensor.name} {tensor.type_name}")
        shape = [tensor.shape[0], len(ids)] if tensor.name == "output.weight" else tensor.shape
        if tensor.name == "output.weight":
            size = row_bytes * len(ids)
        next_offset = (next_offset + alignment - 1) // alignment * alignment
        table += _tensor_entry(tensor.name, shape, tensor.type_id, next_offset)
        specs.append((tensor, next_offset, size, shape))
        next_offset += size
    next_offset = (next_offset + alignment - 1) // alignment * alignment
    d2t_type_id = 26 if map_type == "i32" else 27
    d2t_bytes = struct.pack(f"<{len(ids)}{'i' if map_type == 'i32' else 'q'}", *ids)
    table += _tensor_entry("d2t", [len(ids)], d2t_type_id, next_offset)
    data_start = (len(prefix) + len(table) + alignment - 1) // alignment * alignment
    head = prefix + table + bytearray(data_start - len(prefix) - len(table))
    digest = hashlib.sha256()
    with draft.path.open("rb") as src, partial.open("wb") as out:
        out.write(head)
        digest.update(head)
        for tensor, offset, size, _ in specs:
            padding = data_start + offset - out.tell()
            if padding < 0:
                raise RuntimeError("overlapping GGUF tensor offsets")
            zeros = bytes(padding)
            out.write(zeros)
            digest.update(zeros)
            if tensor.name == "output.weight":
                for token in ids:
                    src.seek(draft.data_start + tensor.offset + token * row_bytes)
                    block = src.read(row_bytes)
                    if len(block) != row_bytes:
                        raise EOFError("short Q8_0 output row")
                    out.write(block)
                    digest.update(block)
            else:
                src.seek(draft.data_start + tensor.offset)
                remaining = size
                while remaining:
                    block = src.read(min(COPY_CHUNK, remaining))
                    if not block:
                        raise EOFError(f"short GGUF tensor: {tensor.name}")
                    out.write(block)
                    digest.update(block)
                    remaining -= len(block)
        padding = data_start + next_offset - out.tell()
        zeros = bytes(padding)
        out.write(zeros)
        digest.update(zeros)
        out.write(d2t_bytes)
        digest.update(d2t_bytes)
    result = GGUFFile(partial)
    if result.metadata != draft.metadata or result.version != 3 or len(result.tensors) != len(draft.tensors) + 1:
        raise RuntimeError("derived MTP GGUF header failed round-trip validation")
    for actual, (original, offset, size, shape) in zip(result.tensors, specs):
        if (actual.name, actual.shape, actual.type_id, actual.offset, actual.expected_bytes()) != \
           (original.name, shape, original.type_id, offset, size):
            raise RuntimeError(f"derived tensor table mismatch: {original.name}")
    if result.tensors[-1].name != "d2t" or result.tensors[-1].type_id != d2t_type_id or \
       result.tensors[-1].shape != [len(ids)] or result.tensors[-1].offset != next_offset:
        raise RuntimeError("derived d2t tensor table mismatch")
    with partial.open("rb") as fh:
        fh.seek(result.data_start + next_offset)
        if fh.read(len(d2t_bytes)) != d2t_bytes:
            raise RuntimeError("derived d2t bytes mismatch")
    partial.replace(output_path)
    return {"source": str(draft.path.resolve()), "output": str(output_path.resolve()),
            "output_bytes": output_path.stat().st_size, "output_sha256": digest.hexdigest(),
            "selected_rows": len(ids), "full_rows": full_rows,
            "row_bytes": row_bytes, "quantization": "Q8_0 rows copied verbatim; no requantization",
            "embedding": "full token_embd.weight copied verbatim", "d2t_type": map_type}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    choose = sub.add_parser("select", help="generate A3B-specific ID list and provenance")
    build = sub.add_parser("build", help="write a separate, smaller MTP GGUF")
    for parser in (choose, build):
        parser.add_argument("--target", type=Path, required=True)
        parser.add_argument("--draft", type=Path, required=True)
        parser.add_argument("--ids", type=Path, required=True)
    choose.add_argument("--ascii", action="store_true", help="include tokens whose decoded bytes are ASCII")
    choose.add_argument("--punct", action="store_true", help="include standalone Unicode punctuation and whitespace pieces")
    choose.add_argument("--add", default="", help="scripts: han,kana,hangul,cjk_punct,cjk,cyrillic")
    choose.add_argument("--corpus", type=Path, nargs="*", default=[])
    choose.add_argument("--coverage", type=float, default=0.99)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--map-type", choices=("i32", "i64"), default="i64",
                       help="I32 enables opt-in compact GPU sampling; default I64 preserves the existing scatter model")
    build.add_argument("--dry-run", action="store_true", help="validate IDs and print expected size only")
    a = p.parse_args()
    target, draft = GGUFFile(a.target), GGUFFile(a.draft)
    if a.action == "select":
        ids, details = choose_ids(target, draft, ascii_all=a.ascii, punctuation=a.punct,
                                  scripts=a.add, corpora=a.corpus, coverage=a.coverage)
        print(json.dumps(write_ids(a.ids, ids, details), ensure_ascii=False, indent=2))
    else:
        ids = read_ids(a.ids, target, draft)
        if draft.path.name != MTP_FILE or draft.path.stat().st_size != MTP_BYTES:
            raise ValueError("build requires the pinned matching A3B Q8_0 MTP GGUF")
        if a.dry_run:
            print(json.dumps({"selected_rows": len(ids), "head_bytes": len(ids) * output_row_bytes(draft),
                              "head_savings_bytes": (len(draft.metadata["tokenizer.ggml.tokens"]) - len(ids)) * output_row_bytes(draft)}, indent=2))
        else:
            source_sha256 = sha256_file(draft.path)
            if source_sha256 != MTP_SHA256:
                raise ValueError("matching A3B Q8_0 MTP GGUF SHA-256 mismatch")
            info = make_subset_gguf(draft, ids, a.output, map_type=a.map_type)
            info["source_sha256"] = source_sha256
            info["ids_sha256"] = hashlib.sha256(a.ids.read_bytes()).hexdigest()
            a.output.with_suffix(a.output.suffix + ".recipe.json").write_text(
                json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
