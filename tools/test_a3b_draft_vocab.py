"""CPU-only checks for A3B draft-vocabulary GGUF construction and ID safety."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

from a3b_draft_vocab import (
    check_pair, make_subset_gguf, punctuation_piece, read_ids, required_ids,
    tokenizer_fingerprint, write_ids,
)
from gguf_reader import GGUFFile
from gguf_writer import GGUFWriter, TYPE_IDS, _Tensor


def fixture(path: Path, *, swapped: bool = False, target: bool = False) -> tuple[GGUFFile, bytes, bytes]:
    writer = GGUFWriter()
    writer.add("general.architecture", "qwen35moe")
    writer.add("qwen35moe.block_count", 40 if target else 41, "u32")
    writer.add("qwen35moe.embedding_length", 2048, "u32")
    if not target:
        writer.add("qwen35moe.nextn_predict_layers", 1, "u32")
    writer.add("tokenizer.ggml.model", "gpt2")
    writer.add("tokenizer.ggml.pre", "qwen35")
    writer.add("tokenizer.ggml.tokens", [f"t{i}" for i in range(8)] if not swapped else
               ["changed", *[f"t{i}" for i in range(1, 8)]], "array:string")
    writer.add("tokenizer.ggml.merges", ["t0 t1"], "array:string")
    writer.add("tokenizer.ggml.token_type", [1, 1, 1, 3, 1, 1, 1, 1], "array:i32")
    writer.add("tokenizer.ggml.eos_token_id", 3, "u32")
    row_bytes = 2048 // 32 * 34
    head = b"".join(bytes([i]) * row_bytes for i in range(8))
    embd = b"".join(bytes([i + 16]) * row_bytes for i in range(8))
    writer.tensors.extend([
        _Tensor("output.weight", [2048, 8], "Q8_0", head),
        _Tensor("token_embd.weight", [2048, 8], "Q8_0", embd),
        _Tensor("blk.40.nextn.eh_proj.weight", [4], "F16", struct.pack("<4e", 1, 2, 3, 4)),
    ])
    TYPE_IDS["Q8_0"] = 8
    writer.write(path)
    return GGUFFile(path), head, embd


def tensor_bytes(path: Path, name: str) -> bytes:
    model = GGUFFile(path)
    tensor = next(t for t in model.tensors if t.name == name)
    count = tensor.expected_bytes()
    if count is None:
        raise AssertionError(f"unknown tensor byte size: {name}")
    with path.open("rb") as fh:
        fh.seek(model.data_start + tensor.offset)
        return fh.read(count)


class DraftVocabTest(unittest.TestCase):
    def test_unicode_punctuation_pieces(self) -> None:
        for text in ("“", "”", "——", "  …", "（ ）"):
            self.assertTrue(punctuation_piece(text), text)
        for text in ("中文", "code()", "mix“quote"):
            self.assertFalse(punctuation_piece(text), text)

    def test_subset_copies_q8_rows_and_full_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            model, original_head, original_embd = fixture(base / "source.gguf")
            target_model, _, _ = fixture(base / "target.gguf", target=True)
            self.assertEqual(check_pair(target_model, model), 8)
            ids = [0, 3, 7]
            target = base / "subset.gguf"
            info = make_subset_gguf(model, ids, target)
            row = 2048 // 32 * 34
            self.assertEqual(tensor_bytes(target, "output.weight"),
                             b"".join(original_head[i * row:(i + 1) * row] for i in ids))
            self.assertEqual(tensor_bytes(target, "token_embd.weight"), original_embd)
            self.assertEqual(tensor_bytes(target, "blk.40.nextn.eh_proj.weight"),
                             struct.pack("<4e", 1, 2, 3, 4))
            self.assertEqual(tensor_bytes(target, "d2t"), struct.pack("<3q", *ids))
            self.assertEqual(GGUFFile(target).metadata, model.metadata)
            self.assertEqual(info["selected_rows"], 3)
            self.assertEqual(info["full_rows"], 8)
            self.assertEqual(hashlib.sha256(target.read_bytes()).hexdigest(), info["output_sha256"])

    def test_i32_map_keeps_the_same_q8_weights(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            model, original_head, original_embd = fixture(base / "source.gguf")
            ids = [0, 3, 7]
            i64 = base / "subset-i64.gguf"
            i32 = base / "subset-i32.gguf"
            make_subset_gguf(model, ids, i64)
            info = make_subset_gguf(model, ids, i32, map_type="i32")
            i64_map = next(t for t in GGUFFile(i64).tensors if t.name == "d2t")
            i32_map = next(t for t in GGUFFile(i32).tensors if t.name == "d2t")
            # The derived I32 file differs before the map payload only in the
            # d2t type field; no Q8 block or other tensor data is rewritten.
            self.assertEqual(i32_map.offset, i64_map.offset)
            self.assertEqual(GGUFFile(i32).metadata, GGUFFile(i64).metadata)
            boundary = GGUFFile(i32).data_start + i32_map.offset
            i32_prefix = i32.read_bytes()[:boundary]
            i64_prefix = i64.read_bytes()[:boundary]
            changed = [i for i, (a, b) in enumerate(zip(i32_prefix, i64_prefix)) if a != b]
            self.assertEqual(len(i32_prefix), len(i64_prefix))
            self.assertEqual(len(changed), 1)  # GGUF type ID 26 versus 27
            for tensor in model.tensors:
                self.assertEqual(tensor_bytes(i32, tensor.name), tensor_bytes(i64, tensor.name))
            self.assertEqual(tensor_bytes(i32, "token_embd.weight"), original_embd)
            row = 2048 // 32 * 34
            self.assertEqual(tensor_bytes(i32, "output.weight"),
                             b"".join(original_head[i * row:(i + 1) * row] for i in ids))
            self.assertEqual(tensor_bytes(i32, "d2t"), struct.pack("<3i", *ids))
            self.assertEqual(next(t for t in GGUFFile(i32).tensors if t.name == "d2t").type_name, "I32")
            self.assertEqual(info["d2t_type"], "i32")

    def test_ids_are_a3b_specific_unique_and_keep_special(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            target, _, _ = fixture(base / "target.gguf", target=True)
            draft, _, _ = fixture(base / "draft.gguf")
            path = base / "ids.bin"
            details = {"tokenizer_sha256": tokenizer_fingerprint(draft.metadata), "n_vocab": 8}
            write_ids(path, [0, 3, 7], details)
            self.assertEqual(read_ids(path, target, draft), [0, 3, 7])
            self.assertEqual(required_ids(target, draft), {3})
            path.write_bytes(struct.pack("<3i", 0, 0, 3))
            manifest = json.loads(path.with_suffix(".bin.json").read_text())
            manifest["ids_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".bin.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "sorted, unique"):
                read_ids(path, target, draft)
            path.write_bytes(struct.pack("<2i", 0, 7))
            manifest["ids_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".bin.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "special/end"):
                read_ids(path, target, draft)
            foreign, _, _ = fixture(base / "foreign.gguf", swapped=True)
            with self.assertRaisesRegex(ValueError, "tokenizer mismatch"):
                check_pair(target, foreign)


if __name__ == "__main__":
    unittest.main()
