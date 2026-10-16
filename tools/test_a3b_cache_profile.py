"""Offline cache-profile regressions: no GPU, network, or model download."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import a3b_cache_profile as profile
from gguf_reader import TensorInfo


def geometry_fixture():
    metadata = {"general.architecture": "qwen35moe"}
    metadata.update({"qwen35moe." + key: value for key, value in profile.GEOMETRY.items()})
    tensors, offset = [], 0
    for layer in range(40):
        for kind in ("gate", "up", "down"):
            shape = [512, 2048, 256] if kind == "down" else [2048, 512, 256]
            tensor = TensorInfo(f"blk.{layer}.ffn_{kind}_exps.weight", shape, 8, "Q8_0", offset)
            tensors.append(tensor)
            offset += tensor.expected_bytes()
    return SimpleNamespace(metadata=metadata, tensors=tensors, data_start=4096), offset + 4096


class CacheProfileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="a3b-cache-profile-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def trace(self, name, rows):
        path = self.root / name
        text = "backend\ttensor\texpert\troutes\tlfu_frequency\n"
        text += "".join(f"{backend}\tblk.{layer}.ffn_{kind}_exps.weight\t{expert}\t{routes}\t999999\n"
                        for backend, layer, kind, expert, routes in rows)
        path.write_text(text, encoding="utf-8")
        return path

    def test_fused_and_unfused_routes_count_once(self):
        fused = self.trace("fused.tsv", [(0, 0, "gate", 2, 11)])
        unfused = self.trace("unfused.tsv", [(0, 0, kind, 2, 11) for kind in ("gate", "up", "down")])
        fused_scores, _ = profile.aggregate_traces([fused])
        unfused_scores, _ = profile.aggregate_traces([unfused])
        self.assertEqual(fused_scores, unfused_scores)
        self.assertEqual(sum(row[2] for row in unfused_scores), 11)
        self.assertEqual(len(unfused_scores), 10240)

    def test_projection_backend_max_then_independent_trace_sum(self):
        first = self.trace("a.tsv", [(0, 3, "gate", 7, 11), (0, 3, "up", 7, 9),
                                      (0, 3, "down", 7, 11), (1, 3, "gate", 7, 12)])
        second = self.trace("b.tsv", [(0, 3, "gate", 7, 4)])
        scores, sources = profile.aggregate_traces([first, second])
        self.assertEqual(scores[0], (3, 7, 16))
        self.assertEqual(sum(row[2] for row in scores), 16)
        self.assertEqual([source["tsv_rows"] for source in sources], [4, 1])
        self.assertEqual(sources[0]["sha256"], profile.file_sha256(first))

    def test_duplicate_trace_path_rejected(self):
        trace = self.trace("a.tsv", [(0, 0, "gate", 0, 1)])
        with self.assertRaisesRegex(ValueError, "same trace path"):
            profile.aggregate_traces([trace, trace.parent / "." / trace.name])

    def test_invalid_route_values_and_geometry_rejected(self):
        rows = [(0, 40, "gate", 0, 1), (0, 0, "gate", 256, 1),
                (0, 0, "gate", 0, -1), (0, 0, "gate", 0, "1.0"),
                (0, 0, "gate", 0, 2**64)]
        for i, row in enumerate(rows):
            with self.subTest(row=row), self.assertRaises(ValueError):
                profile.read_trace(self.trace(f"bad{i}.tsv", [row]))

    def test_invalid_header_and_unknown_tensor_rejected(self):
        path = self.root / "bad.tsv"
        for text in ("tensor\texpert\nfoo\t0\n", "tensor\texpert\troutes\nfoo\t0\t1\n",
                     "tensor\texpert\troutes\n", "tensor\texpert\troutes\nblk.0.ffn_gate_exps.weight\t0\n"):
            path.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises(ValueError):
                profile.read_trace(path)

    def test_duplicate_and_extra_tsv_columns_rejected(self):
        for text in ("tensor\texpert\troutes\troutes\nblk.0.ffn_gate_exps.weight\t0\t1\t2\n",
                     "tensor\texpert\troutes\nblk.0.ffn_gate_exps.weight\t0\t1\t2\n"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                profile.parse_trace(text.encode())

    def test_aggregation_overflow_rejected(self):
        first = self.trace("a.tsv", [(0, 0, "gate", 0, profile.MAX_SCORE)])
        second = self.trace("b.tsv", [(0, 0, "gate", 0, 1)])
        with self.assertRaisesRegex(ValueError, "aggregated expert score overflow"):
            profile.aggregate_traces([first, second])
        separate = self.trace("c.tsv", [(0, 1, "gate", 0, 1)])
        with self.assertRaisesRegex(ValueError, "total expert score overflow"):
            profile.aggregate_traces([first, separate])

    def test_global_budget_is_uneven_and_does_not_cache_zeros(self):
        trace = self.trace("a.tsv", [(0, 0, "gate", 9, 100), (0, 0, "gate", 8, 90),
                                      (0, 1, "gate", 7, 80), (0, 39, "gate", 2, 0)])
        scores, _ = profile.aggregate_traces([trace])
        plan = profile.select_plan(scores, 2 * profile.BUNDLE_BYTES + profile.BUNDLE_BYTES - 1)
        self.assertEqual(plan["expert_ids"][0], [9, 8])
        self.assertEqual(plan["per_layer_capacity"][1], 0)
        self.assertEqual(plan["planned_bytes"], 2 * profile.BUNDLE_BYTES)
        self.assertAlmostEqual(plan["training_trace_coverage"], 190 / 270)
        all_positive = profile.select_plan(scores, 100000 * profile.BUNDLE_BYTES)
        self.assertEqual(all_positive["selected_experts"], 3)
        self.assertEqual(all_positive["training_trace_coverage"], 1)
        empty = profile.select_plan(scores, profile.BUNDLE_BYTES - 1)
        self.assertEqual(empty["planned_bytes"], 0)
        with self.assertRaises(ValueError):
            profile.select_plan(scores, -1)

    def test_ties_are_canonical_and_all_zero_coverage_is_zero(self):
        trace = self.trace("a.tsv", [(0, 2, "gate", 0, 10), (0, 0, "gate", 7, 10),
                                      (0, 0, "gate", 4, 10)])
        scores, _ = profile.aggregate_traces([trace])
        self.assertEqual(scores[:3], [(0, 4, 10), (0, 7, 10), (2, 0, 10)])
        zero = self.trace("zero.tsv", [(0, 0, "gate", 0, 0)])
        scores, _ = profile.aggregate_traces([zero])
        plan = profile.select_plan(scores, 100 * profile.BUNDLE_BYTES)
        self.assertEqual((plan["selected_experts"], plan["training_trace_coverage"]), (0, 0))

    def test_actual_q8_geometry_and_bundle_size(self):
        model, size = geometry_fixture()
        identity = profile.validate_q8_geometry(model, size)
        self.assertEqual(identity["bundle_bytes"], 3342336)
        model.tensors[0].shape.append(1)
        self.assertEqual(profile.validate_q8_geometry(model, size), identity)

    def test_wrong_q8_tensor_type_shape_count_duplicates_and_payload_rejected(self):
        original, size = geometry_fixture()
        mutations = [lambda m: setattr(m.tensors[0], "type_name", "Q4_K"),
                     lambda m: m.tensors[0].shape.__setitem__(0, 1024),
                     lambda m: m.tensors.pop(),
                     lambda m: m.tensors.append(copy.deepcopy(m.tensors[0])),
                     lambda m: setattr(m.tensors[-1], "offset", size),
                     lambda m: setattr(m.tensors[0], "offset", -1),
                     lambda m: setattr(m.tensors[0], "name", "blk.0.ffn_gate_up_exps.weight")]
        for mutation in mutations:
            model = copy.deepcopy(original)
            mutation(model)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                profile.validate_q8_geometry(model, size)
        with self.assertRaises(ValueError):
            profile.validate_q8_geometry(original, size - 1)

    def test_wrong_architecture_metadata_or_split_rejected(self):
        for key, value in [("general.architecture", "qwen3moe"), ("split.count", 2),
                           *[("qwen35moe." + key, expected + 1) for key, expected in profile.GEOMETRY.items()]]:
            model, size = geometry_fixture()
            model.metadata[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                profile.validate_q8_geometry(model, size)

    def test_generate_exact_bytes_provenance_and_whole_file_hash(self):
        model = self.root / "small-test-payload.gguf"
        # Geometry is tested separately above; this fixture exercises actual
        # streaming file hashing and publication without allocating 35 GiB.
        model.write_bytes(bytes(range(256)) * 1000)
        identity = {"layers": 40, "experts": 256, "embedding": 2048, "ffn": 512,
                    "experts_used": 8, "bundle_bytes": profile.BUNDLE_BYTES,
                    "model_bytes": model.stat().st_size, "model_sha256": profile.file_sha256(model),
                    "model_hash_s": 0.123}
        trace = self.trace("a.tsv", [(0, 2, "gate", 7, 42)])
        benchmark = self.root / "benchmark.json"
        benchmark.write_text('{"prompt": "independent source marker", "seed": 42}', encoding="utf-8")
        output = self.root / "profile.txt"
        with patch.object(profile, "inspect_model", return_value=identity) as inspect:
            result = profile.generate_profile(model, [trace], output, [benchmark], profile.BUNDLE_BYTES)
        inspect.assert_called_once_with(model.resolve())
        data = output.read_bytes()
        self.assertNotIn(b"\r", data)
        self.assertEqual(hashlib.sha256(data).hexdigest(), result["profile_sha256"])
        lines = data.decode().splitlines()
        self.assertEqual(lines[0], profile.MAGIC)
        self.assertEqual(lines[1], "model_sha256 " + hashlib.sha256(model.read_bytes()).hexdigest())
        self.assertEqual(lines[10], "scores 10240")
        rows = [tuple(map(int, line.split())) for line in lines[11:]]
        self.assertEqual(len(set((l, e) for l, e, _ in rows)), 10240)
        self.assertEqual(rows, sorted(rows, key=lambda r: (-r[2], r[0], r[1])))
        self.assertEqual(rows[0], (2, 7, 42))
        source = result["source"]
        canonical = json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        self.assertEqual(hashlib.sha256(canonical.encode()).hexdigest(), result["source_fingerprint"])
        self.assertEqual(source["benchmark_json"][0]["sha256"], profile.file_sha256(benchmark))
        self.assertIn("not independent validation", result["scope"])
        sidecar = json.loads(output.with_suffix(".txt.provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(sidecar, result)
        with self.assertRaises(FileExistsError):
            profile.generate_profile(model, [trace], output)

    def test_inspect_model_hashes_complete_file_once(self):
        model = self.root / "model.gguf"
        model.write_bytes(b"header" + b"payload-at-the-end")
        with patch.object(profile, "GGUFFile", return_value=object()), \
                patch.object(profile, "validate_q8_geometry", return_value={}), \
                patch.object(profile, "file_sha256", wraps=profile.file_sha256) as digest:
            result = profile.inspect_model(model)
        digest.assert_called_once_with(model)
        self.assertEqual(result["model_sha256"], hashlib.sha256(model.read_bytes()).hexdigest())
        self.assertEqual(result["model_bytes"], len(model.read_bytes()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
