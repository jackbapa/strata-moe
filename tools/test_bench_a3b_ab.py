"""Offline checks for A/B measurement integrity and log interpretation."""
from __future__ import annotations

import argparse
import hashlib
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_a3b_ab import compare, load_config, parse_engine_log, proposal_trace_complete, sha256_json, verify_engine


class BenchmarkAbTests(unittest.TestCase):
    def test_new_native_log_uses_request_deltas(self):
        log = ("A3B: prompt 50000 tokens 28.000s 1785.71 tok/s; output 256 tokens 4.000s 64.00 tok/s\n"
               "A3B Strata grouped CUDA expert calls: 10000; request prefill 8000 decode 2000\n"
               "A3B transfer: prefill 0.0 MiB; decode 0.0 MiB; decode GPU rows 2048 CPU rows 0; decode hit 0 miss 0\n")
        result = parse_engine_log(log)
        self.assertEqual(result["strata_native_calls_request"], 10000)
        self.assertEqual(result["strata_native_calls_request_prefill"], 8000)
        self.assertEqual(result["strata_native_calls_request_decode"], 2000)
        self.assertEqual(result["strata_native_calls_status"], "per_request")
        self.assertEqual(result["engine_log_prompt_tokens"], 50000)
        self.assertEqual(result["decode_cpu_rows"], 0)

    def test_old_cumulative_log_does_not_claim_request_calls(self):
        result = parse_engine_log("A3B Strata grouped CUDA expert calls: 5070\n")
        self.assertEqual(result["strata_native_calls_cumulative"], 5070)
        self.assertIsNone(result["strata_native_calls_request"])
        self.assertEqual(result["strata_native_calls_status"], "cumulative_only")
        missing = parse_engine_log("A3B: prompt 1 tokens 1.000s 1.00 tok/s; output 1 tokens 1.000s 1.00 tok/s\n")
        self.assertIsNone(missing["strata_native_calls_request"])
        self.assertEqual(missing["strata_native_calls_status"], "not_reported")

    def test_mtp_device_prefill_counts_are_not_confused_with_round_graphs(self):
        log = ("A3B MTP sampling: GPU samples 17; token/guard download 136 bytes; hidden 8.2 ms\n"
               "A3B MTP device prefill: calls 25 rows 50000; avoided hidden D2H 409395200 bytes\n"
               "A3B Strata fused GDN graph matches: lifetime 61; request prefill builds 30 decode builds 31 (not CUDA graph replays)\n")
        result = parse_engine_log(log)
        self.assertEqual(result["mtp_device_prefill_calls"], 25)
        self.assertEqual(result["mtp_device_prefill_rows"], 50000)
        self.assertEqual(result["mtp_device_prefill_avoided_d2h_bytes"], 409395200)
        self.assertEqual(result["mtp_hidden_ms"], 8.2)
        self.assertEqual(result["gdn_graph_matches_request_decode"], 31)
        self.assertNotIn("mtp_fused_rounds", result)
        self.assertNotIn("mtp_device_chain_k2_calls", result)

    def test_mtp_device_chain_counts_require_the_exact_runtime_line(self):
        log = ("A3B MTP device prefill: calls 25 rows 50000; avoided hidden D2H 409395200 bytes\n"
               "A3B MTP device K2 chain: calls 118 fallback 0; token/hidden D2D 966672 bytes; "
               "inter-step context syncs 0; CUDA cross-step graph replays 0\n")
        result = parse_engine_log(log)
        self.assertEqual(result["mtp_device_chain_k2_calls"], 118)
        self.assertEqual(result["mtp_device_chain_k2_fallbacks"], 0)
        self.assertEqual(result["mtp_device_chain_k2_d2d_bytes"], 966672)
        self.assertEqual(result["mtp_device_chain_k2_inter_step_context_syncs"], 0)
        self.assertIsNone(result["mtp_device_chain_k2_cuda_cross_step_replays"])
        self.assertEqual(result["mtp_device_prefill_calls"], 25)
        self.assertNotIn("mtp_device_chain_k2_calls", parse_engine_log("MTP device K2 chain: calls 118"))
        current = log.replace("CUDA cross-step graph replays 0", "CUDA graph replays not instrumented")
        self.assertIsNone(parse_engine_log(current)["mtp_device_chain_k2_cuda_cross_step_replays"])

    def test_opt_in_proposal_trace_hashes_complete_windows_only(self):
        lines = ("A3B MTP proposal trace: window 1 pos 50000 input 42 offered 2 17 19\n"
                 "A3B MTP proposal trace: window 2 pos 50003 input 21 offered 1 23\n")
        result = parse_engine_log(lines)
        self.assertEqual(result["mtp_proposal_trace_windows"], 2)
        self.assertEqual(result["mtp_proposal_trace_sha256"], hashlib.sha256(lines.encode("ascii")).hexdigest())
        self.assertNotEqual(result["mtp_proposal_trace_sha256"],
                            parse_engine_log(lines.replace(" 23\n", " 24\n"))["mtp_proposal_trace_sha256"])
        self.assertNotIn("mtp_proposal_trace_sha256", parse_engine_log("A3B MTP device K2 chain: calls 2\n"))
        with self.assertRaisesRegex(ValueError, "incomplete or malformed"):
            parse_engine_log(lines.replace("offered 2 17 19", "offered 2 17"))

    def test_trace_capture_waits_for_all_reported_windows(self):
        first = "A3B MTP proposal trace: window 1 pos 50000 input 42 offered 2 17 19\n"
        second = "A3B MTP proposal trace: window 2 pos 50003 input 21 offered 1 23\n"
        summary = "A3B MTP: windows 2 offered 3 accepted 1; draft 1.0 ms catchup 2.0 ms\n"
        self.assertFalse(proposal_trace_complete(first + summary))
        self.assertTrue(proposal_trace_complete(first + summary + second))
        self.assertFalse(proposal_trace_complete(first + second + summary.replace("windows 2", "windows 3")))

    def test_config_requires_iq3s_gpu0_and_256k(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "arm.json"
            data = {"args": ["--model", "Qwen3.6-IQ3_S.gguf", "--ctx", "262144", "--device", "0"],
                    "exe": "strata-moe.exe", "log": "engine.log"}
            path.write_text(json.dumps(data), encoding="utf-8")
            config, model, *_ = load_config(path)
            self.assertEqual(model.name, "Qwen3.6-IQ3_S.gguf")
            data["args"][3] = "65536"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "context"):
                load_config(path)

    def test_compare_rejects_different_model_and_reports_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            for name, target in (("baseline", "same"), ("strata", "same")):
                arm = suite / "arms" / name
                arm.mkdir(parents=True)
                data = {"binary_sha256": "same-binary", "model_sha256": target, "mtp_sha256": "mtp", "prompt_sha256": "prompt",
                        "cached_prompt_tokens": 0, "output_sha256": "answer",
                        "http_response_first_byte_s": 0.01,
                        "ttft_any_content_or_reasoning_s": 29.0 if name == "baseline" else 28.0,
                        "engine_predicted_tokens_per_s": 60.0 if name == "baseline" else 65.0,
                        "strata_native_calls_request": None if name == "baseline" else 2500}
                (arm / "run001.json").write_text(json.dumps(data), encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                compare(argparse.Namespace(suite=suite, baseline="baseline", variant="strata"))
            result = json.loads((suite / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(result["comparison"]["engine_predicted_tokens_per_s_variant_minus_baseline"], 5.0)
            self.assertIsNone(result["comparison"]["strata_native_calls_request_variant_minus_baseline"])
            self.assertIsNone(result["mtp_proposal_trace_identical"])
            record = suite / "arms" / "strata" / "run001.json"
            edited = json.loads(record.read_text(encoding="utf-8"))
            edited["model_sha256"] = "different"
            record.write_text(json.dumps(edited), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "do not share"):
                with redirect_stdout(io.StringIO()):
                    compare(argparse.Namespace(suite=suite, baseline="baseline", variant="strata"))
            edited["model_sha256"] = "same"
            edited["binary_sha256"] = "different-binary"
            record.write_text(json.dumps(edited), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "same executable"):
                with redirect_stdout(io.StringIO()):
                    compare(argparse.Namespace(suite=suite, baseline="baseline", variant="strata"))

    def test_request_hash_ignores_json_key_order(self):
        self.assertEqual(sha256_json({"a": 1, "b": 2}), sha256_json({"b": 2, "a": 1}))

    def test_engine_rejects_undeclared_cuda_fusion_switch(self):
        class Process:
            pid = 123

            def cmdline(self):
                return ["strata-moe.exe", "--serve", "--device", "0"]

            def environ(self):
                return {"GGML_CUDA_DISABLE_FUSION": "1"}

            def create_time(self):
                return 0.0

        with self.assertRaisesRegex(RuntimeError, "experiment switches absent"):
            verify_engine(Process(), {"args": ["--device", "0"]}, Path("strata-moe.exe"))


if __name__ == "__main__":
    unittest.main()
