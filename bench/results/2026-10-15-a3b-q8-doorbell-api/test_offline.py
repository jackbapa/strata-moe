"""Doorbell counter corruption checks; no inference, HTTP or GPU access."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from audit import audit_sse
from audit_helpers import extract_counters, save_sse, validate_result


def result():
    return {"trial": "fixture", "case_id": "prompt32-output256", "received_done": True,
            "count_endpoint_matches_actual_usage": True, "reasoning_characters": 0,
            "usage": {"prompt_tokens": 32, "completion_tokens": 256, "total_tokens": 288},
            "timings": {"cache_n": 0, "prompt_n": 32, "predicted_n": 256, "prompt_ms": 500.,
                        "predicted_ms": 5000., "draft_n": 9, "draft_n_accepted": 5},
            "http_response_first_byte_s": .01, "ttft_any_content_or_reasoning_s": .5,
            "ttft_visible_content_s": .5, "engine_prompt_tokens_per_s": 64.,
            "engine_predicted_tokens_per_s": 51.2, "client_usage_decode_tokens_per_s": 51.}


def block(calls=20):
    return f"""A3B static: calls {calls}; mapped host 0.0 MiB; shared GPU scratch 204.0 MiB; prefill route reads 1 decode route reads 0
A3B: prompt 32 tokens 0.500s 64.00 tok/s; output 256 tokens 5.000s 51.20 tok/s
A3B cache: 16384.0 MiB; GPU rows 180 CPU rows 60; hit 180 miss 60; uploaded 64.0 MiB
A3B backend: route reads 50; native layout calls 0; single-column calls 1
A3B transfer: prefill 10.0 MiB; decode 0.0 MiB; decode GPU rows 18 CPU rows 6; decode hit 18 miss 6
A3B doorbell: calls {calls} CPU requests 10 callback 7.000 ms; mailbox to host 32768 bytes to GPU 16384 bytes
A3B hybrid: calls 0; PCIe fraction 0.0000; GPU miss rows 0 experts 0; input allocations 0 bytes 0 fallbacks 0
A3B CPU pool: calls 1 jobs 2 unique inputs 1 experts 2; input quant 1.125 ms gate/up 2.250 ms hidden quant 0.125 ms down 3.500 ms
A3B cache decay: prefill 0; decode 0
A3B MTP: windows 3 offered 9 accepted 5; draft 2.0 ms catchup 1.0 ms
A3B MTP adaptive: K0=1/2.0ms K1=1/3.0ms K2=1/4.0ms
"""


class CounterTests(unittest.TestCase):
    def test_static_line_stays_with_current_trial_and_old_process_is_excluded(self):
        log = "[strata] old engine started:\n" + block(99) + "[strata] new engine started:\n" + block(10) + block(20) + block(30)
        parsed = extract_counters(log.encode(), [result(), result()], 1)
        self.assertEqual([row["doorbell_calls_request"] for row in parsed["requests"]], [20, 30])
        self.assertEqual(parsed["requests"][0]["ordinary_hybrid_calls_request"], 0)

    def test_corrupt_execution_counters_are_rejected(self):
        replacements = [("decode 0.0 MiB", "decode 1.1 MiB"), ("decode route reads 0", "decode route reads 1"),
                        ("CPU rows 6;", "CPU rows 0;"), ("decode GPU rows 18", "decode GPU rows 0"),
                        ("calls 1 jobs 2", "calls 1 jobs 3"), ("offered 9 accepted 5", "offered 9 accepted 4"),
                        ("A3B doorbell:", "missing doorbell:"), ("PCIe fraction 0.0000", "PCIe fraction 0.2500")]
        for old, new in replacements:
            with self.subTest(old=old), self.assertRaises(RuntimeError):
                extract_counters(block().replace(old, new).encode(), [result()], 0)

    def test_done_exported_counts_and_times_are_checked(self):
        for key, value in (("predicted_n", 255), ("prompt_n", 31), ("cache_n", 1), ("predicted_ms", 7000.)):
            row = copy.deepcopy(result()); row["timings"][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                extract_counters(block().encode(), [row], 0)

    def test_literal_done_is_additionally_checked_when_logged(self):
        correct = block().replace("A3B: prompt", "DONE 256 32 500.0 5000.0 length 5 9 0\nA3B: prompt")
        self.assertTrue(extract_counters(correct.encode(), [result()], 0)["requests"][0]["raw_engine_DONE_logged"])
        with self.assertRaises(RuntimeError):
            extract_counters(correct.replace("DONE 256", "DONE 255").encode(), [result()], 0)

    def test_nonfinite_speed_is_never_accepted(self):
        row = result(); row["client_usage_decode_tokens_per_s"] = float("inf")
        with self.assertRaises(RuntimeError): validate_result(row)

    def test_saved_sse_reproduces_ttft_and_usage_rate_for_bursts(self):
        row = result()
        records = [{"at_s": .02, "sse": ": model progress"},
                   {"at_s": .04, "sse": "data: " + json.dumps({"choices": [{"delta": {"role": "assistant"}}]})},
                   {"at_s": .5, "sse": "data: " + json.dumps({"choices": [{"delta": {"content": "a"}}]})},
                   {"at_s": 5.5, "sse": "data: " + json.dumps({"choices": [{"delta": {"content": "b"}}]})},
                   {"at_s": 5.6, "sse": "data: " + json.dumps({"usage": row["usage"], "timings": row["timings"]})},
                   {"at_s": 5.7, "sse": "data: [DONE]"}]
        with tempfile.TemporaryDirectory(prefix="doorbell-audit-") as temporary:
            folder = Path(temporary)
            events = folder / "fixture.events.jsonl"
            events.write_text("".join(json.dumps(event) + "\n" for event in records), encoding="utf-8")
            (folder / "fixture.output.json").write_text(json.dumps({"content": "ab", "reasoning": ""}), encoding="utf-8")
            save_sse(folder, "fixture")
            self.assertEqual(len(audit_sse(folder, row)), 64)
            records[3]["at_s"] = 6.5
            events.write_text("".join(json.dumps(event) + "\n" for event in records), encoding="utf-8")
            save_sse(folder, "fixture")
            with self.assertRaisesRegex(RuntimeError, "timestamps"):
                audit_sse(folder, row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
