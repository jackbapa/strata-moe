"""Offline checks for the opt-in first-model-delta Nsight trigger."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_a3b_ab import NsysOutputCapture
from bench_a3b_latency import stream_request


class _Response:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def read1(self, _size):
        return next(self.chunks, b"")

    def close(self):
        pass


class _Client:
    def __init__(self, chunks):
        self.chunks = chunks

    def request(self, _path, _body, timed=False):
        assert timed
        now = time.perf_counter()
        observation = {"request_started_at": now,
                       "first_response_byte_at": now, "headers_received_at": now}
        return _Response([]), _Response(self.chunks), observation


class NsysTriggerTests(unittest.TestCase):
    def test_callback_fires_once_for_first_nonempty_model_delta(self):
        frames = [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n',
            b'data: {"choices":[],"usage":{"completion_tokens":2},"timings":{}}\n\n',
            b'data: [DONE]\n\n',
        ]
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            result = stream_request(_Client(frames), {"stream": True}, Path(directory), "trial",
                                    on_first_model_delta=calls.append)
            self.assertEqual(len(calls), 1)
            self.assertEqual(result["nonempty_text_events"], 2)
            self.assertTrue(result["received_done"])
            self.assertEqual(json.loads((Path(directory) / "trial.output.json").read_text())["content"], "ab")

    def test_nsys_session_start_stop_shutdown_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control.json"
            capture = NsysOutputCapture(Path("nsys.exe"), "A3BOutputNode", path)
            with patch("bench_a3b_ab.subprocess.run",
                       return_value=subprocess.CompletedProcess([], 0, "ok", "")) as run:
                capture.start_on_first_model_delta(time.perf_counter())
                capture.finish()
            self.assertEqual([call.args[0][1] for call in run.call_args_list],
                             ["start", "stop", "shutdown"])
            self.assertIn("--kill=false", run.call_args_list[-1].args[0])
            saved = json.loads(path.read_text())
            self.assertEqual(len(saved["commands"]), 3)
            self.assertTrue(saved["nsys_started"])


if __name__ == "__main__":
    unittest.main()
