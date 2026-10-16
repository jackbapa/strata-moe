"""Offline downloader regression checks: real files/SHA, mocked HTTPS ranges."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import re
import ssl
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a3b as downloader


class Response:
    def __init__(self, data, start, end, total, broken=False, bad_header=False):
        self.status = 206
        self.headers = {"Content-Range": f"bytes {start}-{end}/{total + int(bad_header)}"}
        self.data = io.BytesIO(data)
        self.broken, self.reads = broken, 0

    def read(self, count):
        self.reads += 1
        if self.broken and self.reads > 1:
            raise OSError("offline simulated interrupted range")
        return self.data.read(min(count, 5) if self.broken else count)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.data.close()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        # A tiny configurable block runs the production algorithm, including
        # real concurrent file writes and real SHA-256, without large fixtures.
        self.parent = (Path(__file__).resolve().parents[1] / "bench/results/a3b-download-test-tmp").resolve()
        self.parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="fixture-", dir=self.parent)
        self.root = Path(self.temp.name).resolve()
        self.assertTrue(self.root.is_relative_to(self.parent))
        self.stack = ExitStack()
        self.addCleanup(self.cleanup_fixture)
        self.stack.enter_context(patch.object(downloader, "ROOT", self.root))
        self.stack.enter_context(patch.object(downloader, "DOWNLOAD_BLOCK_BYTES", 64))
        self.payload = bytes((index * 17 + 3) % 256 for index in range(199))
        self.sha = hashlib.sha256(self.payload).hexdigest()
        self.quant, self.repo, self.revision, self.filename = "TEST", "offline/fixture", "fixed-revision", "fixture.gguf"
        self.stack.enter_context(patch.object(downloader, "QUANTS", {self.quant: (self.filename, len(self.payload), self.sha)}))
        self.stack.enter_context(patch.object(downloader, "QUANT_SOURCES", {self.quant: (self.repo, self.revision)}))
        self.stack.enter_context(patch.object(downloader.time, "sleep", lambda seconds: None))
        self.calls, self.lock = Counter(), threading.Lock()
        self.fail, self.corrupt, self.bad_header = set(), set(), set()
        self.stack.enter_context(patch.object(downloader.urllib.request, "urlopen", self.urlopen))
        self.output = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.output))
        self.folder = self.root / "models/a3b"
        self.folder.mkdir(parents=True)
        self.target = self.folder / self.filename
        self.partial = self.target.with_suffix(".gguf.partial")
        self.state = self.folder / "TEST-download-state.json"

    def cleanup_fixture(self):
        self.stack.close()
        # Verify the resolved absolute deletion target before TemporaryDirectory
        # recursively removes this test's own files on Windows.
        resolved = Path(self.temp.name).resolve()
        if not resolved.is_relative_to(self.parent) or resolved == self.parent:
            raise RuntimeError("refusing cleanup outside the test fixture directory")
        self.temp.cleanup()

    def urlopen(self, request, *, timeout, context):
        self.assertIsInstance(context, ssl.SSLContext)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertEqual(request.full_url,
                         f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{self.filename}?download=true")
        match = re.fullmatch(r"bytes=(\d+)-(\d+)", request.get_header("Range"))
        self.assertIsNotNone(match)
        start, end = map(int, match.groups())
        self.assertLessEqual(end, len(self.payload) - 1)
        self.assertEqual(start % 64, 0)
        index = start // 64
        with self.lock:
            self.calls[index] += 1
        data = self.payload[start:end + 1]
        if index in self.corrupt:
            data = bytes([data[0] ^ 1]) + data[1:]
        return Response(data, start, end, len(self.payload),
                        broken=index in self.fail, bad_header=index in self.bad_header)

    def download_fixture(self):
        downloader.download(quant=self.quant, workers=4)

    def write_state(self, done, *, legacy=False, **changes):
        state = {"revision": self.revision, "block": 64, "done": done}
        if not legacy:
            state.update(repo=self.repo, file=self.filename, bytes=len(self.payload), sha256=self.sha)
        state.update(changes)
        self.state.write_text(json.dumps(state), encoding="utf-8")

    def assert_download_complete(self):
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertFalse(self.partial.exists())
        self.assertFalse(self.state.exists())
        provenance = json.loads((self.folder / "TEST-provenance.json").read_text())
        self.assertEqual(provenance, {"repo": self.repo, "revision": self.revision,
                                     "file": self.filename, "bytes": len(self.payload), "sha256": self.sha})

    def test_preallocated_partial_without_state_redownloads_every_range(self):
        with self.partial.open("wb") as file:
            file.truncate(len(self.payload))
        self.download_fixture()
        self.assertEqual(self.calls, Counter({0: 1, 1: 1, 2: 1, 3: 1}))
        self.assert_download_complete()

    def test_interrupted_range_keeps_other_successes_and_resumes_only_failure(self):
        self.fail = {1}
        with self.assertRaisesRegex(RuntimeError, "ranges failed"):
            self.download_fixture()
        state = json.loads(self.state.read_text())
        self.assertEqual(set(state["done"]), {0, 2, 3})
        self.assertEqual(state["sha256"], self.sha)
        self.assertEqual(self.calls[1], 4)
        self.assertFalse(self.target.exists())
        self.fail.clear()
        self.calls.clear()
        self.download_fixture()
        self.assertEqual(self.calls, Counter({1: 1}))
        self.assert_download_complete()

    def test_wrong_model_identity_rejected_before_http(self):
        for field, value in (("repo", "other/repo"), ("file", "other.gguf"),
                             ("bytes", 200), ("sha256", "0" * 64), ("revision", "other-revision")):
            with self.subTest(field=field):
                self.partial.write_bytes(self.payload)
                self.write_state([0], **{field: value})
                with self.assertRaisesRegex(RuntimeError, "another model"):
                    self.download_fixture()
                self.assertFalse(self.calls)
                self.assertFalse(self.target.exists())

    def test_invalid_done_indices_rejected_before_http(self):
        for done in ([-1], [4], ["0"], [True], "0"):
            with self.subTest(done=done):
                self.partial.write_bytes(self.payload)
                self.write_state(done)
                with self.assertRaisesRegex(RuntimeError, "invalid download range state"):
                    self.download_fixture()
                self.assertFalse(self.calls)

    def test_wrong_partial_size_does_not_trust_recorded_done_ranges(self):
        self.partial.write_bytes(self.payload[:63])
        self.write_state([0, 1, 2, 3])
        self.download_fixture()
        self.assertEqual(self.calls, Counter({0: 1, 1: 1, 2: 1, 3: 1}))
        self.assert_download_complete()

    def test_legacy_state_resumes_and_migrates_identity(self):
        self.partial.write_bytes(self.payload[:64] + bytes(len(self.payload) - 64))
        self.write_state([0], legacy=True)
        self.fail = {1}
        with self.assertRaisesRegex(RuntimeError, "ranges failed"):
            self.download_fixture()
        self.assertNotIn(0, self.calls)
        state = json.loads(self.state.read_text())
        self.assertEqual((state["repo"], state["file"], state["bytes"], state["sha256"]),
                         (self.repo, self.filename, len(self.payload), self.sha))
        self.assertEqual(set(state["done"]), {0, 2, 3})
        self.fail.clear()
        self.calls.clear()
        self.download_fixture()
        self.assertEqual(self.calls, Counter({1: 1}))
        self.assert_download_complete()

    def test_corrupted_payload_rejected_and_next_run_recovers(self):
        self.corrupt = {2}
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            self.download_fixture()
        self.assertFalse(self.target.exists())
        self.assertTrue(self.partial.exists())
        self.assertFalse((self.folder / "TEST-provenance.json").exists())
        self.assertNotEqual(hashlib.sha256(self.partial.read_bytes()).hexdigest(), self.sha)
        self.assertEqual(json.loads(self.state.read_text())["done"], [])
        self.corrupt.clear()
        self.calls.clear()
        self.download_fixture()
        self.assertEqual(self.calls, Counter({0: 1, 1: 1, 2: 1, 3: 1}))
        self.assert_download_complete()

    def test_incorrect_content_range_is_not_accepted(self):
        self.bad_header = {1}
        with self.assertRaisesRegex(RuntimeError, "ranges failed"):
            self.download_fixture()
        self.assertEqual(set(json.loads(self.state.read_text())["done"]), {0, 2, 3})
        self.assertFalse(self.target.exists())

    def test_existing_wrong_target_is_rejected_and_preserved(self):
        self.target.write_bytes(b"user file")
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            self.download_fixture()
        self.assertEqual(self.target.read_bytes(), b"user file")
        self.assertFalse(self.calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)
