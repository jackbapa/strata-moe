"""Measure resident IQ3_S A3B arms without starting, stopping, or reloading a server.

Run one arm at a time after selecting its server config. Reuse the suite's exact
request across arms. ``compare`` reports observed medians, not a speed claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import threading
import time

import psutil

from bench_a3b_latency import Client, prepare_prompt, stream_request


NATIVE_LINE = re.compile(
    r"A3B Strata grouped CUDA expert calls: (\d+)"
    r"(?:; request prefill (\d+) decode (\d+))?"
)
TRANSFER_LINE = re.compile(
    r"A3B transfer: prefill ([\d.]+) MiB; decode ([\d.]+) MiB; "
    r"decode GPU rows (\d+) CPU rows (\d+); decode hit (\d+) miss (\d+)"
)
ENGINE_LINE = re.compile(
    r"A3B: prompt (\d+) tokens ([\d.]+)s ([\d.]+) tok/s; "
    r"output (\d+) tokens ([\d.]+)s ([\d.]+) tok/s"
)
MTP_DEVICE_PREFILL_LINE = re.compile(
    r"A3B MTP device prefill: calls (\d+) rows (\d+)"
    r"(?:; avoided hidden D2H (\d+) bytes)?"
)
MTP_HIDDEN_LINE = re.compile(r"A3B MTP sampling: GPU samples (\d+); token/guard download (\d+) bytes; hidden ([\d.]+) ms")
MTP_DEVICE_CHAIN_K2_LINE = re.compile(
    r"A3B MTP device K2 chain: calls (\d+) fallback (\d+); "
    r"token/hidden D2D (\d+) bytes; inter-step context syncs (\d+); "
    r"(?:CUDA cross-step graph replays 0|CUDA graph replays not instrumented)"
)
MTP_PROPOSAL_TRACE_LINE = re.compile(
    r"^A3B MTP proposal trace: window (\d+) pos (-?\d+) input (-?\d+) offered (\d+)((?: -?\d+)*)$",
    re.MULTILINE,
)
MTP_SUMMARY_LINE = re.compile(r"A3B MTP: windows (\d+) offered (\d+) accepted (\d+)")
GDN_GRAPH_LINE = re.compile(
    r"A3B Strata fused GDN graph matches: lifetime (\d+); "
    r"request prefill builds (\d+) decode builds (\d+)"
)
STRATA_PREFILL_MMQ_LINE = re.compile(
    r"A3B Strata original prefill MMQ: CUDA calls (\d+) rows (\d+); "
    r"request prefill calls (\d+) rows (\d+); decode calls (\d+); "
    r"graph builds (\d+) fallback graph builds (\d+)"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def argument(args: list[str], name: str) -> str:
    index = args.index(name)
    return args[index + 1]


def configured_path(config: dict, config_path: Path, value: str) -> Path:
    base = Path(config.get("cwd") or config_path.parent)
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def load_config(config_path: Path) -> tuple[dict, Path, Path, Path, Path | None]:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    args = config["args"]
    model = configured_path(config, config_path, argument(args, "--model"))
    exe = configured_path(config, config_path, config["exe"])
    log = configured_path(config, config_path, config["log"])
    mtp = configured_path(config, config_path, argument(args, "--mtp-model")) if "--mtp-model" in args else None
    if "IQ3_S" not in model.name.upper():
        raise ValueError(f"A/B suite requires the IQ3_S target GGUF: {model}")
    if int(argument(args, "--ctx")) < 262144 or argument(args, "--device") != "0":
        raise ValueError("A/B suite requires context >= 262144 and GPU 0")
    return config, model, exe, log, mtp


def find_engine(exe: Path, model: Path, pid: int | None) -> psutil.Process:
    if pid is not None:
        candidates = [psutil.Process(pid)]
    else:
        candidates = []
        for process in psutil.process_iter(["pid", "name"]):
            if (process.info["name"] or "").lower() == exe.name.lower():
                candidates.append(process)
    matches = []
    for process in candidates:
        try:
            command = process.cmdline()
            if os.path.normcase(str(Path(process.exe()).resolve())) != os.path.normcase(str(exe)):
                continue
            if "--serve" not in command or "--model" not in command:
                continue
            actual_model = Path(command[command.index("--model") + 1]).resolve()
            if os.path.normcase(str(actual_model)) == os.path.normcase(str(model)):
                matches.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, IndexError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f"expected one resident engine for {model}, found {len(matches)}; pass --engine-pid")
    return matches[0]


def verify_engine(process: psutil.Process, config: dict, exe: Path) -> dict:
    command = process.cmdline()
    expected = ["--serve", *config["args"]]
    if command[1:] != expected:
        raise RuntimeError("resident engine command line differs from the selected config; restart it first")
    try:
        environment = process.environ()
    except (psutil.AccessDenied, NotImplementedError) as error:
        raise RuntimeError("cannot verify resident engine environment") from error
    for key, value in config.get("env", {}).items():
        if environment.get(key) != str(value):
            raise RuntimeError(f"resident engine environment differs from config at {key}")
    relevant = {key: value for key, value in environment.items()
                if key.startswith("STRATA_A3B_") or key.startswith("GGML_CUDA_") or key == "CUDA_VISIBLE_DEVICES"}
    unexpected = {key: value for key, value in relevant.items()
                  if key.startswith(("STRATA_A3B_", "GGML_CUDA_")) and key not in config.get("env", {})}
    if unexpected:
        raise RuntimeError(f"resident engine has experiment switches absent from config: {unexpected}")
    return {"engine_pid": process.pid, "engine_created_at": process.create_time(),
            "engine_exe": str(exe), "engine_command": command, "engine_environment": relevant}


def parse_engine_log(segment: str) -> dict:
    result: dict = {}
    trace = list(MTP_PROPOSAL_TRACE_LINE.finditer(segment))
    if trace:
        # Hash the complete normalized per-window token record. The engine
        # emits the same trace in both arms only when explicitly requested.
        lines = []
        for expected_window, item in enumerate(trace, 1):
            tokens = item[5].split()
            if int(item[1]) != expected_window or len(tokens) != int(item[4]):
                raise ValueError("incomplete or malformed MTP proposal trace")
            lines.append(item[0])
        result["mtp_proposal_trace_windows"] = len(trace)
        result["mtp_proposal_trace_sha256"] = hashlib.sha256(
            ("\n".join(lines) + "\n").encode("ascii")).hexdigest()
    if match := list(ENGINE_LINE.finditer(segment)):
        item = match[-1]
        result["engine_log_prompt_tokens"] = int(item[1])
        result["engine_log_prefill_tok_s"] = float(item[3])
        result["engine_log_output_tokens"] = int(item[4])
        result["engine_log_output_tok_s"] = float(item[6])
    if match := list(NATIVE_LINE.finditer(segment)):
        item = match[-1]
        result["strata_native_calls_cumulative"] = int(item[1])
        result["strata_native_calls_request_prefill"] = int(item[2]) if item[2] is not None else None
        result["strata_native_calls_request_decode"] = int(item[3]) if item[3] is not None else None
        result["strata_native_calls_request"] = (int(item[2]) + int(item[3])) if item[2] is not None else None
        result["strata_native_calls_status"] = "per_request" if item[2] is not None else "cumulative_only"
    else:
        result["strata_native_calls_request"] = None
        result["strata_native_calls_status"] = "not_reported"
    if match := list(TRANSFER_LINE.finditer(segment)):
        item = match[-1]
        result.update(prefill_upload_mib=float(item[1]), decode_upload_mib=float(item[2]),
                      decode_gpu_rows=int(item[3]), decode_cpu_rows=int(item[4]),
                      decode_cache_hits=int(item[5]), decode_cache_misses=int(item[6]))
    if match := list(MTP_DEVICE_PREFILL_LINE.finditer(segment)):
        item = match[-1]
        result.update(mtp_device_prefill_calls=int(item[1]), mtp_device_prefill_rows=int(item[2]),
                      mtp_device_prefill_avoided_d2h_bytes=int(item[3]) if item[3] is not None else None)
    if match := list(MTP_HIDDEN_LINE.finditer(segment)):
        item = match[-1]
        result.update(mtp_gpu_draft_samples=int(item[1]),
                      mtp_draft_sampling_download_bytes=int(item[2]),
                      mtp_hidden_ms=float(item[3]))
    if match := list(MTP_DEVICE_CHAIN_K2_LINE.finditer(segment)):
        item = match[-1]
        result.update(mtp_device_chain_k2_calls=int(item[1]),
                      mtp_device_chain_k2_fallbacks=int(item[2]),
                      mtp_device_chain_k2_d2d_bytes=int(item[3]),
                      mtp_device_chain_k2_inter_step_context_syncs=int(item[4]),
                      # The old engine printed a literal zero, not a CUDA
                      # counter. Preserve it as unknown in old/new logs.
                      mtp_device_chain_k2_cuda_cross_step_replays=None)
    if match := list(STRATA_PREFILL_MMQ_LINE.finditer(segment)):
        item = match[-1]
        result.update(strata_prefill_mmq_cuda_calls=int(item[1]),
                      strata_prefill_mmq_cuda_rows=int(item[2]),
                      strata_prefill_mmq_request_calls=int(item[3]),
                      strata_prefill_mmq_request_rows=int(item[4]),
                      strata_prefill_mmq_decode_calls=int(item[5]),
                      strata_prefill_mmq_graph_builds=int(item[6]),
                      strata_prefill_mmq_fallback_graph_builds=int(item[7]))
    if match := list(GDN_GRAPH_LINE.finditer(segment)):
        item = match[-1]
        result.update(gdn_graph_matches_lifetime=int(item[1]),
                      gdn_graph_matches_request_prefill=int(item[2]),
                      gdn_graph_matches_request_decode=int(item[3]))
    return result


def gpu_zero_used_mib() -> int:
    output = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.used",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout
    return int(output.strip())


class Resources:
    def __init__(self, process: psutil.Process, interval: float):
        self.process = process
        self.interval = interval
        self.stop = threading.Event()
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self) -> dict:
        memory = self.process.memory_info()
        system = psutil.virtual_memory()
        return {"wall_time": time.time(), "monotonic_time": time.perf_counter(),
                "gpu0_total_used_mib": gpu_zero_used_mib(),
                "engine_rss_bytes": memory.rss,
                "engine_private_bytes": getattr(memory, "private", None),
                "system_used_bytes": system.used,
                "system_available_bytes": system.available}

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                self.samples.append(self.sample())
            except (OSError, ValueError, subprocess.SubprocessError, psutil.Error) as error:
                self.errors.append(str(error))
                break
            self.stop.wait(self.interval)

    def start(self) -> None:
        self.samples.append(self.sample())
        self.thread.start()

    def finish(self, out: Path) -> dict:
        self.stop.set()
        self.thread.join(timeout=max(2.0, self.interval * 2))
        self.samples.append(self.sample())
        if self.errors or self.thread.is_alive():
            raise RuntimeError(f"resource sampler failed: {self.errors}")
        out.write_text("".join(json.dumps(sample) + "\n" for sample in self.samples), encoding="utf-8")
        first, last = self.samples[0], self.samples[-1]
        return {"sample_interval_s": self.interval, "resource_samples": len(self.samples),
                **{f"{key}_{suffix}": value for key in
                   ("gpu0_total_used_mib", "engine_rss_bytes", "engine_private_bytes", "system_used_bytes")
                   for suffix, value in (("before", first.get(key)), ("after", last.get(key)),
                                         ("observed_peak", max((s[key] for s in self.samples if s.get(key) is not None),
                                                               default=None)))},
                "gpu0_total_used_mib_observed_peak_over_before":
                    max(s["gpu0_total_used_mib"] for s in self.samples) - first["gpu0_total_used_mib"]}


class NsysOutputCapture:
    """Opt-in Nsight control for one trace; never used by ordinary A/B runs.

    The profiler must already have launched the server with --session-new and
    --start-later=true. A first SSE text delta arrives after the engine began
    output, so this captures the remaining output, not exactly 255 tokens.
    """

    def __init__(self, executable: Path, session: str, record_path: Path):
        self.executable = executable
        self.session = session
        self.record_path = record_path
        self.records: list[dict] = []
        self.start_attempted = False
        self.started = False
        self.first_delta_at: float | None = None

    def command(self, action: str, *extra: str) -> None:
        argv = [str(self.executable), action, f"--session={self.session}", *extra]
        started = time.perf_counter()
        try:
            result = subprocess.run(argv, capture_output=True, text=True, timeout=120,
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired) as error:
            self.records.append({"action": action, "command": argv,
                                 "elapsed_s": time.perf_counter() - started, "error": str(error)})
            raise RuntimeError(f"nsys {action} failed: {error}") from error
        self.records.append({"action": action, "command": argv,
                             "elapsed_s": time.perf_counter() - started,
                             "returncode": result.returncode,
                             "stdout": result.stdout, "stderr": result.stderr})
        if result.returncode:
            raise RuntimeError(f"nsys {action} exited {result.returncode}: {result.stderr or result.stdout}")

    def start_on_first_model_delta(self, observed_at: float) -> None:
        if self.start_attempted:
            raise RuntimeError("Nsight start was triggered more than once")
        self.start_attempted = True
        self.first_delta_at = observed_at
        self.command("start")
        self.started = True

    def finish(self) -> None:
        errors = []
        try:
            if self.started:
                try:
                    self.command("stop")
                except RuntimeError as error:
                    errors.append(str(error))
            try:
                self.command("shutdown", "--kill=false")
            except RuntimeError as error:
                errors.append(str(error))
        finally:
            write_json(self.record_path, {"session": self.session,
                                          "first_model_delta_monotonic_s": self.first_delta_at,
                                          "nsys_start_attempted": self.start_attempted,
                                          "nsys_started": self.started,
                                          "commands": self.records,
                                          "capture_note": "SSE trigger is after model output began; profiling may miss early output tokens and perturbs stream timing."})
        if errors:
            raise RuntimeError("; ".join(errors))


def proposal_trace_complete(segment: str) -> bool:
    summaries = list(MTP_SUMMARY_LINE.finditer(segment))
    traces = list(MTP_PROPOSAL_TRACE_LINE.finditer(segment))
    return (len(summaries) == 1 and len(traces) == int(summaries[0][1]) and
            all(int(match[1]) == index for index, match in enumerate(traces, 1)))


def appended_log(path: Path, start: int, expect_native: bool, expect_trace: bool = False,
                 timeout: float = 5.0) -> str:
    deadline = time.monotonic() + timeout
    text = ""
    while time.monotonic() < deadline:
        with path.open("rb") as source:
            if source.seek(0, os.SEEK_END) < start:
                raise RuntimeError("engine log was truncated during measurement")
            source.seek(start)
            text = source.read().decode("utf-8", "replace")
        if (ENGINE_LINE.search(text) and TRANSFER_LINE.search(text)
                and (not expect_native or NATIVE_LINE.search(text))
                and (not expect_trace or proposal_trace_complete(text))):
            return text
        time.sleep(0.1)
    return text


def median(items: list[dict], key: str) -> float | None:
    values = [item[key] for item in items if item.get(key) is not None]
    return statistics.median(values) if values else None


def run_arm(args: argparse.Namespace) -> None:
    config_path = args.config.resolve()
    config, model_path, exe, log, mtp_path = load_config(config_path)
    process = find_engine(exe, model_path, args.engine_pid)
    engine_identity = verify_engine(process, config, exe)
    client = Client(args.url or f"http://{config['host']}:{config['port']}", args.timeout)
    if client.json("/v1/models")["data"][0]["id"] != config["model_name"]:
        raise RuntimeError("API model ID differs from the selected config")
    suite = args.suite.resolve()
    suite.mkdir(parents=True, exist_ok=True)
    template_path = suite / "request-template.json"
    if template_path.exists():
        template = json.loads(template_path.read_text(encoding="utf-8"))
        for key, expected in (("target_prompt_tokens", args.prompt_tokens), ("max_tokens", args.max_tokens),
                              ("temperature", args.temperature), ("seed", args.seed), ("thinking", args.thinking)):
            if template[key] != expected:
                raise RuntimeError(f"suite request differs at {key}; choose a new suite directory")
    else:
        case = prepare_prompt(client, config["model_name"], args.prompt_tokens, args.thinking)
        if case["count_endpoint_prompt_tokens"] != args.prompt_tokens:
            raise RuntimeError("token-count endpoint did not produce the requested exact prompt length")
        template = {"target_prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens,
                    "temperature": args.temperature, "seed": args.seed, "thinking": args.thinking,
                    "messages": case["messages"]}
        write_json(template_path, template)
    counted = client.json("/v1/messages/count_tokens", {"model": config["model_name"],
                          "messages": template["messages"],
                          "thinking": {"type": "enabled" if args.thinking else "disabled"}})["input_tokens"]
    if counted != args.prompt_tokens:
        raise RuntimeError(f"actual config tokenizer counted {counted} prompt tokens, expected {args.prompt_tokens}")
    model_sha = sha256_file(model_path)
    mtp_sha = sha256_file(mtp_path) if mtp_path else None
    source = {"model_path": str(model_path), "model_bytes": model_path.stat().st_size,
              "model_sha256": model_sha, "mtp_path": str(mtp_path) if mtp_path else None,
              "mtp_sha256": mtp_sha, "prompt_sha256": sha256_json(template),
              "target_prompt_tokens": args.prompt_tokens, "max_tokens": args.max_tokens}
    identity_path = suite / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != source:
            raise RuntimeError("different GGUF/MTP/request in this A/B suite; choose a new suite directory")
    else:
        write_json(identity_path, source)
    arm = suite / "arms" / args.arm
    arm.mkdir(parents=True, exist_ok=True)
    metadata = {**source, "config_path": str(config_path), "config_sha256": sha256_file(config_path),
                "binary_sha256": sha256_file(exe), "binary_path": str(exe),
                "engine_identity": engine_identity, "url": client.url}
    metadata_path = arm / "metadata.json"
    if metadata_path.exists():
        prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        for key in ("config_sha256", "binary_sha256", "model_sha256", "mtp_sha256"):
            if prior[key] != metadata[key]:
                raise RuntimeError(f"arm {args.arm} changed {key}; use a new arm name")
    else:
        write_json(metadata_path, metadata)
    if args.prepare_only:
        print(json.dumps({"suite": str(suite), "arm": args.arm, "prompt_tokens": counted,
                          "model_sha256": model_sha, "config_sha256": metadata["config_sha256"]}))
        return
    body = {"model": config["model_name"], "messages": template["messages"],
            "max_tokens": args.max_tokens, "temperature": args.temperature, "seed": args.seed,
            "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": args.thinking}}
    existing = sorted(arm.glob("run[0-9][0-9][0-9].json"))
    start_number = int(existing[-1].stem[3:]) + 1 if existing else 1
    for number in range(start_number, start_number + args.repeats):
        trial = f"run{number:03d}"
        log_start = log.stat().st_size
        resources = Resources(process, args.sample_interval)
        profile = (NsysOutputCapture(args.nsys_exe, args.nsys_session,
                                     arm / f"{trial}.nsys-control.json")
                   if args.nsys_session else None)
        resources.start()
        try:
            result = stream_request(client, body, arm, trial,
                                    on_first_model_delta=(profile.start_on_first_model_delta if profile else None))
        finally:
            try:
                if profile is not None:
                    profile.finish()
            finally:
                resource_metrics = resources.finish(arm / f"{trial}.resources.jsonl")
        # The engine can flush its high-volume diagnostic trace after its
        # performance counters.  A live read may therefore leave the trace
        # unavailable; an isolated service runner can audit it after stop.
        segment = appended_log(log, log_start, "--strata-native-experts" in config["args"])
        if len(ENGINE_LINE.findall(segment)) != 1 or len(TRANSFER_LINE.findall(segment)) != 1:
            raise RuntimeError(f"{trial} engine log is incomplete or contains another request")
        log_metrics = parse_engine_log(segment)
        usage, timing = result["usage"], result.get("timings") or {}
        if usage.get("prompt_tokens") != args.prompt_tokens or usage.get("completion_tokens") != args.max_tokens:
            raise RuntimeError(f"{trial} generated different token counts; inspect its events and output")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", timing.get("cache_n"))
        if cached != 0:
            raise RuntimeError(f"{trial} has {cached} cached prompt tokens; A/B requires an uncached 50K input")
        if (log_metrics["engine_log_prompt_tokens"] != args.prompt_tokens
                or log_metrics["engine_log_output_tokens"] != args.max_tokens):
            raise RuntimeError(f"{trial} engine log token counts differ from the API response")
        if "--strata-native-experts" in config["args"] and log_metrics["strata_native_calls_status"] == "not_reported":
            raise RuntimeError(f"{trial} enabled Strata native experts but no call count was reported")
        output = json.loads((arm / f"{trial}.output.json").read_text(encoding="utf-8"))
        result.update(resource_metrics)
        result.update(log_metrics)
        result.update({"trial": trial, "arm": args.arm, "config_sha256": metadata["config_sha256"],
                       "binary_sha256": metadata["binary_sha256"], "model_sha256": model_sha,
                       "mtp_sha256": mtp_sha, "prompt_sha256": source["prompt_sha256"],
                       "engine_pid": process.pid, "engine_created_at": process.create_time(),
                       "cached_prompt_tokens": cached,
                       "output_sha256": sha256_json(output),
                       "gpu_memory_scope": "whole GPU 0; other processes may contribute",
                       "host_memory_scope": "engine RSS/private plus system-wide used bytes; observed sampling peak"})
        if profile is not None:
            result["nsys_output_capture_session"] = profile.session
        offered, accepted = timing.get("draft_n"), timing.get("draft_n_accepted")
        result.update(mtp_offered_tokens=offered, mtp_accepted_tokens=accepted,
                      mtp_acceptance_rate=(accepted / offered if offered else None))
        write_json(arm / f"{trial}.json", result)
        print(json.dumps({"arm": args.arm, "trial": trial,
                          "first_byte_s": result["http_response_first_byte_s"],
                          "first_model_delta_s": result["ttft_any_content_or_reasoning_s"],
                          "prefill_tok_s": result["engine_prompt_tokens_per_s"],
                          "output_tok_s": result["engine_predicted_tokens_per_s"],
                          "gpu0_peak_mib": result["gpu0_total_used_mib_observed_peak"],
                          "strata_calls": result["strata_native_calls_request"],
                          "strata_calls_status": result["strata_native_calls_status"]},
                         ensure_ascii=False), flush=True)


def compare(args: argparse.Namespace) -> None:
    suite = args.suite.resolve()
    arms = {}
    for name in (args.baseline, args.variant):
        folder = suite / "arms" / name
        items = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(folder.glob("run[0-9][0-9][0-9].json"))]
        if not items:
            raise RuntimeError(f"arm {name} has no complete trial records")
        if any(item.get("nsys_output_capture_session") for item in items):
            raise RuntimeError("Nsight-profiled trials are diagnostic and cannot enter an unprofiled A/B comparison")
        arms[name] = items
    fields = ("binary_sha256", "model_sha256", "mtp_sha256", "prompt_sha256", "cached_prompt_tokens")
    if any(item[key] != arms[args.baseline][0][key] for trials in arms.values() for item in trials for key in fields):
        raise RuntimeError("A/B arms do not share the same executable/target/MTP/request or uncached prompt")
    metrics = ("http_response_first_byte_s", "ttft_any_content_or_reasoning_s",
               "engine_prompt_tokens_per_s", "engine_predicted_tokens_per_s",
               "gpu0_total_used_mib_observed_peak", "gpu0_total_used_mib_before",
               "gpu0_total_used_mib_after", "gpu0_total_used_mib_observed_peak_over_before",
               "engine_rss_bytes_before", "engine_rss_bytes_after", "engine_rss_bytes_observed_peak",
               "engine_private_bytes_observed_peak", "system_used_bytes_observed_peak",
               "strata_native_calls_request", "strata_native_calls_request_prefill",
               "mtp_device_prefill_calls", "mtp_device_prefill_rows", "mtp_hidden_ms",
               "mtp_device_prefill_avoided_d2h_bytes", "gdn_graph_matches_request_prefill",
               "mtp_device_chain_k2_calls", "mtp_device_chain_k2_fallbacks",
               "mtp_device_chain_k2_d2d_bytes", "mtp_device_chain_k2_inter_step_context_syncs",
               "mtp_device_chain_k2_cuda_cross_step_replays",
               "mtp_proposal_trace_windows",
               "gdn_graph_matches_request_decode",
               "strata_prefill_mmq_request_calls", "strata_prefill_mmq_request_rows",
               "strata_native_calls_request_decode", "mtp_acceptance_rate")
    summary = {name: {"runs": len(items), "output_sha256": sorted(set(item["output_sha256"] for item in items)),
                      "mtp_proposal_trace_sha256": sorted(set(item.get("mtp_proposal_trace_sha256") for item in items)),
                      **{key + "_median": median(items, key) for key in metrics}}
               for name, items in arms.items()}
    baseline_trace = summary[args.baseline]["mtp_proposal_trace_sha256"]
    variant_trace = summary[args.variant]["mtp_proposal_trace_sha256"]
    # A missing trace is unknown, not evidence that the proposal streams differ.
    # Some older runs recorded acceptance and output but did not enable the
    # opt-in per-window trace. Keep false for two fully observed, unequal arms.
    summary["mtp_proposal_trace_identical"] = (
        None if None in baseline_trace or None in variant_trace else
        baseline_trace == variant_trace and len(baseline_trace) == 1)
    summary["comparison"] = {key + "_variant_minus_baseline":
                             (summary[args.variant][key + "_median"] - summary[args.baseline][key + "_median"])
                             if summary[args.variant][key + "_median"] is not None
                             and summary[args.baseline][key + "_median"] is not None else None
                             for key in metrics}
    summary["note"] = ("GPU memory is total observed GPU 0 usage, not process-attributed VRAM. "
                       "Request-native calls are unavailable for old cumulative-only logs. "
                       "One trial per arm is exploratory; alternate arms and repeat before claiming an improvement.")
    write_json(suite / "comparison.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Measure the current resident service; never reloads it")
    run.add_argument("--suite", required=True, type=Path)
    run.add_argument("--arm", required=True, help="A stable arm name, e.g. ggml or strata")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--url", help="Override the local config URL")
    run.add_argument("--engine-pid", type=int, help="Specify resident engine PID if discovery is ambiguous")
    run.add_argument("--prompt-tokens", type=int, default=50000)
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--temperature", type=float, default=0)
    run.add_argument("--seed", type=int, default=12345)
    run.add_argument("--thinking", action="store_true")
    run.add_argument("--sample-interval", type=float, default=0.5)
    run.add_argument("--timeout", type=float, default=600)
    run.add_argument("--prepare-only", action="store_true", help="Prepare exact request/hashes without inference")
    run.add_argument("--nsys-session", help="Diagnostic only: start an existing --start-later Nsight session at first model text, then stop it after [DONE]")
    run.add_argument("--nsys-exe", type=Path,
                     default=Path(r"C:\Program Files\NVIDIA Corporation\Nsight Systems 2024.4.2\target-windows-x64\nsys.exe"))
    report = commands.add_parser("compare", help="Compare stored arm medians")
    report.add_argument("--suite", required=True, type=Path)
    report.add_argument("--baseline", required=True)
    report.add_argument("--variant", required=True)
    args = parser.parse_args()
    if args.command == "run":
        if (not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", args.arm)
                or min(args.prompt_tokens, args.max_tokens, args.repeats) < 1
                or args.sample_interval <= 0):
            parser.error("invalid arm, token count, repeat count, or sample interval")
        if args.nsys_session and (not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.nsys_session)
                                  or args.repeats != 1 or args.prepare_only or not args.nsys_exe.is_file()):
            parser.error("--nsys-session needs one inference run, a simple session name, and an existing --nsys-exe")
        run_arm(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
