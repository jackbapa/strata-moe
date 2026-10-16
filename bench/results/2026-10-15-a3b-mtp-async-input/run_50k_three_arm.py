"""Run balanced, isolated 50K/256 service trials and restore selected 8096."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
from bench_a3b_ab import find_engine, load_config, verify_engine  # noqa: E402

HERE = Path(__file__).resolve().parent
SUITE = HERE / "50k-256"
ORIGINAL = ROOT / "strata-a3b-iq3s-q6-kv-tier-f16-auto-native.json"
RUNNER = ROOT / "tools/bench_a3b_ab.py"
MODEL_ID = "qwen3.6-35b-a3b-ud-iq3_s"
URL = "http://127.0.0.1:8096/v1/models"
ARMS = ("selected", "k2-sync", "k2-async")
ORDER = ("selected", "k2-sync", "k2-async", "k2-async", "k2-sync", "selected")
EXPERIMENT_KEYS = tuple(k for k in os.environ if
                        k.startswith(("STRATA_A3B_", "GGML_CUDA_")) or
                        k == "CUDA_VISIBLE_DEVICES")
CLEAN_ENV = {k: v for k, v in os.environ.items() if k not in EXPERIMENT_KEYS}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def output_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def port_open():
    try:
        with socket.create_connection(("127.0.0.1", 8096), timeout=1):
            return True
    except OSError:
        return False


def loaded_model():
    with urlopen(URL, timeout=3) as response:
        result = json.load(response)
    for item in result["data"]:
        if item["id"] == MODEL_ID and item.get("status", {}).get("value") == "loaded":
            return item
    return None


def wait_loaded(server, timeout=180):
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited early with {server.returncode}")
        try:
            if model := loaded_model():
                return model
        except (OSError, ValueError, KeyError) as error:
            last_error = str(error)
        time.sleep(1)
    raise TimeoutError(f"model did not load in {timeout}s; last error: {last_error}")


def start(config_path, label):
    if port_open():
        raise RuntimeError("port 8096 is already occupied; refusing to start another service")
    log_path = SUITE / "servers" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "serve.server", "--engine", "strata",
               "--config", str(config_path), "--port", "8096"]
    print(json.dumps({"starting": label, "command": command}), flush=True)
    with log_path.open("wb") as log:
        server = subprocess.Popen(command, cwd=ROOT, env=CLEAN_ENV,
                                  stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        model = wait_loaded(server)
        config, model_path, exe, _, _ = load_config(config_path)
        identity = verify_engine(find_engine(exe, model_path, None), config, exe)
    except BaseException:
        stop(server)
        raise
    print(json.dumps({"ready": label, "server_pid": server.pid,
                      "engine_pid": identity["engine_pid"],
                      "loaded": model.get("status", {}).get("value"),
                      "n_ctx": model.get("meta", {}).get("n_ctx")}), flush=True)
    return server, model


def stop(server):
    if server is None:
        return
    try:
        parent = psutil.Process(server.pid)
        children = parent.children(recursive=True)
    except psutil.NoSuchProcess:
        children = []
        parent = None
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    if parent is not None:
        try:
            parent.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs([*children, *([parent] if parent else [])], timeout=8)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    server.wait(timeout=10)
    deadline = time.monotonic() + 10
    while port_open() and time.monotonic() < deadline:
        time.sleep(0.25)
    if port_open():
        raise RuntimeError("port 8096 remained occupied after stopping test server")


def make_configs(original):
    configs = {}
    (SUITE / "engines").mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        config = json.loads(json.dumps(original))
        config["log"] = str(SUITE / "engines" / f"{arm}.log")
        config["env"].update({
            "STRATA_A3B_MTP_DEVICE_CHAIN_K2": "1" if arm != "selected" else "0",
            "STRATA_A3B_MTP_ASYNC_INPUT": "1" if arm == "k2-async" else "0",
            "STRATA_A3B_MTP_ASYNC_INPUT_FORCE_STEP2_FALLBACK": "0",
            "STRATA_A3B_MTP_PROPOSAL_TRACE": "1",
            "STRATA_A3B_NVTX": "0",
        })
        config_path = SUITE / "configs" / f"{arm}.json"
        output_json(config_path, config)
        configs[arm] = config_path
    return configs


def trial(arm, config_path, index):
    command = [sys.executable, str(RUNNER), "run", "--suite", str(SUITE),
               "--arm", arm, "--config", str(config_path),
               "--prompt-tokens", "50000", "--max-tokens", "256",
               "--repeats", "1", "--timeout", "900"]
    path = SUITE / "runner-logs" / f"{index:02d}-{arm}.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"trial_started": index, "arm": arm}), flush=True)
    begin = time.monotonic()
    with path.open("wb") as log:
        completed = subprocess.run(command, cwd=ROOT, env=CLEAN_ENV,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   timeout=1800,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    text = path.read_text(encoding="utf-8", errors="replace")
    print(json.dumps({"trial_finished": index, "arm": arm,
                      "elapsed_s": round(time.monotonic() - begin, 2),
                      "returncode": completed.returncode,
                      "runner_output": text[-2000:]}, ensure_ascii=False), flush=True)
    if completed.returncode:
        raise RuntimeError(f"trial {index} {arm} failed; inspect {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if port_open():
        raise RuntimeError("8096 is occupied; selected service must be stopped first")
    original_hash = sha(ORIGINAL)
    original = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    binary = (ROOT / original["exe"]).resolve() if not Path(original["exe"]).is_absolute() else Path(original["exe"])
    binary_hash = sha(binary)
    configs = make_configs(original)
    output_json(SUITE / "plan.json", {"original_config_sha256": original_hash,
                "binary_sha256": binary_hash, "order": ORDER[:3] if args.rounds == 1 else ORDER,
                "inherited_experiment_keys_removed": EXPERIMENT_KEYS,
                "configs": {a: {"path": str(p), "sha256": sha(p)} for a, p in configs.items()}})
    active = None
    try:
        for index, arm in enumerate(ORDER[:3] if args.rounds == 1 else ORDER, 1):
            active, _ = start(configs[arm], f"{index:02d}-{arm}")
            try:
                trial(arm, configs[arm], index)
            finally:
                stop(active)
                active = None
    finally:
        cleanup_error = None
        if active is not None:
            try:
                stop(active)
            except Exception as error:
                cleanup_error = str(error)
        try:
            if sha(ORIGINAL) != original_hash:
                raise RuntimeError("original selected config changed during test; refusing automatic restore")
            restored, model = start(ORIGINAL, "restored-selected")
            output_json(SUITE / "restored.json", {"server_pid": restored.pid,
                        "config_path": str(ORIGINAL), "config_sha256": sha(ORIGINAL),
                        "binary_sha256": sha(binary), "model": model,
                        "inherited_experiment_keys_removed": EXPERIMENT_KEYS})
            print(json.dumps({"restored_8096_pid": restored.pid,
                              "loaded": model.get("status", {}).get("value"),
                              "n_ctx": model.get("meta", {}).get("n_ctx"),
                              "cleanup_error": cleanup_error}), flush=True)
        except Exception as restore_error:
            output_json(SUITE / "restore-error.json", {"cleanup_error": cleanup_error,
                        "restore_error": str(restore_error)})
            raise
        if cleanup_error:
            raise RuntimeError(f"test-server cleanup failed, selected service restored: {cleanup_error}")


if __name__ == "__main__":
    main()
