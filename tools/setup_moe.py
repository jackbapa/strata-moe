"""Install and start the independent Strata-Moe A3B engine.

Only the selected IQ3_S target and, in full-GPU or hybrid mode, its matching MTP file
are fetched. Model weights and build products stay in ignored local folders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import a3b  # pinned model identities and SHA-256 checks live in one place

MODEL = ROOT / "models/a3b/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"
MTP = ROOT / "models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0.gguf"
CONFIG = ROOT / "strata-moe.json"
HYBRID_MTP_MIN_VRAM_MIB = 8192


def auto_mode(vram_mib: int) -> str:
    if vram_mib >= 20480:
        return "full"
    if vram_mib >= HYBRID_MTP_MIN_VRAM_MIB:
        return "hybrid"
    return "hybrid-no-mtp" if vram_mib >= 4096 else "cpu"


def needs_mtp(mode: str) -> bool:
    return mode in {"full", "hybrid"}


def gpu_info(device: int) -> tuple[int, str]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return 0, ""
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=compute_cap,memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=10,
        )
        cap, mib = [part.strip() for part in out.splitlines()[device].split(",")[:2]]
        return int(mib), cap.replace(".", "")
    except (IndexError, ValueError, OSError, subprocess.SubprocessError):
        return 0, ""


def run(*args: str, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, env=env, check=True)


def install_requirements() -> None:
    req = ROOT / "requirements.txt"
    digest = hashlib.sha256(req.read_bytes()).hexdigest()
    marker = ROOT / ".venv/.strata-moe-requirements.sha256"
    if marker.exists() and marker.read_text(encoding="ascii").strip() == digest:
        print("Python dependencies already installed", flush=True)
        return
    run(sys.executable, "-m", "pip", "install", "-r", str(req))
    marker.write_text(digest + "\n", encoding="ascii")


def verified_model(path: Path, provenance: Path, *, repo: str, revision: str,
                   size: int, sha256: str) -> bool:
    """Trust only a downloader-produced receipt newer than the local weight file."""
    if not path.is_file() or not provenance.is_file() or path.stat().st_size != size:
        return False
    if path.stat().st_mtime > provenance.stat().st_mtime:
        return False
    try:
        record = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return all(record.get(key) == value for key, value in {
        "repo": repo, "revision": revision, "file": path.name, "bytes": size, "sha256": sha256,
    }.items())


def export_tokenizer() -> Path:
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile

    meta = GGUFFile(MODEL).metadata
    folder = ROOT / "models/a3b/Qwen3.6-35B-A3B-UD-IQ3_S-tokenizer"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "vocab.json").write_text(
        json.dumps({word: i for i, word in enumerate(meta["tokenizer.ggml.tokens"])}, ensure_ascii=False),
        encoding="utf-8",
    )
    (folder / "merges.txt").write_text("\n".join(meta["tokenizer.ggml.merges"]), encoding="utf-8")
    (folder / "token_type.json").write_text(json.dumps(meta["tokenizer.ggml.token_type"]), encoding="utf-8")
    (folder / "chat_template.jinja").write_text(meta["tokenizer.chat_template"], encoding="utf-8")
    return folder


def config_for(mode: str, device: int, port: int, tokenizer: Path) -> dict:
    cpu = mode == "cpu"
    exe_dir = "build-a3b-cpu" if cpu else "build-a3b"
    exe = ROOT / exe_dir / "bin" / ("strata-moe.exe" if os.name == "nt" else "strata-moe")
    args = ["--model", str(MODEL)]
    env: dict[str, str] = {}
    if mode == "full":
        args += ["--ctx", "262144", "--batch", "2048", "--device", str(device), "--vram-mib", "0",
                 "--mtp-model", str(MTP), "--mtp-draft", "2", "--mtp-ubatch", "512", "--exact-verify",
                 "--native-gpu", "--gpu-embedding", "--gpu-sampling", "--defer-mtp-catchup",
                 "--fused-moe", "--native-dense-exact", "--kv-tiered", "--kv-tiered-f16-short"]
        if device == 0:
            args += ["--strata-native-experts"]
            env = {
                "STRATA_A3B_Q6_GROUPED_EXACT": "1", "STRATA_A3B_MTP_KV_ONLY_DIAGNOSTIC": "0",
                "STRATA_A3B_MTP_PROPOSAL_TRACE": "0", "STRATA_A3B_FUSED_GU": "1",
                "STRATA_A3B_DOWN_MODE": "1", "STRATA_A3B_Q6_REUSE_EXACT": "1", "STRATA_A3B_MTP_KV_ONLY": "1",
            }
    elif mode == "hybrid":
        args += ["--ctx", "262144", "--batch", "2048", "--device", str(device),
                 "--vram-mib", "0", "--cache-reserve-mib", "640",
                 "--threads", str(min(16, max(1, (os.cpu_count() or 2) // 2))),
                 "--kv-tiered", "--kv-tiered-f16-short", "--hybrid-experts", "--cpu-expert-pool",
                 "--pcie-frac", "0.25", "--expert-staging-slots", "8", "--gpu-embedding",
                 "--gpu-sampling", "--fused-moe", "--mtp-model", str(MTP), "--mtp-draft", "2",
                 "--mtp-ubatch", "512", "--exact-verify", "--defer-mtp-catchup",
                 "--native-attention-exact"]
    elif mode == "hybrid-no-mtp":
        args += ["--ctx", "32768", "--batch", "512", "--device", str(device),
                 "--vram-mib", "0", "--cache-reserve-mib", "640"]
    elif mode == "6g":
        args += ["--ctx", "262144", "--batch", "128", "--device", str(device),
                 "--vram-mib", "6144", "--cache-reserve-mib", "640", "--cache-mib", "2048",
                 "--kv-tiered", "--kv-tier-low-vram"]
    elif mode == "2g":
        args += ["--ctx", "262144", "--batch", "32", "--device", str(device),
                 "--vram-mib", "2048", "--cache-reserve-mib", "128", "--cache-mib", "0",
                 "--kv-tiered", "--kv-tier-low-vram"]
    else:
        args += ["--ctx", "262144", "--batch", "128", "--cpu", "--kv-tiered",
                 "--threads", str(min(16, max(1, (os.cpu_count() or 2) // 2)))]
    return {
        "mode": mode, "model_name": "qwen3.6-35b-a3b-ud-iq3_s", "exe": str(exe), "cwd": str(ROOT),
        "tokenizer": str(tokenizer), "host": "127.0.0.1", "port": port,
        "log": str(ROOT / "strata-moe-engine.log"), "args": args,
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
        "open_browser": True, "idle_unload_s": 0, "env": env,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["auto", "full", "hybrid", "hybrid-no-mtp", "6g", "cpu"], default="auto")
    p.add_argument("--device", type=int, default=None, help="NVIDIA GPU index; default 0")
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--workers", type=int, default=8, help="model download workers")
    p.add_argument("--no-start", action="store_true", help="install only; start later with START-HERE.bat")
    p.add_argument("--rebuild", action="store_true")
    p.add_argument("--reconfigure", action="store_true", help="replace the generated strata-moe.json")
    p.add_argument("--dry-run", action="store_true", help="show selected mode and downloads without changing files")
    a = p.parse_args()
    existing = json.loads(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() and not a.reconfigure else None
    if existing:
        saved_args = existing.get("args", [])
        saved_mode = existing.get("mode") or (
            "cpu" if "--cpu" in saved_args else "full" if "--native-gpu" in saved_args else
            "6g" if "6144" in saved_args and "--vram-mib" in saved_args else
            "2g" if "2048" in saved_args and "--vram-mib" in saved_args else "hybrid"
        )
        if saved_mode == "hybrid" and "--mtp-model" not in saved_args:
            saved_mode = "hybrid-no-mtp"
            print("Existing legacy hybrid config has no MTP; use --reconfigure to enable it.", flush=True)
        if a.mode != "auto" and a.mode != saved_mode:
            p.error(f"existing config uses {saved_mode}; add --reconfigure to switch modes")
        a.mode = saved_mode
        saved_device = int(saved_args[saved_args.index("--device") + 1]) if "--device" in saved_args else 0
        if a.device is not None and a.device != saved_device:
            p.error(f"existing config uses GPU {saved_device}; add --reconfigure to change it")
        a.device = saved_device
        if a.port is None:
            a.port = int(existing.get("port", 8096))
    if a.device is None:
        a.device = 0
    if a.port is None:
        a.port = 8096
    if a.device < 0 or not 1 <= a.port <= 65535 or not 1 <= a.workers <= 64:
        p.error("invalid device, port or worker count")
    vram, arch = gpu_info(a.device)
    mode = a.mode
    if mode == "auto":
        # Keep the bounded 6 GiB profile explicit until measured on a physical
        # card of that size. Below the hybrid+MTP floor, retain the old path.
        mode = auto_mode(vram)
    if mode != "cpu" and not vram:
        p.error("NVIDIA GPU not found; use --mode cpu or install a working NVIDIA driver")
    if mode == "full" and vram < 20480:
        p.error("full mode needs at least 20 GiB of detected VRAM; use --mode hybrid")
    if mode == "hybrid" and vram < HYBRID_MTP_MIN_VRAM_MIB:
        p.error("hybrid+MTP needs at least 8 GiB of detected VRAM; use --mode hybrid-no-mtp")
    if mode == "6g" and vram < 6144:
        p.error("6g mode needs at least 6 GiB of detected VRAM; use --mode cpu")
    print(f"Strata-Moe mode={mode}, GPU={a.device}, detected VRAM={vram} MiB, CUDA arch={arch or 'none'}", flush=True)
    if mode == "full" and a.device != 0:
        print("Strata grouped IQ expert kernel currently supports GPU 0 only; this device uses the GGML expert path.", flush=True)
    print("Target model: Unsloth Qwen3.6-35B-A3B UD-IQ3_S (13.68 GB)", flush=True)
    if needs_mtp(mode):
        print("MTP model: ggml-org matching Q8_0 (1.99 GB)", flush=True)
    if a.dry_run:
        print("Dry run: no dependencies, weights, build, config or server changed")
        return 0
    install_requirements()
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    sys.path.insert(0, str(ROOT))
    import setup
    if mode != "cpu":
        nvcc, vcvars = setup.install_build_tools({"arch": int(arch)}, yes=False)
        env["CUDACXX"] = nvcc
        if vcvars:
            env["STRATA_VCVARS"] = str(vcvars)
    elif os.name == "nt":
        vcvars = setup.find_vcvars(None)
        if not vcvars:
            winget = shutil.which("winget")
            if not winget:
                raise RuntimeError("Visual Studio 2022 C++ Build Tools are missing; install them and rerun SETUP.bat")
            print("Installing Visual Studio 2022 C++ Build Tools for the CPU engine...", flush=True)
            subprocess.run([winget, "install", "-e", "--source", "winget", "--accept-package-agreements",
                            "--accept-source-agreements", "--disable-interactivity", "--id",
                            "Microsoft.VisualStudio.2022.BuildTools", "--override",
                            "--quiet --wait --norestart --nocache --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"],
                           cwd=ROOT, check=False)
            vcvars = setup.find_vcvars(None)
        if not vcvars:
            raise RuntimeError("Visual Studio 2022 C++ Build Tools could not be found after installation")
        env["STRATA_VCVARS"] = str(vcvars)
    elif not shutil.which("g++"):
        raise RuntimeError("A C++ compiler is required for the CPU engine (on Ubuntu/Debian: install build-essential)")
    exe_dir = "build-a3b-cpu" if mode == "cpu" else "build-a3b"
    exe = ROOT / exe_dir / "bin" / ("strata-moe.exe" if os.name == "nt" else "strata-moe")
    if a.rebuild or not exe.exists():
        run(sys.executable, str(ROOT / "tools/a3b.py"), "build", "--backend", "cpu" if mode == "cpu" else "cuda",
            "--arch", arch or "75", "--jobs", str(max(2, min(8, (os.cpu_count() or 4) // 2))), env=env)
    if not verified_model(MODEL, MODEL.parent / "IQ3_S-provenance.json",
                          repo=a3b.MODEL_REPO, revision=a3b.MODEL_REVISION,
                          size=a3b.QUANTS["IQ3_S"][1], sha256=a3b.QUANTS["IQ3_S"][2]):
        run(sys.executable, str(ROOT / "tools/a3b.py"), "download", "--quant", "IQ3_S", "--workers", str(a.workers), env=env)
    if needs_mtp(mode) and not verified_model(MTP, MTP.parent / "mtp-provenance.json",
                                              repo=a3b.MTP_REPO, revision=a3b.MTP_REVISION,
                                              size=a3b.MTP_QUANTS["Q8_0"][1], sha256=a3b.MTP_QUANTS["Q8_0"][2]):
        run(sys.executable, str(ROOT / "tools/a3b.py"), "download", "--mtp", "--workers", str(a.workers), env=env)
    tokenizer = export_tokenizer()
    if not CONFIG.exists() or a.reconfigure:
        CONFIG.write_text(json.dumps(config_for(mode, a.device, a.port, tokenizer), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {CONFIG}", flush=True)
    else:
        print(f"Keeping existing {CONFIG}; use --reconfigure to replace it", flush=True)
    print(f"Web UI: http://127.0.0.1:{a.port}/  API: http://127.0.0.1:{a.port}/v1", flush=True)
    if not a.no_start:
        print("Starting local server; press Ctrl+C to stop. Next time use START-HERE.bat.", flush=True)
        run(sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(CONFIG),
            "--port", str(a.port), "--open", env=env)
    else:
        print("Installation complete. Run START-HERE.bat to start the web app and API.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"Strata-Moe setup failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
