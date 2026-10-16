"""Build and run the opt-in Qwen3.5/3.6 A3B engine. Does not invoke setup.py."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import ssl
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "unsloth/Qwen3.6-35B-A3B-GGUF"
MODEL_REVISION = "a483e9e6cbd595906af30beda3187c2663a1118c"
MODEL_FILE = "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"
MODEL_BYTES = 26456194016
MODEL_SHA256 = "c13ce26253ea334df472bd8fbd2d6da66d8a41195c17f6fcbf44c4d20ece0932"
MTP_REPO = "ggml-org/Qwen3.6-35B-A3B-GGUF"
MTP_REVISION = "baec3ebee244827cda0f4557eafa8b28f7545fa6"
MTP_FILE = "mtp-Qwen3.6-35B-A3B-Q8_0.gguf"
MTP_BYTES = 1990649632
MTP_SHA256 = "f6cb995fab78e7284097b2e13508e8daad4ffc6a747b433aef399f4e48524b5a"
MTP_QUANTS = {
    "Q8_0": (MTP_FILE, MTP_BYTES, MTP_SHA256),
    "Q4_0": ("mtp-Qwen3.6-35B-A3B-Q4_0.gguf", 1060038432,
             "606fca331adcbfbdadc107512ce6a7161e84e1646ba0e0018256426f6296877f"),
}
QUANTS = {
    "Q5_K_M": (MODEL_FILE, MODEL_BYTES, MODEL_SHA256),
    "Q4_K_S": ("Qwen3.6-35B-A3B-UD-Q4_K_S.gguf", 20893015008,
               "a8138f183e3993f12cdc23afd2babb8cdb084e64088ce4a256d49101d47b949c"),
    "Q4_K_M": ("Qwen3.6-35B-A3B-UD-Q4_K_M.gguf", 22134528992,
               "ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61"),
    "Q8_0": ("Qwen3.6-35B-A3B-Q8_0.gguf", 36903139328,
             "d8d7842cc657d720f39546878e431937c84473aa130c36371669ac17c80c7361"),
    "IQ3_S": ("Qwen3.6-35B-A3B-UD-IQ3_S.gguf", 13676723168,
              "66a3ca888ce13482b40c333db2432c0ebde3a7b13754fc29f0c6f5e89703ec66"),
}
QUANT_SOURCES = {"Q8_0": (MTP_REPO, MTP_REVISION)}
DOWNLOAD_BLOCK_BYTES = 64 * 1024 * 1024

def download(mtp=False, quant="Q5_K_M", workers=32, mtp_quant="Q8_0"):
    repo, revision, filename, expected_bytes, expected_sha = (
        (MTP_REPO, MTP_REVISION, *MTP_QUANTS[mtp_quant]) if mtp else
        (*QUANT_SOURCES.get(quant, (MODEL_REPO, MODEL_REVISION)), *QUANTS[quant]))
    folder = ROOT / "models" / "a3b"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / filename
    partial = target.with_suffix(".gguf.partial")
    if not target.exists():
        try:
            import certifi
            tls_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            tls_context = ssl.create_default_context()
        state_path = folder / (("mtp-download-state.json" if mtp_quant == "Q8_0" else
                                f"mtp-{mtp_quant}-download-state.json") if mtp else f"{quant}-download-state.json")
        block = DOWNLOAD_BLOCK_BYTES
        count = (expected_bytes + block - 1) // block
        if state_path.exists():
            state = json.loads(state_path.read_text())
            identity = {"repo": repo, "file": filename, "bytes": expected_bytes, "sha256": expected_sha}
            if (state.get("revision") != revision or state.get("block") != block or
                any(key in state and state[key] != value for key, value in identity.items())):
                raise RuntimeError("download state belongs to another model")
            completed = state["done"]
            if not isinstance(completed, list) or any(type(i) is not int or not 0 <= i < count for i in completed):
                raise RuntimeError("invalid download range state")
            done = set(completed) if partial.exists() and partial.stat().st_size == expected_bytes else set()
        else:
            # The partial file is preallocated. Its length cannot prove which
            # ranges arrived if the companion state was lost.
            done = set()
        def save():
            tmp = state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"repo": repo, "revision": revision, "file": filename,
                "bytes": expected_bytes, "sha256": expected_sha, "block": block, "done": sorted(done)}))
            tmp.replace(state_path)
        save()
        with partial.open("r+b" if partial.exists() else "wb") as out:
            out.truncate(expected_bytes)
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        if not endpoint.startswith("https://"):
            raise ValueError("HF_ENDPOINT must use HTTPS")
        url = f"{endpoint}/{repo}/resolve/{revision}/{filename}?download=true"
        def fetch(index):
            start, end = index * block, min(expected_bytes, (index + 1) * block) - 1
            for attempt in range(4):
                try:
                    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                    with urllib.request.urlopen(req, timeout=60, context=tls_context) as response, partial.open("r+b") as out:
                        if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{expected_bytes}":
                            raise RuntimeError("server returned an incorrect range")
                        out.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = response.read(min(4 * 1024 * 1024, remaining))
                            if not chunk: raise RuntimeError("short download range")
                            out.write(chunk); remaining -= len(chunk)
                    return index
                except Exception as error:
                    reason = str(getattr(error, "reason", ""))[:100]
                    print(f"range {index}: attempt {attempt + 1}/4 {type(error).__name__} {reason}", flush=True)
                    if attempt == 3: raise
                    time.sleep(2 * (attempt + 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, i) for i in range(count) if i not in done]
            failed = []
            for future in as_completed(futures):
                try:
                    done.add(future.result())
                except Exception as error:
                    # Keep recording completed disjoint ranges even when one
                    # connection fails. A retry must resume every success.
                    failed.append(type(error).__name__)
                    continue
                save()
                if len(done) % 4 == 0 or len(done) == count:
                    print(f"downloaded {min(len(done) * block, expected_bytes) / 2**30:.2f}/{expected_bytes / 2**30:.2f} GiB", flush=True)
            if failed:
                raise RuntimeError(f"{len(failed)} ranges failed; rerun the same command to resume verified ranges")
        check = partial
    else:
        check = target
    print("checking SHA-256", flush=True)
    h = hashlib.sha256()
    with check.open("rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    if check.stat().st_size != expected_bytes or h.hexdigest() != expected_sha:
        if check == partial:
            # A completed range list cannot identify the corrupt block from
            # the whole-file hash. Keep the temporary bytes, but trust none
            # of them on the next run; every range will be overwritten.
            done.clear()
            save()
            raise RuntimeError("model size or SHA-256 mismatch; partial state reset, rerun the same command to redownload all ranges")
        raise RuntimeError("model size or SHA-256 mismatch; existing target preserved")
    if check == partial:
        partial.replace(target)
        state_path.unlink(missing_ok=True)
    (folder / (("mtp-provenance.json" if mtp_quant == "Q8_0" else f"mtp-{mtp_quant}-provenance.json") if mtp else ("provenance.json" if quant == "Q5_K_M" else f"{quant}-provenance.json"))).write_text(json.dumps({"repo": repo, "revision": revision,
        "file": filename, "bytes": expected_bytes, "sha256": expected_sha}, indent=2) + "\n")
    print(target)

def apply_engine_patches():
    # Patches belong to the independent A3B build. Keep the pinned download
    # reproducible; refuse unexpected source changes instead of forcing hunks.
    patches = sorted((ROOT / "a3b/patches").glob("*.patch"))
    vendor = ROOT / "third_party/llama.cpp"
    stamp = vendor / ".strata-a3b-patch-state.json"
    patch_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in patches}
    changed_files = set()
    for patch in patches:
        for line in patch.read_text(encoding="utf-8").splitlines():
            if line.startswith("+++ b/"):
                changed_files.add(line[6:])

    def file_hashes():
        return {name: hashlib.sha256((vendor / name).read_bytes()).hexdigest()
                for name in sorted(changed_files)}

    def write_stamp():
        stamp.write_text(json.dumps({"patches": patch_hashes, "files": file_hashes()},
                                    indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if stamp.exists():
        state = json.loads(stamp.read_text(encoding="utf-8"))
        if state.get("patches") == patch_hashes and state.get("files") == file_hashes():
            return
        # A new opt-in patch may be appended to an otherwise verified local
        # build.  Apply only that suffix after checking every previously
        # patched source byte and patch hash.  Reordered, edited or removed
        # patches still fail closed instead of silently repatching the vendor.
        prior_patches = state.get("patches")
        prior_files = state.get("files")
        prior_names = sorted(prior_patches) if isinstance(prior_patches, dict) else []
        names = [patch.name for patch in patches]
        if (isinstance(prior_files, dict) and prior_names and
                names[:len(prior_names)] == prior_names and len(names) > len(prior_names) and
                all(patch_hashes.get(name) == prior_patches[name] for name in prior_names) and
                all((vendor / name).is_file() and
                    hashlib.sha256((vendor / name).read_bytes()).hexdigest() == digest
                    for name, digest in prior_files.items())):
            for patch in patches[len(prior_names):]:
                base = ["git", "apply", "--directory=third_party/llama.cpp"]
                check = subprocess.run([*base, "--check", str(patch)], cwd=ROOT,
                                       capture_output=True, text=True)
                if check.returncode:
                    raise RuntimeError(f"A3B patch does not match pinned llama.cpp: {patch.name}\n{check.stderr}")
                subprocess.run([*base, str(patch)], cwd=ROOT, check=True)
            write_stamp()
            return
        raise RuntimeError("pinned llama.cpp or A3B patches changed after application; inspect before rebuilding")

    # Existing A3B installations predate the patch-state file. The newest
    # patch can only reverse cleanly on the complete current stack; record it
    # once, then verify every patched file on future builds.
    if patches:
        final = ["git", "apply", "--directory=third_party/llama.cpp", "--reverse", "--check", str(patches[-1])]
        if subprocess.run(final, cwd=ROOT, capture_output=True).returncode == 0:
            write_stamp()
            return

    for patch in patches:
        base = ["git", "apply", "--directory=third_party/llama.cpp"]
        reverse = subprocess.run([*base, "--reverse", "--check", str(patch)], cwd=ROOT,
                                 capture_output=True, text=True)
        if reverse.returncode == 0:
            continue
        check = subprocess.run([*base, "--check", str(patch)], cwd=ROOT,
                               capture_output=True, text=True)
        if check.returncode:
            raise RuntimeError(f"A3B patch does not match pinned llama.cpp: {patch.name}\n{check.stderr}")
        subprocess.run([*base, str(patch)], cwd=ROOT, check=True)
    write_stamp()

def build(args):
    source = ROOT / "third_party" / "llama.cpp"
    if not (source / "src/models/qwen35moe.cpp").exists():
        # Reuse Strata's pinned source downloader, without running installation or touching its configs.
        sys.path.insert(0, str(ROOT))
        import setup
        setup.get_llama_cpp()
    apply_engine_patches()
    directory = ROOT / ("build-a3b" if args.backend == "cuda" else f"build-a3b-{args.backend}")
    command = ["cmake", "-S", str(ROOT / "a3b"), "-B", str(directory), "-DCMAKE_BUILD_TYPE=Release"]
    if args.backend == "cuda":
        command += ["-DGGML_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}"]
    elif args.backend == "hip":
        command += ["-DGGML_HIP=ON"]
    elif args.backend == "sycl":
        command += ["-DGGML_SYCL=ON"]
    build_command = ["cmake", "--build", str(directory), "--config", "Release", "-j", str(args.jobs)]
    if os.name == "nt":
        selected_vcvars = os.environ.get("STRATA_VCVARS")
        if selected_vcvars:
            vcvars = Path(selected_vcvars)
            if not vcvars.is_file():
                raise FileNotFoundError(f"STRATA_VCVARS does not exist: {vcvars}")
            install = vcvars.parents[3]
            compiler_setup = [str(vcvars)]
        else:
            vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
            install = Path(subprocess.check_output([str(vswhere), "-latest", "-property", "installationPath"], text=True).strip())
            compiler_setup = [str(install / "Common7/Tools/VsDevCmd.bat"), "-arch=x64"]
        ninja = shutil.which("ninja") or str(Path(install) / "Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe")
        command += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
        # Only build commands cross into cmd; filesystem operations stay in Python.
        script = directory / "configure-build.cmd"
        directory.mkdir(parents=True, exist_ok=True)
        script.write_text("@echo off\ncall " + subprocess.list2cmdline(compiler_setup) +
            "\nif errorlevel 1 exit /b %errorlevel%\n" + subprocess.list2cmdline(command) +
            "\nif errorlevel 1 exit /b %errorlevel%\n" + subprocess.list2cmdline(build_command) + "\n", encoding="utf-8")
        subprocess.run(["cmd", "/d", "/c", str(script)], check=True)
    else:
        subprocess.run(command, check=True)
        subprocess.run(build_command, check=True)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--backend", choices=["cuda", "cpu", "hip", "sycl"], default="cuda")
    b.add_argument("--arch", default="75", help="CUDA compute capability; 75=RTX20, 86=RTX30, 89=RTX40, 120=RTX50")
    b.add_argument("--jobs", type=int, default=8)
    d = sub.add_parser("download", help="download a pinned quantization or MTP weights; verify SHA-256")
    d.add_argument("--mtp", action="store_true", help="download a separate MTP model (Q8_0 by default)")
    d.add_argument("--mtp-quant", choices=list(MTP_QUANTS), default="Q8_0")
    d.add_argument("--quant", choices=list(QUANTS), default="Q5_K_M")
    d.add_argument("--workers", type=int, choices=range(1, 65), default=32, metavar="1..64")
    r = sub.add_parser("run")
    r.add_argument("engine_args", nargs=argparse.REMAINDER)
    s = sub.add_parser("config", help="export the GGUF tokenizer and create a separate web/API config")
    s.add_argument("--model", type=Path, default=ROOT / "models/a3b" / MODEL_FILE)
    s.add_argument("--out", type=Path, default=ROOT / "strata-a3b.json")
    s.add_argument("--ctx", type=int, default=4096)
    s.add_argument("--batch", type=int, default=128)
    s.add_argument("--device", type=int, default=0)
    s.add_argument("--vram-mib", type=int, default=0, help="0 uses all available GPU memory")
    s.add_argument("--native-gpu", action="store_true", help="keep all expert weights directly on the chosen GPU")
    s.add_argument("--strata-native-experts", action="store_true", help="opt in to Strata's grouped IQ CUDA expert kernel for short decode/MTP windows; requires native GPU 0")
    s.add_argument("--strata-prefill-mmq", action="store_true", help="opt in to original Strata IQ MMQ for 9..2048-token resident A3B prompts; requires native GPU 0")
    s.add_argument("--strata-fused-gu", action="store_true", help="opt in to Strata's fused IQ gate/up kernel; requires --strata-native-experts")
    s.add_argument("--gpu-only-experts", action="store_true", help="cache experts in RAM/VRAM; compute every routed expert on GPU")
    s.add_argument("--hybrid-experts", action="store_true", help="opt in to concurrent CPU/GPU expert math with a GPU trunk; mutually exclusive with native-GPU and GPU-only modes")
    s.add_argument("--hybrid-doorbell", action="store_true", help="experimental CUDA graph/CPU mailbox; implies hybrid and CPU pool; requires device 0, a local profile, PCIe 0, decay 0 and mtp-draft <=7")
    s.add_argument("--cpu-expert-pool", action="store_true", help="opt in to a shared-activation quantized CPU expert pool; requires --hybrid-experts")
    s.add_argument("--prefill-cpu-share", action="store_true", help="opt in to sparse CPU expert fanout during prefill; requires hybrid and CPU pool (doorbell implies both)")
    s.add_argument("--pcie-frac", type=float, default=None, help="opt in to sending this fraction of distinct missed experts to GPU (0..1); requires --hybrid-experts; omitted keeps CPU misses")
    s.add_argument("--native-dense-exact", action="store_true", help="opt in to exact GPU dense verification; requires a selected native-GPU, GPU-only or hybrid mode")
    s.add_argument("--q6-grouped-exact", action="store_true", help="opt in to bitwise-exact grouped Q6_K dense verification on CUDA sm_75; requires --native-dense-exact and MTP")
    s.add_argument("--native-attention-exact", action="store_true", help="opt in to exact GPU attention verification; requires a selected native-GPU, GPU-only or hybrid mode")
    s.add_argument("--gpu-static-experts", action="store_true", help="experimental CUDA-only GPU routing with a fixed Q8 expert profile; requires GPU-only, device 0 and decay 0")
    s.add_argument("--gpu-dynamic-experts", action="store_true", help="experimental CUDA-managed expert replacement; implies --gpu-static-experts with the same requirements")
    s.add_argument("--expert-staging-slots", type=int, default=8)
    s.add_argument("--expert-ring-banks", type=int, default=2, help="copy-ahead bank count independent of expert compute group size (2..8)")
    s.add_argument("--prefill-copy-ahead", action="store_true", help="stage pageable expert weights on a host worker while prior groups compute")
    s.add_argument("--prefill-independent-issuer", action="store_true", help="experimental separate expert H2D issuer")
    s.add_argument("--prefill-packed-gate-up", action="store_true", help="Q8 long prefill: pack gate/up rows for one shared-input MMQ product")
    s.add_argument("--pageable-experts", action="store_true", help="use ordinary RAM and bounded pinned transfer buffers")
    s.add_argument("--async-expert-events", action="store_true", help="opt in to GPU-only or hybrid inter-layer event scheduling")
    s.add_argument("--expert-profile", type=Path, help="opt in to a local Q8 expert profile; requires GPU-only or hybrid; engine verifies model SHA at startup")
    s.add_argument("--cache-reserve-mib", type=int, default=640)
    s.add_argument("--cache-decay-interval", type=int, default=0, help="opt in to GPU-only or hybrid LFU decay every N windows (0..1024); 0 keeps existing policy")
    s.add_argument("--mtp-model", type=Path, help="opt in to a matching separate MTP GGUF")
    s.add_argument("--mtp-draft", type=int, choices=list(range(1, 9)), default=3)
    s.add_argument("--mtp-ubatch", type=int, default=512)
    s.add_argument("--mtp-adaptive", action="store_true", help="opt in to choosing 0..mtp-draft from measured acceptance/cost; requires --mtp-model")
    s.add_argument("--mtp-fused-rounds", action="store_true", help="opt in to accepted catch-up plus first proposal; requires MTP, a GPU 0 mode, native dense exact and mtp-draft <=7; GPU sampling/defer/exact attention follow the existing MTP mode")
    s.add_argument("--kv-q8", action="store_true", help="opt in to Q8_0 K/V cache; trades KV precision for VRAM and requires performance/quality validation")
    s.add_argument("--kv-q4", action="store_true", help="opt in to Q4_0 K/V cache; trades KV precision for VRAM and requires performance/quality validation")
    s.add_argument("--kv-tiered", action="store_true", help="resident serving: allocate K/V in 64K context tiers while advertising the full --ctx limit")
    s.add_argument("--kv-tier-low-vram", action="store_true", help="with --kv-tiered: 2G uses 1K initial/GPU K/V through 8K; 6G+ uses 64K/GPU K/V, then host K/V")
    s.add_argument("--kv-tiered-f16-short", action="store_true", help="with --kv-tiered: F16 K/V through 131K, Q8_0 above 131K; excludes explicit --kv-q8/--kv-q4")
    args = p.parse_args()
    if args.command == "download": download(args.mtp, args.quant, args.workers, args.mtp_quant)
    elif args.command == "build": build(args)
    elif args.command == "config": make_config(args)
    else:
        exe = ROOT / "build-a3b/bin" / ("strata-moe.exe" if os.name == "nt" else "strata-moe")
        extra = args.engine_args
        if extra and extra[0] == "--": extra = extra[1:]
        if "--model" not in extra and "-m" not in extra: extra = ["--model", str(ROOT / "models/a3b" / MODEL_FILE)] + extra
        subprocess.run([str(exe), *extra], check=True)

def make_config(args):
    from gguf_reader import GGUFFile
    path = args.out.resolve()
    if path.exists():
        raise FileExistsError(f"{path} already exists; retain your settings or move it before regenerating")
    if args.ctx < 32 or args.ctx > 262144 or args.device < 0 or args.vram_mib < 0 or args.batch < 1 or args.mtp_ubatch < 1:
        raise ValueError("invalid context, device or VRAM budget")
    hybrid_experts = args.hybrid_experts or args.hybrid_doorbell
    cpu_expert_pool = args.cpu_expert_pool or args.hybrid_doorbell
    pcie_fraction = 0.0 if args.hybrid_doorbell and args.pcie_frac is None else args.pcie_frac
    if (sum((args.native_gpu, args.gpu_only_experts, hybrid_experts)) > 1 or not 1 <= args.expert_staging_slots <= 64 or
        not 2 <= args.expert_ring_banks <= 8 or
        not 64 <= args.cache_reserve_mib <= 4096):
        raise ValueError("select one GPU expert mode, 1..64 staging slots and 2..8 ring banks")
    managed_experts = args.gpu_only_experts or hybrid_experts
    if args.strata_native_experts and (not args.native_gpu or args.device != 0):
        raise ValueError("--strata-native-experts requires --native-gpu and --device 0")
    if args.strata_prefill_mmq and (not args.native_gpu or args.device != 0):
        raise ValueError("--strata-prefill-mmq requires --native-gpu and --device 0")
    if args.strata_fused_gu and not args.strata_native_experts:
        raise ValueError("--strata-fused-gu requires --strata-native-experts")
    if args.kv_tiered and args.ctx <= 65536:
        raise ValueError("--kv-tiered requires --ctx > 65536")
    if args.kv_tier_low_vram and not args.kv_tiered:
        raise ValueError("--kv-tier-low-vram requires --kv-tiered")
    if args.kv_tiered_f16_short and (not args.kv_tiered or args.kv_q8 or args.kv_q4):
        raise ValueError("--kv-tiered-f16-short requires --kv-tiered and excludes --kv-q8/--kv-q4")
    if args.prefill_copy_ahead and not managed_experts:
        raise ValueError("--prefill-copy-ahead requires managed GPU experts")
    if args.prefill_independent_issuer and (not args.prefill_copy_ahead or not args.pageable_experts):
        raise ValueError("--prefill-independent-issuer requires --prefill-copy-ahead and --pageable-experts")
    if args.prefill_packed_gate_up and not args.prefill_independent_issuer:
        raise ValueError("--prefill-packed-gate-up requires --prefill-independent-issuer")
    if cpu_expert_pool and not hybrid_experts:
        raise ValueError("--cpu-expert-pool requires --hybrid-experts")
    if args.prefill_cpu_share and not (hybrid_experts and cpu_expert_pool):
        raise ValueError("--prefill-cpu-share requires --hybrid-experts and --cpu-expert-pool (or --hybrid-doorbell)")
    if pcie_fraction is not None and (not hybrid_experts or not math.isfinite(pcie_fraction) or
                                      not 0 <= pcie_fraction <= 1):
        raise ValueError("--pcie-frac requires --hybrid-experts and a finite fraction in 0..1")
    if args.async_expert_events and not managed_experts:
        raise ValueError("--async-expert-events requires --gpu-only-experts or --hybrid-experts")
    if (args.native_dense_exact or args.native_attention_exact) and not (args.native_gpu or managed_experts):
        raise ValueError("exact native options require --native-gpu, --gpu-only-experts or --hybrid-experts")
    if args.q6_grouped_exact and (not args.native_dense_exact or not args.mtp_model):
        raise ValueError("--q6-grouped-exact requires --native-dense-exact and --mtp-model")
    if args.mtp_model and args.batch <= args.mtp_draft:
        raise ValueError("MTP requires batch greater than mtp-draft")
    if args.mtp_adaptive and not args.mtp_model:
        raise ValueError("--mtp-adaptive requires --mtp-model")
    if not 0 <= args.cache_decay_interval <= 1024:
        raise ValueError("--cache-decay-interval must be 0..1024")
    if args.cache_decay_interval and not managed_experts:
        raise ValueError("--cache-decay-interval requires --gpu-only-experts or --hybrid-experts")
    expert_profile = args.expert_profile.resolve() if args.expert_profile else None
    if expert_profile:
        if not managed_experts:
            raise ValueError("--expert-profile requires --gpu-only-experts or --hybrid-experts")
        if not expert_profile.is_file():
            raise FileNotFoundError(expert_profile)
    gpu_static_experts = args.gpu_static_experts or args.gpu_dynamic_experts
    if args.hybrid_doorbell and (gpu_static_experts or args.device != 0 or not expert_profile or
                                pcie_fraction != 0.0 or args.cache_decay_interval or args.mtp_draft > 7):
        raise ValueError("--hybrid-doorbell requires device 0, a local expert profile, PCIe fraction 0, decay interval 0 and mtp-draft <=7")
    if gpu_static_experts and (not args.gpu_only_experts or not expert_profile or
                              args.device != 0 or args.cache_decay_interval):
        selected_mode = "--gpu-dynamic-experts" if args.gpu_dynamic_experts else "--gpu-static-experts"
        raise ValueError(f"{selected_mode} requires --gpu-only-experts, --expert-profile, --device 0 and decay interval 0")
    if args.mtp_fused_rounds:
        # Every selected GPU mode already injects GPU sampling/deferred MTP
        # catch-up and exact attention. Static MTP also injects exact dense.
        if (not args.mtp_model or not (args.native_gpu or managed_experts) or args.device != 0 or
            not (args.native_dense_exact or gpu_static_experts) or args.mtp_draft > 7):
            raise ValueError("--mtp-fused-rounds requires --mtp-model, a GPU 0 mode, --native-dense-exact and mtp-draft <=7")
        if not args.mtp_model.is_file():
            raise FileNotFoundError(args.mtp_model.resolve())
    model = args.model.resolve()
    gguf = GGUFFile(model)
    meta = gguf.metadata
    if meta.get("general.architecture") != "qwen35moe":
        raise ValueError("requires a qwen35moe GGUF")
    # Sharing requires the same quantized storage format before the engine
    # checks full tensor contents. IQ3_S has Q6_K embedding/head tensors;
    # the Q8 draft's corresponding tensors are Q8_0. Synthetic config
    # fixtures have no tensors and retain the earlier config-test behavior.
    tensor_types = {tensor.name: tensor.type_name for tensor in gguf.tensors}
    share_mtp = bool(args.mtp_model) and model.name != QUANTS["IQ3_S"][0] and (
        not tensor_types or (args.mtp_model.name == MTP_FILE and
            tensor_types.get("token_embd.weight") == "Q8_0" and
            tensor_types.get("output.weight") == "Q8_0" and
            tensor_types.get("output_norm.weight") == "F32"))
    folder = ROOT / "models/a3b" / (model.stem + "-tokenizer")
    folder.mkdir(parents=True, exist_ok=True)
    tokens = meta["tokenizer.ggml.tokens"]
    (folder / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False), encoding="utf-8")
    (folder / "merges.txt").write_text("\n".join(meta["tokenizer.ggml.merges"]), encoding="utf-8")
    (folder / "token_type.json").write_text(json.dumps(meta["tokenizer.ggml.token_type"]), encoding="utf-8")
    (folder / "chat_template.jinja").write_text(meta["tokenizer.chat_template"], encoding="utf-8")
    cfg = {"model_name": model.stem.lower(), "exe": str(ROOT / "build-a3b/bin/strata-moe.exe") if os.name == "nt" else str(ROOT / "build-a3b/bin/strata-moe"),
        "cwd": str(ROOT), "tokenizer": str(folder), "host": "127.0.0.1", "port": 8096,
        "log": str(ROOT / "a3b-engine.log"),
        "args": ["--model", str(model), "--ctx", str(args.ctx), "--batch", str(args.batch), "--device", str(args.device), "--vram-mib", str(args.vram_mib)],
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20}, "open_browser": False}
    if args.mtp_model:
        mtp = args.mtp_model.resolve()
        if not mtp.is_file():
            raise FileNotFoundError(mtp)
        cfg["args"] += ["--mtp-model", str(mtp), "--mtp-draft", str(args.mtp_draft), "--mtp-ubatch", str(args.mtp_ubatch), "--exact-verify"]
        if args.mtp_adaptive:
            cfg["args"] += ["--mtp-adaptive"]
        if args.mtp_fused_rounds:
            cfg["args"] += ["--mtp-fused-rounds"]
    if args.native_gpu:
        cfg["args"] += ["--native-gpu", "--gpu-embedding", "--gpu-sampling"]
        if args.strata_native_experts:
            cfg["args"] += ["--strata-native-experts"]
        if args.strata_prefill_mmq:
            cfg["args"] += ["--strata-prefill-mmq"]
        if args.mtp_model:
            cfg["args"] += (["--share-mtp-weights"] if share_mtp else []) + ["--defer-mtp-catchup", "--fused-moe"]
            if not args.kv_tiered_f16_short:
                cfg["args"] += ["--native-attention-exact"]
    if managed_experts:
        mode = "--hybrid-experts" if hybrid_experts else "--gpu-only-experts"
        cfg["args"] += [mode, "--expert-staging-slots", str(args.expert_staging_slots),
                        "--gpu-embedding", "--gpu-sampling"]
        if args.expert_ring_banks != 2:
            cfg["args"] += ["--expert-ring-banks", str(args.expert_ring_banks)]
        if args.prefill_copy_ahead:
            cfg["args"] += ["--prefill-copy-ahead"]
        if args.prefill_independent_issuer:
            cfg["args"] += ["--prefill-independent-issuer"]
        if args.prefill_packed_gate_up:
            cfg["args"] += ["--prefill-packed-gate-up"]
        if hybrid_experts:
            cfg["args"] += ["--fused-moe"]
            if args.hybrid_doorbell:
                cfg["args"] += ["--hybrid-doorbell"]
            if cpu_expert_pool:
                cfg["args"] += ["--cpu-expert-pool"]
            if args.prefill_cpu_share:
                cfg["args"] += ["--prefill-cpu-share"]
            if pcie_fraction is not None:
                cfg["args"] += ["--pcie-frac", str(pcie_fraction)]
        if args.pageable_experts:
            cfg["args"] += ["--pageable-experts"]
        if args.async_expert_events:
            cfg["args"] += ["--async-expert-events"]
        if expert_profile:
            cfg["args"] += ["--expert-profile", str(expert_profile)]
        if args.cache_decay_interval or args.hybrid_doorbell:
            cfg["args"] += ["--cache-decay-interval", str(args.cache_decay_interval)]
        if gpu_static_experts:
            cfg["args"] += ["--gpu-static-experts"]
            if args.gpu_dynamic_experts:
                cfg["args"] += ["--gpu-dynamic-experts"]
            if args.mtp_model:
                cfg["args"] += ["--native-dense-exact"]
        if args.mtp_model:
            cfg["args"] += (["--share-mtp-weights"] if share_mtp else []) + ["--defer-mtp-catchup", "--native-attention-exact"]
    for option, selected in (("--native-dense-exact", args.native_dense_exact),
                             ("--native-attention-exact", args.native_attention_exact)):
        if selected and option not in cfg["args"]:
            cfg["args"] += [option]
    if args.kv_q8 and args.kv_q4:
        raise ValueError("--kv-q8 and --kv-q4 are mutually exclusive")
    if args.kv_q8:
        cfg["args"] += ["--kv-q8"]
    if args.kv_q4:
        cfg["args"] += ["--kv-q4"]
    if args.kv_tiered:
        cfg["args"] += ["--kv-tiered"]
    if args.kv_tier_low_vram:
        cfg["args"] += ["--kv-tier-low-vram"]
    if args.kv_tiered_f16_short:
        cfg["args"] += ["--kv-tiered-f16-short"]
    if args.q6_grouped_exact or args.strata_fused_gu:
        cfg["env"] = {}
        if args.q6_grouped_exact:
            cfg["env"]["STRATA_A3B_Q6_GROUPED_EXACT"] = "1"
        if args.strata_fused_gu:
            cfg["env"]["STRATA_A3B_FUSED_GU"] = "1"
    if args.cache_reserve_mib != 640:
        cfg["args"] += ["--cache-reserve-mib", str(args.cache_reserve_mib)]
    cfg["idle_unload_s"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Created {path}\nStart: python -m serve.server --engine strata --config {path} --port 8096")

if __name__ == "__main__":
    main()
