"""Windows A3B process-memory audit; launches and quits only its own engine."""
import argparse
import ctypes as C
from ctypes import wintypes as W
import json
from pathlib import Path
import queue
import subprocess
import threading
import time

import psutil


class MBI(C.Structure):
    _fields_ = [("BaseAddress", C.c_void_p), ("AllocationBase", C.c_void_p),
                ("AllocationProtect", W.DWORD), ("PartitionId", W.WORD),
                ("RegionSize", C.c_size_t), ("State", W.DWORD),
                ("Protect", W.DWORD), ("Type", W.DWORD)]


def windows_regions(pid):
    k = C.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.argtypes = [W.DWORD, W.BOOL, W.DWORD]
    k.OpenProcess.restype = W.HANDLE
    k.VirtualQueryEx.argtypes = [W.HANDLE, C.c_void_p, C.POINTER(MBI), C.c_size_t]
    k.VirtualQueryEx.restype = C.c_size_t
    k.K32QueryWorkingSetEx.argtypes = [W.HANDLE, C.c_void_p, W.DWORD]
    k.K32QueryWorkingSetEx.restype = W.BOOL
    k.CloseHandle.argtypes = [W.HANDLE]
    h = k.OpenProcess(0x0400 | 0x0010, False, pid)
    if not h:
        raise C.WinError(C.get_last_error())
    names = {0x20000: "private", 0x40000: "mapped", 0x1000000: "image"}
    totals = {}
    largest = []
    committed = []
    addr = 0
    try:
        while True:
            m = MBI()
            if not k.VirtualQueryEx(h, C.c_void_p(addr), C.byref(m), C.sizeof(m)):
                break
            base, size = m.BaseAddress or 0, m.RegionSize
            if size == 0:
                break
            if m.State == 0x1000:
                name = names.get(m.Type, hex(m.Type))
                key = name + "/protect=" + hex(m.Protect)
                totals[key] = totals.get(key, 0) + size
                committed.append((base, size, name, m.Protect))
                if size >= 64 * 1024 * 1024:
                    largest.append({"address": hex(base), "allocation_base": hex(m.AllocationBase or 0),
                                    "bytes": size, "type": name, "protect": hex(m.Protect)})
            addr = base + size
        # A regular 64 KiB sample estimates residency independently from commitment.
        # Valid + shared bits come from PSAPI_WORKING_SET_EX_BLOCK.
        by_type = {}
        chunk = []

        class WS(C.Structure):
            _fields_ = [("address", C.c_void_p), ("flags", C.c_size_t)]

        def flush():
            if not chunk:
                return
            a = (WS * len(chunk))(*[WS(v[0], 0) for v in chunk])
            if not k.K32QueryWorkingSetEx(h, a, C.sizeof(a)):
                raise C.WinError(C.get_last_error())
            for v, item in zip(a, chunk):
                _, weight, name = item
                s = by_type.setdefault(name, {"sampled_committed": 0, "estimated_resident": 0,
                                              "estimated_resident_shared": 0})
                s["sampled_committed"] += weight
                if v.flags & 1:
                    s["estimated_resident"] += weight
                    if v.flags & (1 << 15):
                        s["estimated_resident_shared"] += weight
            chunk.clear()

        for base, size, name, protect in committed:
            if protect & 0x100 or protect == 0x01:
                continue
            for offset in range(0, size, 65536):
                chunk.append((base + offset, min(65536, size - offset), name))
                if len(chunk) == 16384:
                    flush()
        flush()
        return {"committed_by_type_and_protection": totals,
                "largest_committed_regions": sorted(largest, key=lambda x: x["bytes"], reverse=True)[:40],
                "residency_64k_sample": by_type}
    finally:
        k.CloseHandle(h)


def vram():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
                             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    return result.stdout.strip().splitlines()


def snapshot(pid=None):
    r = {"time": time.time(), "system_available": psutil.virtual_memory().available,
         "system_used": psutil.virtual_memory().used, "vram_mib": vram()}
    if pid is not None:
        p = psutil.Process(pid)
        r["pid"] = pid
        r["memory"] = p.memory_info()._asdict()
        r["mapped_files"] = [{"path": m.path, "rss": m.rss}
                               for m in p.memory_maps(grouped=True) if m.rss >= 64 * 1024 * 1024]
        r["virtual_regions"] = windows_regions(pid)
    return r


def arm(exe, model, directory, name, extra):
    command = [str(exe), "--serve", "--model", str(model), "--ctx", "4096", "--device", "0",
               "--vram-mib", "16384", *extra]
    result = {"name": name, "command": command, "before": snapshot()}
    lines = queue.Queue()
    with (directory / (name + ".stderr.log")).open("w", encoding="utf-8") as err:
        p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                             text=True, encoding="utf-8", errors="replace", bufsize=1)

        def read_stdout():
            for line in p.stdout:
                lines.put(line.rstrip())
            lines.put(None)

        thread = threading.Thread(target=read_stdout, daemon=True)
        thread.start()
        output = []
        try:
            while True:
                line = lines.get(timeout=180)
                if line is None:
                    raise RuntimeError("engine exited before READY")
                output.append(line)
                if line.startswith("READY"):
                    break
            result["ready"] = snapshot(p.pid)
            p.stdin.write("GEN 4 temperature=0 151644,872,198,9707,151645,198,151644,77091,198\n")
            p.stdin.flush()
            while True:
                line = lines.get(timeout=180)
                if line is None:
                    raise RuntimeError("engine exited before DONE")
                output.append(line)
                if line.startswith("DONE") or line.startswith("ERR"):
                    break
            result["after_generation"] = snapshot(p.pid)
        finally:
            if p.poll() is None:
                p.stdin.write("QUIT\n")
                p.stdin.flush()
                p.wait(timeout=60)
            result["returncode"] = p.returncode
            result["stdout"] = output
    result["after_quit"] = snapshot()
    (directory / (name + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(name, "ready RSS/private:", result["ready"]["memory"]["rss"],
          result["ready"]["memory"].get("private"), "after GEN RSS/private:",
          result["after_generation"]["memory"]["rss"],
          result["after_generation"]["memory"].get("private"), flush=True)
    print("GPU after quit:", result["after_quit"]["vram_mib"], flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int)
    ap.add_argument("--exe", type=Path, default=Path(__file__).resolve().parents[1] / "build-a3b/bin/strata-moe.exe")
    ap.add_argument("--model", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", choices=["cache0", "cache16g", "cache16g-mmap", "cache16g-pageable"], default=["cache0", "cache16g"])
    ap.add_argument("--arm", choices=["cache0", "cache16g", "both"], help="Legacy single-arm selector")
    ap.add_argument("--extra-arg", action="append", default=[], help="Append engine argument, e.g. --extra-arg=--mmap")
    args = ap.parse_args()
    if not args.pid and not args.model:
        ap.error("--model is required unless inspecting --pid")
    args.out.mkdir(parents=True, exist_ok=True)
    if args.pid:
        (args.out / "snapshot.json").write_text(json.dumps(snapshot(args.pid), indent=2), encoding="utf-8")
    else:
        arms = {"cache0": ["--cache-mib", "0"], "cache16g": [],
                "cache16g-mmap": ["--mmap"], "cache16g-pageable": ["--pageable-experts"]}
        selected = args.arms if args.arm is None else (["cache0", "cache16g"] if args.arm == "both" else [args.arm])
        for name in selected:
            arm(args.exe.resolve(), args.model.resolve(), args.out, name, arms[name] + args.extra_arg)
