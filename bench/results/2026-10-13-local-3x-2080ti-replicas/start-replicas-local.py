"""Local Windows launcher for this PC's three 2080 Ti replicas.

Separate each engine onto eight physical 3960X cores. The upstream Windows
pool otherwise pins every engine to the same first cores. No engine changes.
"""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import sys

import psutil

sys.path.insert(0, r"G:\Strata")
from serve import server


def pin_engine(pid, index):
    if os.name != "nt" or psutil.cpu_count() != 48:
        raise RuntimeError("This local affinity layout requires Windows and the 48-thread 3960X")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)

    class GroupAffinity(ctypes.Structure):
        _fields_ = [("Mask", ctypes.c_size_t), ("Group", wintypes.WORD),
                    ("Reserved", wintypes.WORD * 3)]

    kernel.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenThread.restype = wintypes.HANDLE
    kernel.GetThreadGroupAffinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(GroupAffinity)]
    kernel.GetThreadGroupAffinity.restype = wintypes.BOOL
    kernel.SetThreadGroupAffinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(GroupAffinity),
                                             ctypes.POINTER(GroupAffinity)]
    kernel.SetThreadGroupAffinity.restype = wintypes.BOOL
    kernel.SetThreadSelectedCpuSets.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.ULONG), wintypes.ULONG]
    kernel.SetThreadSelectedCpuSets.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    first = index * 16
    full_mask = ((1 << 16) - 1) << first
    process = psutil.Process(pid)
    pinned = []
    for thread in process.threads():
        handle = kernel.OpenThread(0x0040 | 0x0020, False, thread.id)
        if not handle:
            if ctypes.get_last_error() == 87:  # thread finished between enumeration and opening
                continue
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            old = GroupAffinity()
            if not kernel.GetThreadGroupAffinity(handle, ctypes.byref(old)):
                raise ctypes.WinError(ctypes.get_last_error())
            mask = int(old.Mask)
            if thread.id == pid:
                target = 1 << first
            elif old.Group == 0 and mask and mask.bit_count() == 1 and mask.bit_length() <= 16:
                target = mask << first
            else:
                target = full_mask
            if not kernel.SetThreadSelectedCpuSets(handle, None, 0):
                raise ctypes.WinError(ctypes.get_last_error())
            new = GroupAffinity(target, 0)
            if not kernel.SetThreadGroupAffinity(handle, ctypes.byref(new), None):
                raise ctypes.WinError(ctypes.get_last_error())
            if target.bit_count() == 1:
                pinned.append({"thread": thread.id, "cpu": target.bit_length() - 1})
        finally:
            kernel.CloseHandle(handle)
    process.cpu_affinity(list(range(first, first + 16)))
    print(f"[local affinity] GPU {index}, PID {pid}: logical CPUs {first}-{first + 15}, "
          f"pinned threads {pinned}", flush=True)
    return {"logical_cpus": list(range(first, first + 16)), "threads": pinned}


original_init = server.StrataEngine.__init__


def local_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if self.proc is None:
        return
    env = self.spawn[4] or {}
    gpu = env.get("CUDA_VISIBLE_DEVICES", "")
    if gpu not in ("0", "1", "2"):
        raise RuntimeError(f"This launcher expects one of GPU 0/1/2 per replica, got {gpu!r}")
    self.info["local_cpu_affinity"] = pin_engine(self.proc.pid, int(gpu))


if __name__ == "__main__":
    os.environ["STRATA_REPLICA_START"] = "serial"
    server.StrataEngine.__init__ = local_init
    raise SystemExit(server.main())
