"""Read-only Q8_0 MTP kernel and GGUF geometry inspection.

No GPU use, model load, tensor data read, or runtime mutation. The GGUF reader
memory-maps the file but this script only accesses tensor metadata.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "third_party" / "llama.cpp" / "gguf-py"))
from gguf.gguf_reader import GGUFReader  # noqa: E402


def main() -> None:
    trace = ROOT / "bench/results/2026-10-15-a3b-iq3s-f16-native-node/f16-native-output.sqlite"
    db = sqlite3.connect(f"file:{trace.as_posix()}?mode=ro", uri=True)
    print("Q8_0 kernel template counts/durations, full trace:")
    for name, count, ms in db.execute(
        """SELECT s.value, COUNT(*), ROUND(SUM(k.end-k.start)/1e6, 3)
           FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
           JOIN StringIds AS s ON s.id=k.demangledName
           WHERE s.value LIKE 'void mul_mat_vec_q<(ggml_type)8,%'
           GROUP BY s.value ORDER BY 3 DESC"""
    ):
        print(f"{count:6d} {ms:9.3f} ms {name}")
    kernel_cols = {item[1] for item in db.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    if {"gridX", "gridY", "gridZ", "blockX", "blockY"} <= kernel_cols:
        print("\nQ8_0 grid/block geometry, full trace:")
        for row in db.execute(
            """SELECT k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY,
                      COUNT(*), ROUND(SUM(k.end-k.start)/1e6, 3),
                      ROUND(AVG(k.end-k.start)/1000, 3)
               FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
               JOIN StringIds AS s ON s.id=k.demangledName
               WHERE s.value LIKE 'void mul_mat_vec_q<(ggml_type)8,%'
               GROUP BY k.gridX, k.gridY, k.gridZ, k.blockX, k.blockY
               ORDER BY 7 DESC"""
        ):
            print(f"grid={row[0:3]} block={row[3:5]} calls={row[5]} total={row[6]}ms mean={row[7]}us")
        print("\nQ8_0 attribution by NVTX phase and grid:")
        for row in db.execute(
            """WITH launches AS (
                   SELECT n.text AS phase, k.gridX AS gx, k.gridY AS gy, k.gridZ AS gz,
                          k.start AS ks, k.end AS ke
                   FROM NVTX_EVENTS AS n
                   JOIN CUPTI_ACTIVITY_KIND_RUNTIME AS r
                     ON r.globalTid=n.globalTid AND r.start>=n.start AND r.end<=n.end
                   JOIN CUPTI_ACTIVITY_KIND_KERNEL AS k ON k.correlationId=r.correlationId
                   JOIN StringIds AS s ON s.id=k.demangledName
                   WHERE n.text IN ('a3b/output/target-verify','a3b/output/mtp-propose')
                     AND n.end IS NOT NULL
                     AND s.value LIKE 'void mul_mat_vec_q<(ggml_type)8,%'
               )
               SELECT phase, gx, gy, gz, COUNT(*), ROUND(SUM(ke-ks)/1e6, 3)
               FROM launches GROUP BY phase, gx, gy, gz ORDER BY phase, 6 DESC"""
        ):
            print(f"{row[0]:26s} grid={row[1:4]} calls={row[4]:4d} total={row[5]:8.3f}ms")
    db.close()

    draft = ROOT / "models/a3b/mtp-Qwen3.6-35B-A3B-Q8_0.gguf"
    reader = GGUFReader(draft)
    print("\nMTP GGUF Q8_0 2D tensors (metadata only):")
    for tensor in reader.tensors:
        if tensor.tensor_type.name == "Q8_0" and len(tensor.shape) == 2:
            print(f"{tensor.name:55s} {[int(x) for x in tensor.shape]} bytes={tensor.n_bytes} file_offset={tensor.data_offset}")


if __name__ == "__main__":
    main()
