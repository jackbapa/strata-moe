"""Summarize CUDA runtime work inside real A3B K2 NVTX ranges.

Usage: python analyze_graph.py k2-on-nvtx.sqlite
Counts are CUDA API calls, not a claim that the two GGML decode graphs have
become one original-Strata round/step/commit graph.
"""

import json
import sqlite3
import sys
from collections import Counter


def main(path):
    db = sqlite3.connect(path)
    ranges = list(db.execute("""
        SELECT n.start, n.end, n.globalTid
        FROM NVTX_EVENTS n LEFT JOIN StringIds s ON n.textId=s.id
        WHERE n.end IS NOT NULL
          AND coalesce(n.text,s.value)='a3b/mtp/proposal/device-chain-k2'
        ORDER BY n.start
    """))
    api = Counter()
    api_duration_ns = Counter()
    copies = Counter()
    per_range = []
    for start, end, tid in ranges:
        rows = list(db.execute("""
            SELECT s.value,r.end-r.start FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN StringIds s ON s.id=r.nameId
            WHERE r.start>=? AND r.end<=? AND r.globalTid=?
              AND (s.value LIKE '%Graph%' OR s.value LIKE '%Synchronize%'
                   OR s.value LIKE '%Memcpy%')
        """, (start, end, tid)))
        calls = Counter(name for name, _ in rows)
        api.update(calls)
        api_duration_ns.update({name: sum(duration for n, duration in rows if n == name)
                                for name in calls})
        copies.update((kind, size) for kind, size in db.execute("""
            SELECT m.copyKind, m.bytes FROM CUPTI_ACTIVITY_KIND_RUNTIME r
            JOIN CUPTI_ACTIVITY_KIND_MEMCPY m ON m.correlationId=r.correlationId
            WHERE r.start>=? AND r.end<=? AND r.globalTid=?
        """, (start, end, tid)))
        per_range.append({"duration_ms": (end-start)/1e6, "cuda_api": calls})
    print(json.dumps({"chain_ranges": len(ranges),
                      "total_duration_ms": sum(x["duration_ms"] for x in per_range),
                      "cuda_api_total": api,
                      "cuda_api_total_duration_ms": {k: v/1e6 for k, v in api_duration_ns.items()},
                      "graph_launch_per_range": dict(Counter(x["cuda_api"].get("cudaGraphLaunch_v10000", 0)
                                                            for x in per_range)),
                      "memcpy_kind_bytes_count": [{"kind": kind, "bytes": size, "count": count}
                                                  for (kind, size), count in sorted(copies.items())],
                      "first_five_ranges": per_range[:5]}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
