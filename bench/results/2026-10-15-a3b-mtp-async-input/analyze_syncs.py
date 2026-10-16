"""Classify CUDA stream sync calls inside the NVTX MTP K2 ranges."""

import json
import sqlite3
import sys
from collections import Counter, defaultdict


def main(path):
    db = sqlite3.connect(path)
    mem = {cid: (kind, size) for cid, kind, size in db.execute("""
        SELECT correlationId, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY
    """)}
    ranges = list(db.execute("""
        SELECT n.start, n.end, n.globalTid FROM NVTX_EVENTS n
        LEFT JOIN StringIds s ON n.textId=s.id
        WHERE n.end IS NOT NULL
          AND coalesce(n.text,s.value)='a3b/mtp/proposal/device-chain-k2'
        ORDER BY n.start
    """))
    count = Counter()
    duration = defaultdict(int)
    examples = defaultdict(list)
    for start, end, tid in ranges:
        calls = list(db.execute("""
            SELECT r.start, r.end, s.value, r.correlationId
            FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
            WHERE r.start>=? AND r.end<=? AND r.globalTid=? ORDER BY r.start
        """, (start, end, tid)))
        for i, (beg, done, name, _) in enumerate(calls):
            if not name.startswith("cudaStreamSynchronize"):
                continue
            prev = calls[i - 1] if i else None
            if prev is None:
                label = "no prior API"
            else:
                _, _, pn, pcorr = prev
                pmem = mem.get(pcorr)
                label = f"{pn}: {pmem}" if pmem else pn
            count[label] += 1
            duration[label] += done - beg
            if len(examples[label]) < 3:
                examples[label].append((beg - start) / 1e6)
    print(json.dumps({
        "ranges": len(ranges),
        "sync_total": sum(count.values()),
        "by_preceding_api": [
            {"previous": k, "count": v, "sync_duration_ms": duration[k] / 1e6,
             "example_offset_ms": examples[k]}
            for k, v in count.most_common()
        ],
    }, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
