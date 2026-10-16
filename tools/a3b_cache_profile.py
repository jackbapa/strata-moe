"""Build a model-bound, globally ranked Q8 expert cache profile from route TSVs.

Each independent trace contributes max(gate, up, down) per layer/expert once.
Coverage is measured on the supplied training traces, never a speed or a
generalization guarantee. The model's entire file is SHA-256 hashed.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time

from gguf_reader import GGUFFile

MAGIC = "STRATA_A3B_EXPERT_PROFILE_V1"
LAYERS, EXPERTS, BUNDLE_BYTES = 40, 256, 3342336
MAX_SCORE = 2**64 - 1
GEOMETRY = {"block_count": 40, "embedding_length": 2048, "expert_count": 256,
            "expert_used_count": 8, "expert_feed_forward_length": 512,
            "attention.head_count": 16, "attention.head_count_kv": 2,
            "ssm.state_size": 128, "ssm.group_count": 16, "ssm.time_step_rank": 32,
            "ssm.conv_kernel": 4, "full_attention_interval": 4}
ROUTED = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while data := source.read(16 * 1024 * 1024):
            digest.update(data)
    return digest.hexdigest()


def validate_q8_geometry(model, model_bytes):
    if model.metadata.get("general.architecture") != "qwen35moe":
        raise ValueError("profile requires qwen35moe")
    if model.metadata.get("split.count", 1) != 1:
        raise ValueError("profile requires one complete GGUF")
    for key, expected in GEOMETRY.items():
        if model.metadata.get("qwen35moe." + key) != expected:
            raise ValueError("profile model geometry mismatch: " + key)
    seen, bundle = set(), [0] * LAYERS
    for tensor in model.tensors:
        size = tensor.expected_bytes()
        if size is None or tensor.offset < 0 or model.data_start + tensor.offset + size > model_bytes:
            raise ValueError("unsupported or truncated model tensor: " + tensor.name)
        match = ROUTED.fullmatch(tensor.name)
        if not match:
            if "_exps.weight" in tensor.name:
                raise ValueError("merged/unknown expert tensor unsupported")
            continue
        layer, projection = int(match[1]), match[2]
        shape = [512, 2048, 256] if projection == "down" else [2048, 512, 256]
        if (not 0 <= layer < LAYERS or (layer, projection) in seen or tensor.type_name != "Q8_0" or
                tensor.shape not in (shape, shape + [1]) or size != BUNDLE_BYTES // 3 * EXPERTS):
            raise ValueError("profile requires 120 canonical Q8_0 expert tensors")
        seen.add((layer, projection))
        bundle[layer] += size // EXPERTS
    if len(seen) != 120 or any(size != BUNDLE_BYTES for size in bundle):
        raise ValueError("Q8_0 expert bundle mismatch")
    return {"layers": LAYERS, "experts": EXPERTS, "embedding": 2048,
            "ffn": 512, "experts_used": 8, "bundle_bytes": BUNDLE_BYTES}


def inspect_model(path):
    path = Path(path)
    model_bytes = path.stat().st_size
    geometry = validate_q8_geometry(GGUFFile(path), model_bytes)
    began = time.perf_counter()
    sha = file_sha256(path)
    return {**geometry, "model_bytes": model_bytes, "model_sha256": sha,
            "model_hash_s": time.perf_counter() - began}


def integer(value, field):
    if not re.fullmatch(r"[0-9]+", value or ""):
        raise ValueError("invalid nonnegative integer in " + field)
    number = int(value)
    if number > MAX_SCORE:
        raise ValueError("integer overflow in " + field)
    return number


def parse_trace(data):
    scores, rows = {}, 0
    with io.StringIO(data.decode("utf-8-sig"), newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if not {"tensor", "expert", "routes"}.issubset(reader.fieldnames or []):
            raise ValueError("trace needs tensor/expert/routes TSV columns")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("duplicate trace columns")
        for row in reader:
            if None in row:
                raise ValueError("trace row has too many TSV columns")
            match = ROUTED.fullmatch(row["tensor"] or "")
            if not match:
                raise ValueError("unsupported trace tensor: " + str(row["tensor"]))
            layer, expert = int(match[1]), integer(row["expert"], "expert")
            score = integer(row["routes"], "routes")
            if not 0 <= layer < LAYERS or not 0 <= expert < EXPERTS:
                raise ValueError("trace layer/expert out of range")
            # Fused traces record only gate; unfused traces record three
            # projections. Repeated backend rows also remain one observation.
            scores[layer, expert] = max(scores.get((layer, expert), 0), score)
            rows += 1
    if not rows:
        raise ValueError("empty route trace")
    return scores, rows


def read_trace(path):
    return parse_trace(Path(path).read_bytes())


def aggregate_traces(paths):
    totals, provenance, seen = {}, [], set()
    for value in paths:
        path = Path(value).resolve()
        if path in seen:
            raise ValueError("same trace path supplied more than once")
        seen.add(path)
        # Fingerprint the exact snapshot being parsed, even if a caller has
        # accidentally left its trace writer running.
        data = path.read_bytes()
        scores, rows = parse_trace(data)
        for key, score in scores.items():
            totals[key] = totals.get(key, 0) + score
            if totals[key] > MAX_SCORE:
                raise ValueError("aggregated expert score overflow")
        provenance.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(),
                           "bytes": len(data), "tsv_rows": rows,
                           "expert_observations": len(scores)})
    if not provenance:
        raise ValueError("at least one independent training trace is required")
    ordered = [(layer, expert, totals.get((layer, expert), 0))
               for layer in range(LAYERS) for expert in range(EXPERTS)]
    ordered.sort(key=lambda row: (-row[2], row[0], row[1]))
    if sum(row[2] for row in ordered) > MAX_SCORE:
        raise ValueError("total expert score overflow")
    return ordered, provenance


def select_plan(scores, budget_bytes):
    if budget_bytes < 0:
        raise ValueError("budget must be nonnegative")
    capacity = budget_bytes // BUNDLE_BYTES
    positive = [row for row in scores if row[2] > 0]
    selected = positive[:capacity]
    expert_ids = [[] for _ in range(LAYERS)]
    for layer, expert, _ in selected:
        expert_ids[layer].append(expert)
    total = sum(row[2] for row in scores)
    covered = sum(row[2] for row in selected)
    return {"budget_bytes": budget_bytes, "planned_bytes": len(selected) * BUNDLE_BYTES,
            "bundle_bytes": BUNDLE_BYTES, "selected_experts": len(selected),
            "total_score": total, "covered_score": covered,
            "training_trace_coverage": covered / total if total else 0,
            "per_layer_capacity": list(map(len, expert_ids)), "expert_ids": expert_ids}


def generate_profile(model, traces, output, benchmark_json=(), budget_bytes=0):
    output, model = Path(output).resolve(), Path(model).resolve()
    sidecar = output.with_suffix(output.suffix + ".provenance.json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("profile and provenance outputs must be new files")
    identity = inspect_model(model)
    scores, sources = aggregate_traces(traces)
    benchmarks = []
    for value in benchmark_json:
        path = Path(value).resolve()
        # Validate JSON syntax but never execute its contents or treat prose as instructions.
        json.loads(path.read_text(encoding="utf-8"))
        benchmarks.append({"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size})
    source = {"aggregation": "per trace max across projections/backends for each layer/expert; sum independent traces",
              "model_sha256": identity["model_sha256"], "model_bytes": identity["model_bytes"],
              "traces": sources, "benchmark_json": benchmarks}
    canonical = json.dumps(source, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    fields = ["model_sha256", "model_bytes", "layers", "experts", "embedding", "ffn", "experts_used", "bundle_bytes"]
    text = MAGIC + "\n" + "".join(f"{key} {identity[key]}\n" for key in fields)
    text += f"source_fingerprint {fingerprint}\nscores {LAYERS * EXPERTS}\n"
    text += "".join(f"{layer}\t{expert}\t{score}\n" for layer, expert, score in scores)
    plan = select_plan(scores, budget_bytes)
    provenance = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "model": str(model),
                  "model_identity": identity, "source": source, "source_fingerprint": fingerprint,
                  "profile_sha256": hashlib.sha256(text.encode()).hexdigest(), "preview_plan": plan,
                  "scope": "Coverage is on these training traces only. Replaying the training trace is not independent validation, generalization evidence or measured throughput."}
    output.parent.mkdir(parents=True, exist_ok=True)
    partial, partial_sidecar = output.with_suffix(output.suffix + ".tmp"), sidecar.with_suffix(sidecar.suffix + ".tmp")
    if partial.exists() or partial_sidecar.exists():
        raise FileExistsError("unfinished profile output exists")
    # Write exact LF bytes so the stored digest also matches on Windows.
    partial.write_bytes(text.encode("utf-8"))
    partial_sidecar.write_bytes((json.dumps(provenance, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    partial_sidecar.replace(sidecar)
    partial.replace(output)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--benchmark-json", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-mib", type=int, default=0, help="Preview only; runtime uses its actual available budget")
    args = parser.parse_args()
    if args.budget_mib < 0:
        parser.error("budget must be nonnegative")
    result = generate_profile(args.model, args.trace, args.out, args.benchmark_json, args.budget_mib * 1024**2)
    print(json.dumps({"profile": str(args.out.resolve()), "model_sha256": result["model_identity"]["model_sha256"],
                      "model_hash_s": result["model_identity"]["model_hash_s"],
                      "source_fingerprint": result["source_fingerprint"],
                      "plan": {k: v for k, v in result["preview_plan"].items() if k != "expert_ids"},
                      "scope": result["scope"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
