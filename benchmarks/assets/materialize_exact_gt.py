#!/usr/bin/env python3
"""Materialize frozen V2 exact top-10 ground truth without DB or K-Means work."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from contracts import ContractError, load_json, require, sha256_file


def obj(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing JSON: {path}")
    value = load_json(path)
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def exact_index(metric: str, dimension: int) -> faiss.Index:
    if metric == "ip":
        return faiss.IndexFlatIP(dimension)
    if metric == "l2":
        return faiss.IndexFlatL2(dimension)
    raise ContractError(f"unsupported frozen metric: {metric}")


def recall_at_10(exact: np.ndarray, embedded: np.ndarray) -> list[float]:
    return [len(set(map(int, e)) & set(map(int, r))) / 10.0 for e, r in zip(exact, embedded)]


def apply_frozen_normalization(values: np.ndarray, enabled: bool, label: str) -> dict[str, float | bool]:
    norms = np.linalg.norm(values, axis=1)
    require(np.all(np.isfinite(norms)), f"non-finite {label} vector norm")
    result: dict[str, float | bool] = {"l2_normalization_applied": bool(enabled), "raw_norm_min": float(np.min(norms)), "raw_norm_max": float(np.max(norms))}
    if enabled:
        require(np.all(norms > 0.0), f"zero-norm {label} vector violates normalized=true contract")
        values /= norms[:, None]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exact-gt-plan", type=Path, required=True)
    parser.add_argument("--zero-norm-semantics-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    try:
        require(args.cpu_threads > 0, "cpu-threads must be positive")
        require(not args.output_dir.exists() and args.output_dir.parent.is_dir(), "output-dir must be fresh and parent must exist")
        plan = obj(args.exact_gt_plan)
        require(plan.get("status") == "V2_EXACT_GT_MATERIALIZATION_PLAN_FROZEN_NOT_EXECUTION_AUTHORIZED", "wrong frozen exact-GT plan")
        require(plan.get("scientific_units_executed") == 0 and plan.get("seekdb_qps_units_executed") == 0 and plan.get("rq3_touched") is False, "plan must not contain execution")
        jobs = plan.get("jobs")
        require(isinstance(jobs, list) and len(jobs) == 10, "expected 10 frozen exact-GT jobs")
        zero_audit = obj(args.zero_norm_semantics_audit)
        require(zero_audit.get("status") == "V2_ZERO_NORM_SEMANTICS_AUDITED_PREEXECUTION_TERMINALS_FROZEN_NO_PHYSICAL_QPS_AUTHORIZED", "wrong V2 zero-norm semantics audit")
        require(zero_audit.get("scientific_units_executed") == 0 and zero_audit.get("seekdb_qps_units_executed") == 0 and zero_audit.get("rq3_touched") is False, "zero-norm audit must remain read-only")
        audited_jobs = zero_audit.get("jobs")
        require(isinstance(audited_jobs, list) and len(audited_jobs) == 10, "zero-norm audit job denominator drift")
        audited_by_id = {str(row.get("job_id")): row for row in audited_jobs if isinstance(row, dict) and isinstance(row.get("job_id"), str)}
        require(set(audited_by_id) == {str(job.get("job_id")) for job in jobs}, "zero-norm audit job identities drift")
        faiss.omp_set_num_threads(args.cpu_threads)
        args.output_dir.mkdir()
        records: list[dict[str, Any]] = []
        materialized_jobs = 0
        preexecution_terminal_jobs = 0
        for job in jobs:
            started = time.monotonic()
            require(job.get("status") == "FROZEN_NOT_MATERIALIZED", f"job state drift: {job.get('job_id')}")
            audit_row = audited_by_id[str(job["job_id"])]
            audit_status = audit_row.get("semantic_disposition")
            require(audit_status in {"EXACT_GT_MATERIALIZATION_ELIGIBLE", "DETERMINISTICALLY_INFEASIBLE_ANGULAR_ZERO_NORM_PREEXECUTION"}, f"unexpected zero-norm audit status: {job.get('job_id')}")
            if audit_status == "DETERMINISTICALLY_INFEASIBLE_ANGULAR_ZERO_NORM_PREEXECUTION":
                terminal = {"job_id": job["job_id"], "status": "PREEXECUTION_DETERMINISTICALLY_INFEASIBLE_ANGULAR_ZERO_NORM", "ground_truth_path": None, "metric": job["metric"], "normalized": job["normalized"], "imported_train_prefix_rows": job["imported_train_prefix_rows"], "query_selection": job["query_selection"], "zero_norm_train_rows": audit_row["zero_norm_train_rows"], "zero_norm_query_rows": audit_row["zero_norm_query_rows"], "workload_ids": audit_row["workload_ids"], "physical_qps_execution": "PROHIBITED", "candidate_failure_interpretation": "PROHIBITED"}
                records.append(terminal)
                preexecution_terminal_jobs += 1
                print(json.dumps({"event": "exact_gt_job_preexecution_terminal", "job_id": job["job_id"], "completed": len(records), "denominator": len(jobs), "scientific_units_executed": 0, "seekdb_qps_units_executed": 0, "rq3_touched": False}, ensure_ascii=False, sort_keys=True), flush=True)
                continue
            source = Path(str(job.get("source_path", "")))
            require(source.is_file() and not source.is_symlink(), f"unsafe/missing source: {source}")
            require(sha256_file(source) == job.get("source_sha256"), f"source identity drift: {source}")
            n, d = int(job["imported_train_prefix_rows"]), int(job["d"])
            query = job.get("query_selection")
            target = job.get("exact_target")
            require(query == {"source": "test", "indices": "0:1000", "query_rows": 1000, "calibration_indices": "0:500", "evaluation_indices": "500:1000"}, f"query contract drift: {job.get('job_id')}")
            require(isinstance(target, dict) and target.get("top_k") == 10 and target.get("scope") == "EXACT_TOP10_OVER_FROZEN_IMPORTED_TRAIN_PREFIX", f"exact target drift: {job.get('job_id')}")
            with h5py.File(source, "r") as handle:
                require("train" in handle and "test" in handle and "neighbors" in handle, f"HDF5 arrays missing: {source}")
                train = np.asarray(handle["train"][:n], dtype=np.float32, order="C")
                queries = np.asarray(handle["test"][:1000], dtype=np.float32, order="C")
                embedded = np.asarray(handle["neighbors"][:1000, :10], dtype=np.int64, order="C")
            require(train.shape == (n, d) and queries.shape == (1000, d) and embedded.shape == (1000, 10), f"HDF5 shape drift: {job.get('job_id')}")
            train_preprocessing = apply_frozen_normalization(train, bool(job["normalized"]), "train")
            query_preprocessing = apply_frozen_normalization(queries, bool(job["normalized"]), "query")
            index = exact_index(str(job["metric"]), d)
            index.add(train)
            distances, neighbor_ids = index.search(queries, 10)
            require(neighbor_ids.shape == (1000, 10) and distances.shape == (1000, 10) and np.all(neighbor_ids >= 0), f"exact search failed: {job.get('job_id')}")
            per_query_recall = recall_at_10(neighbor_ids, embedded)
            gt_path = args.output_dir / f"{job['job_id']}_exact_top10.npz"
            np.savez_compressed(gt_path, neighbor_ids=neighbor_ids.astype(np.int64), distances=distances.astype(np.float32), query_indices=np.arange(1000, dtype=np.int64))
            metadata = {"job_id": job["job_id"], "status": "MATERIALIZED_EXACT_TOP10_NORMALIZATION_CONTRACT_ENFORCED", "ground_truth_path": str(gt_path.resolve()), "ground_truth_sha256": sha256_file(gt_path), "ground_truth_shape": [1000, 10], "source_path": str(source), "source_sha256": job["source_sha256"], "scope": target["scope"], "metric": job["metric"], "normalized": job["normalized"], "effective_preprocessing": {"train": train_preprocessing, "query": query_preprocessing}, "imported_train_prefix_rows": n, "query_selection": query, "embedded_neighbors_semantic_validation": {"per_query_recall_at_10": per_query_recall, "mean_recall_at_10": float(np.mean(per_query_recall)), "minimum_recall_at_10": float(np.min(per_query_recall)), "interpretation": "reported-only; embedded neighbors may refer to a larger source corpus; materialized exact prefix GT is the V2 reference"}}
            (args.output_dir / f"{job['job_id']}_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            records.append(metadata)
            materialized_jobs += 1
            print(json.dumps({"event": "exact_gt_job_materialized", "job_id": job["job_id"], "completed": len(records), "denominator": len(jobs), "elapsed_seconds": round(time.monotonic() - started, 3), "scientific_units_executed": 0, "seekdb_qps_units_executed": 0, "rq3_touched": False}, ensure_ascii=False, sort_keys=True), flush=True)
            del index, train, queries, embedded, distances, neighbor_ids
        require(len(records) == len(jobs) and materialized_jobs + preexecution_terminal_jobs == len(jobs), "exact-GT terminal accounting drift")
        report = {"artifact_id": "RQ2-DB-AWARE-SELECTOR-V2-EXACT-GT-MATERIALIZATION-V1", "artifact_revision": "V2_NORMALIZATION_CONTRACT_AND_ZERO_NORM_AUDIT_FIX", "supersedes": "any V2 exact-GT artifact produced before applying frozen normalized=True preprocessing or without the V2 zero-norm semantics audit", "status": "V2_EXACT_GT_ELIGIBLE_SCOPES_MATERIALIZED_ZERO_NORM_TERMINALS_RETAINED_CENTROID_BINDING_PENDING_NO_PHYSICAL_QPS_AUTHORIZED", "exact_gt_plan": str(args.exact_gt_plan), "zero_norm_semantics_audit": str(args.zero_norm_semantics_audit), "jobs": records, "workload_to_job": plan["workload_to_job"], "summary": {"exact_gt_jobs_resolved": len(records), "exact_gt_jobs_materialized": materialized_jobs, "preexecution_terminal_jobs": preexecution_terminal_jobs, "workloads_covered": 12, "query_rows_per_scope": 1000, "top_k": 10}, "next_gates": ["freeze centroid provenance for every workload-candidate artifact", "evaluate selector coverage and abstention with zero-norm preexecution terminals retained", "explicit separate authorization before any physical carrier or QPS work", "V1 PROVISIONAL_14_ACTION_DB_SELECTOR_RQ3_FREEZE_BLOCKED remains immutable"], "scientific_units_executed": 0, "seekdb_qps_units_executed": 0, "rq3_touched": False}
        (args.output_dir / "RQ2_DB_SELECTOR_V2_EXACT_GT_MATERIALIZATION.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "output": str(args.output_dir), "exact_gt_jobs_materialized": materialized_jobs, "preexecution_terminal_jobs": preexecution_terminal_jobs, "workloads_covered": 12, "scientific_units_executed": 0, "seekdb_qps_units_executed": 0, "rq3_touched": False}, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "REFUSED_OR_FAILED", "error": f"{type(exc).__name__}: {exc}", "scientific_units_executed": 0, "seekdb_qps_units_executed": 0, "rq3_touched": False}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
