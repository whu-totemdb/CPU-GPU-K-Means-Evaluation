#!/usr/bin/env python3
"""Evaluate exported K-Means centroids on the exact deterministic task slice.

Version 2 separates two different notions of quality:

* an exhaustive assignment objective over the *same deterministic training
  slice* supplied to the candidate; this is available without query ground
  truth, and
* the legacy external-centroid IVF retrieval recall, which is attempted only
  for registry entries whose ground truth is explicitly ``VALIDATED``.

For L2 tasks the objective is the ordinary squared-Euclidean SSE.  For IP
tasks the frozen objective is squared Euclidean distance after L2-normalising
both vectors and centroids (angular/chord SSE, ``2 - 2 cos(theta)``).  It is
therefore never mislabeled as raw inner-product K-Means inertia.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from record_io import append_jsonl, read_jsonl, result_key
from result_contract import file_identity, file_sha256, stable_hash, utc_now
from unified_contract import data_slice_id as contract_data_slice_id


ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = ROOT / "data_registry.json"
QUALITY_VERSION = "quality_eval_v2"
OBJECTIVE_PROTOCOL_VERSION = "exact_task_slice_assignment_objective_v2"
IVF_PROTOCOL_VERSION = "faiss_ivf_external_centroids_v2"
PROTOCOL_ENVELOPE_ID = "faiss_ivf_external_centroids_v2"
REGISTRY_CONTRACT_VERSION = "kmeans_data_registry_schema_v1"
REGISTRY_EVIDENCE_CONTRACT_VERSION = "dataset_registry_evidence_v2"
REGISTRY_REQUIRED_DATASET_FIELDS = ("filename", "metric", "normalize", "expected_d")
GPU_ADMISSION_CONTRACT_VERSION = "quality_eval_gpu_admission_v1"
GPU_ADMISSION_MIN_FREE_FRACTION = 0.70
GPU_ADMISSION_MAX_UTIL_PERCENT = 10


def load_registry(path: Path) -> dict[str, Any]:
    """Load one registry instance while keeping its content out of protocol identity."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"registry is unreadable: {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("datasets"), dict):
        raise ValueError("registry must be an object containing a datasets object")
    declared = payload.get("registry_contract_version", REGISTRY_CONTRACT_VERSION)
    if declared != REGISTRY_CONTRACT_VERSION:
        raise ValueError(
            f"registry contract {declared!r} is unsupported; expected {REGISTRY_CONTRACT_VERSION!r}"
        )
    datasets = payload["datasets"]
    for dataset_id, spec in datasets.items():
        if not isinstance(dataset_id, str) or not dataset_id or not isinstance(spec, dict):
            raise ValueError("registry dataset ids must be non-empty strings with object entries")
        missing = [field for field in REGISTRY_REQUIRED_DATASET_FIELDS if field not in spec]
        if missing:
            raise ValueError(f"registry dataset {dataset_id!r} lacks required fields {missing}")
        if spec.get("metric") not in {"l2", "ip"}:
            raise ValueError(f"registry dataset {dataset_id!r} has unsupported metric {spec.get('metric')!r}")
        try:
            expected_d = int(spec["expected_d"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"registry dataset {dataset_id!r} expected_d must be an integer") from exc
        if expected_d <= 0:
            raise ValueError(f"registry dataset {dataset_id!r} expected_d must be positive")
        if not isinstance(spec.get("normalize"), bool):
            raise ValueError(f"registry dataset {dataset_id!r} normalize must be boolean")
    return {
        "path": path,
        "contract_version": REGISTRY_CONTRACT_VERSION,
        "declared_contract_version": payload.get("registry_contract_version"),
        "datasets": datasets,
        "artifact": {
            **file_identity(path),
            "sha256": file_sha256(path),
        },
    }


def registry_evidence_for_row(row: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    dataset = str(row.get("dataset"))
    spec = registry["datasets"].get(dataset)
    if isinstance(spec, dict):
        state = {
            "sha256_status": spec.get("sha256_status") or "UNDECLARED",
            "ground_truth_status": spec.get("ground_truth_status") or "UNVERIFIED",
            "license_status": spec.get("license_status") or "UNDECLARED",
        }
    else:
        absent_state = "NOT_APPLICABLE" if dataset == "synthetic" else "MISSING"
        state = {
            "sha256_status": absent_state,
            "ground_truth_status": absent_state,
            "license_status": absent_state,
        }
    return {
        "contract_version": REGISTRY_EVIDENCE_CONTRACT_VERSION,
        "registry_artifact": registry["artifact"],
        "dataset_id": dataset,
        "dataset_entry_sha256": stable_hash(spec) if isinstance(spec, dict) else None,
        "dataset_entry_state": state,
    }


def validate_recorded_registry_evidence(
    row: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    """Hard-link candidate and quality stages by immutable registry hashes."""
    recorded = row.get("registry_evidence")
    if not isinstance(recorded, dict):
        raise ValueError("candidate row has no registry_evidence; cross-stage registry identity is unknown")
    if recorded.get("contract_version") != REGISTRY_EVIDENCE_CONTRACT_VERSION:
        raise ValueError(
            "candidate row registry_evidence contract mismatch: "
            f"{recorded.get('contract_version')!r}"
        )
    recorded_artifact = recorded.get("registry_artifact")
    current_artifact = current.get("registry_artifact")
    recorded_sha = recorded_artifact.get("sha256") if isinstance(recorded_artifact, dict) else None
    current_sha = current_artifact.get("sha256") if isinstance(current_artifact, dict) else None
    mismatches = {}
    if recorded_sha != current_sha:
        mismatches["registry_artifact.sha256"] = {
            "candidate": recorded_sha, "quality": current_sha,
        }
    if recorded.get("dataset_entry_sha256") != current.get("dataset_entry_sha256"):
        mismatches["dataset_entry_sha256"] = {
            "candidate": recorded.get("dataset_entry_sha256"),
            "quality": current.get("dataset_entry_sha256"),
        }
    if str(recorded.get("dataset_id")) != str(current.get("dataset_id")):
        mismatches["dataset_id"] = {
            "candidate": recorded.get("dataset_id"), "quality": current.get("dataset_id"),
        }
    if mismatches:
        raise ValueError(f"candidate/quality registry evidence mismatch: {mismatches}")
    return {
        "status": "VERIFIED",
        "hard_keys": ["registry_artifact.sha256", "dataset_entry_sha256", "dataset_id"],
    }


def _metric(row: dict[str, Any], registry: dict[str, Any]) -> str:
    observed = row.get("metric")
    if observed in {"l2", "ip"}:
        return str(observed)
    if row.get("dataset") == "synthetic":
        return "l2"
    dataset = str(row.get("dataset"))
    datasets = registry["datasets"]
    if dataset not in datasets:
        raise ValueError(f"dataset {dataset!r} is absent from data_registry.json")
    metric = str(datasets[dataset].get("metric"))
    if metric not in {"l2", "ip"}:
        raise ValueError(f"unsupported metric {metric!r} for {dataset}")
    return metric


def objective_name(metric: str) -> str:
    """Return the frozen, non-ambiguous assignment-objective name."""
    if metric == "l2":
        return "squared_euclidean_sse"
    if metric == "ip":
        return "angular_chord_sse_on_l2_normalized_vectors"
    raise ValueError(f"unsupported metric {metric!r}")


def sampling_provenance(row: dict[str, Any]) -> dict[str, Any]:
    """Extract the recorded sampling rule from Python or native-runner rows."""
    fingerprint = row.get("data_fingerprint")
    if not isinstance(fingerprint, dict):
        return {}
    direct = fingerprint.get("sampling")
    if isinstance(direct, dict):
        return dict(direct)
    prepared = fingerprint.get("prepared")
    if isinstance(prepared, dict) and isinstance(prepared.get("sampling"), dict):
        return dict(prepared["sampling"])
    return {}


def expected_sampling(*, total: int, n: int, seed: int) -> dict[str, Any]:
    return {
        "method": "all_rows" if n == total else "sorted_choice_without_replacement",
        "seed": int(seed),
        "requested_rows": int(n),
        "source_train_rows": int(total),
    }


def seed_provenance(row: dict[str, Any]) -> dict[str, Any]:
    """Return decoupled run/data/initialisation seeds with legacy fallbacks."""
    run_seed = int(row["seed"])
    data_seed_raw = row.get("data_seed")
    init_seed_raw = row.get("init_seed")
    return {
        "run_seed": run_seed,
        "data_seed": int(run_seed if data_seed_raw is None else data_seed_raw),
        "init_seed": int(run_seed if init_seed_raw is None else init_seed_raw),
        "repeat_id": row.get("repeat_id"),
        "order_block": row.get("order_block"),
        "order_index": row.get("order_index"),
        "warmup_state": row.get("warmup_state"),
    }


def validate_sampling_provenance(
    recorded: dict[str, Any], *, total: int, n: int, seed: int, synthetic: bool = False
) -> dict[str, Any]:
    """Refuse to guess a real-data slice when its recorded provenance differs.

    Historical synthetic rows record a generator rather than a sampling
    object; their complete ``n x d`` stream is nevertheless determined by the
    row seed and the frozen generator name.
    """
    if synthetic:
        return {"method": "synthetic_generator", "seed": int(seed), "requested_rows": int(n)}
    wanted = expected_sampling(total=total, n=n, seed=seed)
    if not recorded:
        raise ValueError("data_fingerprint has no sampling provenance; exact task slice cannot be established")
    mismatches = {
        key: {"recorded": recorded.get(key), "expected": value}
        for key, value in wanted.items()
        if recorded.get(key) != value
    }
    if mismatches:
        raise ValueError(f"recorded sampling provenance does not match the task row: {mismatches}")
    return wanted


def _normalise_rows(matrix: Any) -> Any:
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, 1e-12)
    return matrix


def _normalise_angular_rows_or_raise(matrix: Any, *, object_name: str) -> Any:
    """L2-normalise angular-objective rows, rejecting undefined zero vectors."""
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1)
    if bool(np.any(norms <= 1e-12)):
        raise ValueError(f"angular objective is undefined for a zero-norm {object_name}")
    return _normalise_rows(matrix)


def _sampled_indices(total: int, n: int, seed: int) -> Any:
    import numpy as np

    if n <= 0 or n > total:
        raise ValueError(f"n must be in [1, {total}], got {n}")
    if n == total:
        return np.arange(total, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(total, n, replace=False))


def _iter_synthetic(*, n: int, d: int, seed: int, batch_rows: int) -> Iterator[Any]:
    import numpy as np

    rng = np.random.default_rng(seed)
    for start in range(0, n, batch_rows):
        yield rng.standard_normal((min(batch_rows, n - start), d), dtype=np.float32)


def _iter_hdf5_slice(
    *, source: Path, indices: Any, normalize: bool, batch_rows: int
) -> Iterator[Any]:
    import h5py
    import numpy as np

    with h5py.File(source, "r") as handle:
        train = handle["train"]
        for start in range(0, len(indices), batch_rows):
            rows = indices[start:start + batch_rows]
            block = np.ascontiguousarray(train[rows], dtype=np.float32)
            if normalize:
                _normalise_rows(block)
            yield block


def _centroids_for_objective(centroids: Any, metric: str) -> Any:
    import numpy as np

    centres = np.ascontiguousarray(centroids, dtype=np.float32).copy()
    if not np.isfinite(centres).all():
        raise ValueError("centroids contain NaN or infinity")
    if metric == "ip":
        _normalise_angular_rows_or_raise(centres, object_name="centroid")
    return centres


def _minimum_squared_distances_numpy(block: Any, centroids: Any, *, centroid_batch_rows: int) -> Any:
    """Exact exhaustive assignment with a bounded ``rows x centroid_batch`` matrix."""
    import numpy as np

    row_norms = np.einsum("ij,ij->i", block, block, dtype=np.float64)
    best = np.full(len(block), np.inf, dtype=np.float64)
    for start in range(0, len(centroids), centroid_batch_rows):
        centres = centroids[start:start + centroid_batch_rows]
        centre_norms = np.einsum("ij,ij->i", centres, centres, dtype=np.float64)
        # The matmul follows the deployed FP32 data/centroid representation;
        # norms and final SSE accumulation use FP64 for numerical stability.
        dots = block @ centres.T
        distances = row_norms[:, None] + centre_norms[None, :] - 2.0 * dots
        np.maximum(distances, 0.0, out=distances)
        best = np.minimum(best, distances.min(axis=1))
    return best


def _exact_assignment_costs(
    block: Any, centroids: Any, *, metric: str, backend: str, centroid_batch_rows: int,
    faiss_index: Any | None = None,
) -> Any:
    """Return exact nearest-centroid costs using the frozen backend.

    Both ``faiss_flat`` and ``faiss_gpu_flat`` are exhaustive (no
    IVF/PQ/approximation). ``numpy_exact`` remains available as an independent,
    dependency-light cross-check on small fixtures.
    """
    import numpy as np

    if backend == "numpy_exact":
        return _minimum_squared_distances_numpy(
            block, centroids, centroid_batch_rows=centroid_batch_rows
        )
    if backend not in {"faiss_flat", "faiss_gpu_flat"}:
        raise ValueError(f"unsupported exact objective backend {backend!r}")
    if faiss_index is None:
        raise ValueError("faiss_flat backend requires a prebuilt exact index")
    if metric == "ip":
        similarities, _ = faiss_index.search(np.ascontiguousarray(block, dtype=np.float32), 1)
        return np.maximum(2.0 - 2.0 * similarities[:, 0].astype(np.float64), 0.0)
    distances, _ = faiss_index.search(np.ascontiguousarray(block, dtype=np.float32), 1)
    return np.maximum(distances[:, 0].astype(np.float64), 0.0)


def _build_exact_assignment_index(
    centroids: Any, metric: str, backend: str, *, gpu_resources: Any | None = None
) -> Any | None:
    if backend == "numpy_exact":
        return None
    if backend not in {"faiss_flat", "faiss_gpu_flat"}:
        raise ValueError(f"unsupported exact objective backend {backend!r}")
    import faiss
    import numpy as np

    d = int(centroids.shape[1])
    index = faiss.IndexFlatIP(d) if metric == "ip" else faiss.IndexFlatL2(d)
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    if backend == "faiss_gpu_flat":
        if gpu_resources is None:
            raise ValueError("faiss_gpu_flat requires admitted FAISS GPU resources")
        # CUDA_VISIBLE_DEVICES has already isolated the admitted physical GPU;
        # FAISS must therefore use logical device zero exclusively.
        index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
    return index


def _prepared_output_sha256(row: dict[str, Any]) -> str | None:
    fingerprint = row.get("data_fingerprint")
    if not isinstance(fingerprint, dict):
        return None
    prepared = fingerprint.get("prepared")
    value = prepared.get("output_sha256") if isinstance(prepared, dict) else None
    return str(value) if isinstance(value, str) and len(value) == 64 else None


def _recorded_source_identity(row: dict[str, Any]) -> dict[str, Any] | None:
    fingerprint = row.get("data_fingerprint")
    if not isinstance(fingerprint, dict):
        return None
    direct = fingerprint.get("observed_file_identity")
    if isinstance(direct, dict):
        return direct
    prepared = fingerprint.get("prepared")
    nested = prepared.get("input_identity") if isinstance(prepared, dict) else None
    return nested if isinstance(nested, dict) else None


def _recorded_normalization(row: dict[str, Any]) -> bool | None:
    fingerprint = row.get("data_fingerprint")
    if not isinstance(fingerprint, dict):
        return None
    prepared = fingerprint.get("prepared")
    if isinstance(prepared, dict) and prepared.get("normalized") is not None:
        return bool(prepared["normalized"])
    for key in ("normalization_applied", "normalized"):
        if fingerprint.get(key) is not None:
            return bool(fingerprint[key])
    return None


def _source_artifact(source: Path | None, *, compute_sha256: bool, hash_cache: dict[Path, str]) -> dict[str, Any]:
    if source is None:
        return {"kind": "synthetic_generator", "sha256": None}
    artifact: dict[str, Any] = {"kind": "hdf5", **file_identity(source), "sha256": None}
    if compute_sha256:
        if source not in hash_cache:
            hash_cache[source] = file_sha256(source)
        artifact["sha256"] = hash_cache[source]
    return artifact


def _slice_and_shape(
    row: dict[str, Any], data_root: Path, registry: dict[str, Any], *, batch_rows: int
) -> tuple[Iterator[Any], dict[str, Any], Path | None]:
    n, d = int(row["n"]), int(row["d"])
    seeds = seed_provenance(row)
    data_seed = int(seeds["data_seed"])
    if row.get("dataset") == "synthetic":
        if row.get("metric") not in (None, "l2"):
            raise ValueError(f"synthetic task metric must be 'l2', observed {row.get('metric')!r}")
        fingerprint = row.get("data_fingerprint", {})
        generator = None
        spec = None
        prepared = None
        if isinstance(fingerprint, dict):
            spec = fingerprint.get("generator_spec")
            prepared = fingerprint.get("prepared")
            generator = spec.get("generator") if isinstance(spec, dict) else fingerprint.get("generator")
            if generator is None and isinstance(prepared, dict):
                generator = prepared.get("generator")
        expected_generator = "numpy.default_rng.standard_normal.float32.v1"
        if generator != expected_generator:
            raise ValueError(
                "exact synthetic slice requires recorded generator provenance "
                f"{expected_generator!r}, observed {generator!r}"
            )
        recorded_synthetic = spec if isinstance(spec, dict) else prepared if isinstance(prepared, dict) else {}
        synthetic_mismatches = {
            key: {"recorded": recorded_synthetic.get(key), "expected": value}
            for key, value in {"seed": data_seed, "n": n, "d": d}.items()
            if recorded_synthetic.get(key) != value
        }
        if synthetic_mismatches:
            raise ValueError(
                f"recorded synthetic task provenance disagrees with data_seed/n/d: {synthetic_mismatches}"
            )
        sampling = validate_sampling_provenance({}, total=n, n=n, seed=data_seed, synthetic=True)
        return _iter_synthetic(n=n, d=d, seed=data_seed, batch_rows=batch_rows), {
            "sampling": sampling,
            "seed_provenance": seeds,
            "normalization_applied": False,
            "generator": "numpy.default_rng.standard_normal.float32.v1",
            "source_train_rows": n,
            "registry_contract_version": registry["contract_version"],
            "registry_file_sha256": registry["artifact"]["sha256"],
            "registry_dataset_entry_sha256": None,
        }, None

    dataset = str(row.get("dataset"))
    datasets = registry["datasets"]
    if dataset not in datasets:
        raise ValueError(f"dataset {dataset!r} is absent from data_registry.json")
    spec = datasets[dataset]
    import h5py

    source = data_root / str(spec["filename"])
    if not source.is_file():
        raise FileNotFoundError(source)
    with h5py.File(source, "r") as handle:
        if "train" not in handle:
            raise ValueError(f"{source} has no train array")
        total, observed_d = map(int, handle["train"].shape)
    if observed_d != d:
        raise ValueError(f"row d={d}, but {source.name} train d={observed_d}")
    expected_d = int(spec["expected_d"])
    if observed_d != expected_d:
        raise ValueError(f"registry expected d={expected_d}, but {source.name} has d={observed_d}")
    registry_metric = str(spec.get("metric"))
    if row.get("metric") is not None and row.get("metric") != registry_metric:
        raise ValueError(
            f"row metric {row.get('metric')!r} disagrees with registry metric {registry_metric!r}"
        )
    current_identity = file_identity(source)
    recorded_identity = _recorded_source_identity(row)
    if recorded_identity is None:
        raise ValueError("data_fingerprint has no source file identity; exact source artifact cannot be established")
    identity_mismatches = {
        key: {"recorded": recorded_identity.get(key), "current": current_identity.get(key)}
        for key in ("size_bytes", "mtime_ns")
        if recorded_identity.get(key) != current_identity.get(key)
    }
    if identity_mismatches:
        raise ValueError(f"dataset artifact identity changed since candidate execution: {identity_mismatches}")
    sampling = validate_sampling_provenance(
        sampling_provenance(row), total=total, n=n, seed=data_seed, synthetic=False
    )
    indices = _sampled_indices(total, n, data_seed)
    normalize = bool(spec.get("normalize"))
    recorded_normalize = _recorded_normalization(row)
    if recorded_normalize is None or recorded_normalize != normalize:
        raise ValueError(
            "recorded normalization provenance is missing or disagrees with the registry: "
            f"recorded={recorded_normalize}, registry={normalize}"
        )
    return _iter_hdf5_slice(
        source=source, indices=indices, normalize=normalize, batch_rows=batch_rows
    ), {
        "sampling": sampling,
        "seed_provenance": seeds,
        "normalization_applied": normalize,
        "sampled_indices_sha256": hashlib.sha256(indices.tobytes(order="C")).hexdigest(),
        "source_train_rows": total,
        "registry_spec_sha256": stable_hash(spec),
        "registry_contract_version": registry["contract_version"],
        "registry_file_sha256": registry["artifact"]["sha256"],
        "registry_dataset_entry_sha256": stable_hash(spec),
        "run_recorded_source_identity": recorded_identity,
        "evaluation_source_identity": current_identity,
    }, source


def _evaluate_objective(
    row: dict[str, Any], data_root: Path, centroids: Any, registry: dict[str, Any], *, data_batch_rows: int,
    centroid_batch_rows: int, objective_backend: str, compute_source_sha256: bool,
    hash_cache: dict[Path, str], gpu_resources: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    import numpy as np

    started = time.perf_counter()
    metric = _metric(row, registry)
    centres = _centroids_for_objective(centroids, metric)
    exact_index = _build_exact_assignment_index(
        centres, metric, objective_backend, gpu_resources=gpu_resources
    )
    iterator, slice_meta, source = _slice_and_shape(
        row, data_root, registry, batch_rows=data_batch_rows
    )
    source_hash_started = time.perf_counter()
    source_artifact = _source_artifact(
        source, compute_sha256=compute_source_sha256, hash_cache=hash_cache
    )
    source_hash_seconds = time.perf_counter() - source_hash_started
    digest = hashlib.sha256()
    count = 0
    total_cost = 0.0
    max_cost = 0.0
    slice_io_seconds = 0.0
    kernel_seconds = 0.0
    iterator = iter(iterator)
    while True:
        io_started = time.perf_counter()
        try:
            block = next(iterator)
        except StopIteration:
            break
        slice_io_seconds += time.perf_counter() - io_started
        if block.ndim != 2 or int(block.shape[1]) != int(row["d"]):
            raise ValueError(f"reconstructed slice block has invalid shape {block.shape}")
        if not np.isfinite(block).all():
            raise ValueError("reconstructed task slice contains NaN or infinity")
        digest.update(np.ascontiguousarray(block, dtype=np.float32).tobytes(order="C"))
        work = np.ascontiguousarray(block, dtype=np.float32)
        if metric == "ip":
            # Registry-normalised inputs are normalised once more by definition
            # of the angular objective, avoiding any hidden reliance on norms.
            # Cosine/angular distance is undefined for a zero vector, so the
            # canonical evaluator fails closed instead of assigning it the
            # artificial constant chord cost induced by epsilon division.
            _normalise_angular_rows_or_raise(work, object_name="input vector")
        kernel_started = time.perf_counter()
        minimum = _exact_assignment_costs(
            work, centres, metric=metric, backend=objective_backend,
            centroid_batch_rows=centroid_batch_rows, faiss_index=exact_index,
        )
        kernel_seconds += time.perf_counter() - kernel_started
        total_cost += float(minimum.sum(dtype=np.float64))
        max_cost = max(max_cost, float(minimum.max()))
        count += len(work)
    expected_n = int(row["n"])
    if count != expected_n:
        raise ValueError(f"reconstructed {count} rows, expected {expected_n}")
    slice_sha256 = digest.hexdigest()
    prepared_sha256 = _prepared_output_sha256(row)
    if prepared_sha256 is not None and prepared_sha256 != slice_sha256:
        raise ValueError(
            "reconstructed task-slice bytes disagree with the native prepared-input artifact: "
            f"{slice_sha256} != {prepared_sha256}"
        )
    objective = {
        "status": "OK",
        "name": objective_name(metric),
        "metric": metric,
        "definition": (
            "sum_x min_c ||x-c||_2^2 over the exact deterministic candidate input slice"
            if metric == "l2" else
            "sum_x min_c ||normalize(x)-normalize(c)||_2^2 = 2-2cos(theta), over the exact deterministic candidate input slice"
        ),
        "scope": "exact_deterministic_candidate_training_slice",
        "ground_truth_independent": True,
        "n_evaluated": count,
        "sum_assignment_cost": total_cost,
        "mean_assignment_cost": total_cost / max(1, count),
        "max_assignment_cost": max_cost,
        "data_dtype": "float32",
        "centroid_dtype": "float32",
        "distance_dot_dtype": "float32",
        "norm_and_sse_accumulator_dtype": "float64",
        "exact_assignment_backend": objective_backend,
        "centroids_l2_normalized_for_objective": metric == "ip",
        "vectors_l2_normalized_for_objective": metric == "ip",
    }
    slice_artifact = {
        **slice_meta,
        "source_artifact": source_artifact,
        "reconstructed_slice_sha256": slice_sha256,
        "prepared_input_sha256": prepared_sha256,
        "data_fingerprint_sha256": stable_hash(row.get("data_fingerprint", {})),
        "unified_data_slice_id": contract_data_slice_id(row),
        "slice_identity_sha256": stable_hash({
            "dataset": row.get("dataset"), "n": row.get("n"), "d": row.get("d"),
            "data_seed": seed_provenance(row)["data_seed"],
            "sampling": slice_meta.get("sampling"),
            "normalization": slice_meta.get("normalization_applied"),
            "source_sha256": source_artifact.get("sha256"), "slice_sha256": slice_sha256,
        }),
        "run_seed_provenance_sha256": stable_hash(seed_provenance(row)),
    }
    timing = {
        "source_artifact_hash_seconds": source_hash_seconds,
        "slice_read_generate_normalize_seconds": slice_io_seconds,
        "objective_kernel_seconds": kernel_seconds,
        "objective_total_seconds": time.perf_counter() - started,
    }
    return objective, slice_artifact, timing


def _index_memory_estimate_gib(n: int, d: int) -> float:
    return n * d * 4 * 1.25 / (1024 ** 3)


def _normalise_if_ip(matrix: Any, metric: str) -> None:
    if metric != "ip":
        return
    import faiss

    faiss.normalize_L2(matrix)


def _create_ivf_index(
    centroids: Any, metric: str, backend: str = "faiss_cpu_ivf", *,
    gpu_resources: Any | None = None,
) -> Any:
    import faiss
    import numpy as np

    d, nlist = int(centroids.shape[1]), int(centroids.shape[0])
    centres = np.ascontiguousarray(centroids, dtype=np.float32).copy()
    _normalise_if_ip(centres, metric)
    if metric == "ip":
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    else:
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    quantizer.add(centres)
    index.is_trained = True
    if backend == "faiss_gpu_ivf":
        if gpu_resources is None:
            raise ValueError("faiss_gpu_ivf requires admitted FAISS GPU resources")
        index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)
    elif backend != "faiss_cpu_ivf":
        raise ValueError(f"unsupported IVF backend {backend!r}")
    return index


def _validated_ivf_recall(
    row: dict[str, Any], data_root: Path, centroids: Any, registry: dict[str, Any], *, nprobe_values: list[int],
    topk: int, add_batch_rows: int, query_batch_rows: int, max_index_memory_gib: float,
    ivf_backend: str = "faiss_cpu_ivf", gpu_resources: Any | None = None,
) -> dict[str, Any]:
    dataset = row.get("dataset")
    if dataset == "synthetic":
        return {"status": "NOT_APPLICABLE_SYNTHETIC", "metrics": {}}
    datasets = registry["datasets"]
    if str(dataset) not in datasets:
        return {"status": "ERROR_DATASET_ABSENT_FROM_REGISTRY", "metrics": {}}
    spec = datasets[str(dataset)]
    gt_status = str(spec.get("ground_truth_status", "UNVERIFIED")).upper()
    if gt_status != "VALIDATED":
        return {
            "status": "NOT_RUN_GROUND_TRUTH_NOT_VALIDATED",
            "ground_truth_status": gt_status,
            "ground_truth_reason": spec.get("ground_truth_reason"),
            "metrics": {},
        }
    import h5py
    import numpy as np

    source = data_root / str(spec["filename"])
    with h5py.File(source, "r") as handle:
        missing = [name for name in ("train", "test", "neighbors") if name not in handle]
        if missing:
            return {
                "status": "ERROR_VALIDATED_GROUND_TRUTH_MISSING",
                "error": f"registry says VALIDATED but HDF5 lacks {missing}", "metrics": {},
            }
        train, test, neighbors = handle["train"], handle["test"], handle["neighbors"]
        n, d = map(int, train.shape)
        estimate = _index_memory_estimate_gib(n, d)
        if estimate > max_index_memory_gib:
            return {
                "status": "SKIPPED_RESOURCE_INDEX_MEMORY",
                "index_memory_estimate_gib": estimate,
                "max_index_memory_gib": max_index_memory_gib,
                "metrics": {},
            }
        metric = _metric(row, registry)
        index = _create_ivf_index(
            centroids, metric, ivf_backend, gpu_resources=gpu_resources
        )
        build_started = time.perf_counter()
        for start in range(0, n, add_batch_rows):
            block = np.ascontiguousarray(train[start:start + add_batch_rows], dtype=np.float32)
            _normalise_if_ip(block, metric)
            index.add(block)
        build_seconds = time.perf_counter() - build_started
        metrics: dict[str, float] = {}
        total_queries = int(test.shape[0])
        truth_width = int(neighbors.shape[1]) if len(neighbors.shape) == 2 else 0
        if truth_width < topk:
            return {
                "status": "ERROR_GROUND_TRUTH_WIDTH",
                "error": f"neighbors width {truth_width} < topk {topk}", "metrics": {},
            }
        for nprobe in nprobe_values:
            index.nprobe = int(nprobe)
            hits = 0
            query_seconds = 0.0
            for start in range(0, total_queries, query_batch_rows):
                queries = np.ascontiguousarray(test[start:start + query_batch_rows], dtype=np.float32)
                _normalise_if_ip(queries, metric)
                query_started = time.perf_counter()
                _, found = index.search(queries, topk)
                query_seconds += time.perf_counter() - query_started
                truth = neighbors[start:start + len(queries), :topk]
                for predicted, expected in zip(found, truth):
                    predicted_set = {int(value) for value in predicted if value >= 0}
                    expected_set = {int(value) for value in expected if value >= 0}
                    hits += len(predicted_set & expected_set)
            suffix = f"np{int(nprobe)}"
            metrics[f"recall_at_{topk}_{suffix}"] = hits / max(1, total_queries * topk)
            metrics[f"query_seconds_{suffix}"] = query_seconds
    return {
        "status": "OK",
        "protocol_version": IVF_PROTOCOL_VERSION,
        "backend": ivf_backend,
        "ground_truth_status": "VALIDATED",
        "index_scope": "complete_source_train_array",
        "n_indexed": n,
        "n_query": total_queries,
        "nlist": int(row["k"]),
        "index_memory_estimate_gib": estimate,
        "ivf_add_seconds": build_seconds,
        "metrics": metrics,
    }


def gpu_backend_requested(args: argparse.Namespace) -> bool:
    return (
        str(getattr(args, "objective_backend", "faiss_flat")) == "faiss_gpu_flat"
        or str(getattr(args, "ivf_backend", "faiss_cpu_ivf")) == "faiss_gpu_ivf"
    )


def gpu_execution_protocol(args: argparse.Namespace) -> dict[str, Any]:
    """Return stable GPU rules; the volatile admission observation is row evidence."""
    requested = gpu_backend_requested(args)
    return {
        "contract_version": GPU_ADMISSION_CONTRACT_VERSION,
        "required": requested,
        "physical_gpu_selection": (
            "explicit_cli_argument_recorded_only_in_evidence" if requested else "not_applicable"
        ),
        "visible_logical_gpu_index": 0 if requested else None,
        "cuda_visible_devices_set_after_admission": requested,
        "admission_call_count": 1 if requested else 0,
        "admission_policy": {
            "min_free_memory_fraction": GPU_ADMISSION_MIN_FREE_FRACTION,
            "max_existing_gpu_util_percent": GPU_ADMISSION_MAX_UTIL_PERCENT,
            "refuse_if_compute_process_exists": True,
        },
        "volatile_report_excluded_from_protocol_hash": True,
        "volatile_report_evidence_field": "gpu_admission_report",
    }


def admit_physical_gpu(args: argparse.Namespace) -> dict[str, Any] | None:
    """Admit a physical GPU exactly once, before FAISS or another CUDA module import."""
    if not gpu_backend_requested(args):
        return None
    physical_gpu = getattr(args, "gpu", None)
    if physical_gpu is None:
        raise ValueError("GPU quality backends require an explicit physical --gpu index")
    already_imported = sorted(
        name for name in ("faiss", "torch", "cupy", "cuml") if name in sys.modules
    )
    if already_imported:
        raise RuntimeError(
            "GPU admission must precede every CUDA-capable import; already imported: "
            + ", ".join(already_imported)
        )
    from preflight import admission_report

    report = admission_report(
        int(physical_gpu),
        min_free_fraction=GPU_ADMISSION_MIN_FREE_FRACTION,
        max_existing_util=GPU_ADMISSION_MAX_UTIL_PERCENT,
        refuse_if_busy=True,
    )
    if not report.get("safe_to_start"):
        raise RuntimeError(f"GPU quality admission refused execution: {report.get('blockers')}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu))
    return {
        **report,
        "quality_admission_contract_version": GPU_ADMISSION_CONTRACT_VERSION,
        "physical_gpu_cli_index": int(physical_gpu),
        "cuda_visible_devices": str(int(physical_gpu)),
        "visible_logical_gpu_index": 0,
        "report_sha256": stable_hash(report),
    }


def create_faiss_gpu_resources(required: bool) -> Any | None:
    if not required:
        return None
    import faiss

    if not hasattr(faiss, "StandardGpuResources") or not hasattr(faiss, "index_cpu_to_gpu"):
        raise RuntimeError("installed FAISS build has no GPU resource/transfer API")
    if hasattr(faiss, "get_num_gpus") and int(faiss.get_num_gpus()) < 1:
        raise RuntimeError("FAISS sees no GPU after CUDA_VISIBLE_DEVICES isolation")
    return faiss.StandardGpuResources()


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    code_sha256 = file_sha256(Path(__file__))
    module_versions: dict[str, str | None] = {}
    for module_name in ("numpy", "h5py", "faiss"):
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", None)
            module_versions[module_name] = str(version) if version is not None else None
        except Exception:
            module_versions[module_name] = None
    return {
        "quality_version": QUALITY_VERSION,
        "objective": {
            "version": OBJECTIVE_PROTOCOL_VERSION,
            "l2": "sum min squared Euclidean distance",
            "ip": "sum min squared Euclidean distance after per-vector and per-centroid L2 normalization",
            "assignment": "exact_exhaustive",
            "backend": str(args.objective_backend),
            "data_batch_rows": int(args.data_batch_rows),
            "centroid_batch_rows": int(args.centroid_batch_rows),
            "normalization_epsilon": 1e-12,
        },
        "ivf_recall": {
            "version": IVF_PROTOCOL_VERSION,
            "enabled": not args.skip_ivf_recall,
            "backend": str(getattr(args, "ivf_backend", "faiss_cpu_ivf")),
            "eligibility": "registry_ground_truth_status_equals_VALIDATED",
            "nprobe": [int(value) for value in args.nprobe],
            "topk": int(args.topk),
            "add_batch_rows": int(args.add_batch_rows),
            "query_batch_rows": int(args.query_batch_rows),
            "max_index_memory_gib": float(args.max_index_memory_gib),
        },
        "source_sha256_policy": "computed" if not args.skip_source_sha256 else "not_computed_explicit_cli_override",
        "registry_contract": {
            "contract_version": REGISTRY_EVIDENCE_CONTRACT_VERSION,
            "registry_schema_version": REGISTRY_CONTRACT_VERSION,
            "schema": {
                "top_level_datasets_type": "object",
                "required_dataset_fields": list(REGISTRY_REQUIRED_DATASET_FIELDS),
                "allowed_metrics": ["ip", "l2"],
                "ground_truth_eligibility_field": "ground_truth_status",
                "validated_value": "VALIDATED",
                "evidence_fields": [
                    "registry_artifact", "dataset_id", "dataset_entry_sha256",
                    "dataset_entry_state",
                ],
                "dataset_entry_state_fields": [
                    "sha256_status", "ground_truth_status", "license_status",
                ],
            },
            "instance_content_in_protocol_hash": False,
        },
        "gpu_execution": gpu_execution_protocol(args),
        "code_artifacts": {
            "quality_eval_v2.py_sha256": code_sha256,
        },
        "evaluation_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "module_versions": module_versions,
            "thread_environment": {
                key: os.environ.get(key)
                for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            },
        },
    }


def freeze_protocol_envelope(path: Path, protocol: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create or byte-semantically verify the standalone frozen protocol."""
    protocol_sha256 = stable_hash(protocol)
    envelope = {
        "id": PROTOCOL_ENVELOPE_ID,
        "status": "FROZEN",
        "seekdb_dependency": False,
        "protocol_sha256": protocol_sha256,
        "protocol": protocol,
    }
    if path.exists():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"existing protocol envelope is unreadable: {path}: {exc}") from exc
        if observed != envelope:
            raise ValueError(
                "refusing to overwrite a different frozen quality protocol envelope: "
                f"{path}; expected_protocol_sha256={protocol_sha256}, "
                f"observed_protocol_sha256={observed.get('protocol_sha256') if isinstance(observed, dict) else None}"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        payload = json.dumps(envelope, ensure_ascii=False, indent=2) + "\n"
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    artifact = {
        **file_identity(path),
        "file_sha256": file_sha256(path),
        "payload_sha256": stable_hash(envelope),
    }
    return envelope, artifact


def evaluate_centroids_v2(
    row: dict[str, Any], data_root: Path, registry: dict[str, Any], *, protocol: dict[str, Any],
    compute_source_sha256: bool, hash_cache: dict[Path, str], gpu_resources: Any | None = None,
) -> dict[str, Any]:
    import numpy as np

    centroids_path = Path(str(row.get("centroids_file", "")))
    if not centroids_path.is_file():
        raise FileNotFoundError(f"centroid file missing: {centroids_path}")
    centroid_load_started = time.perf_counter()
    centroids = np.load(centroids_path, mmap_mode="r")
    centroid_load_seconds = time.perf_counter() - centroid_load_started
    expected_shape = (int(row["k"]), int(row["d"]))
    if tuple(centroids.shape) != expected_shape:
        raise ValueError(f"centroids shape {tuple(centroids.shape)}, expected {expected_shape}")
    objective_config = protocol["objective"]
    objective, slice_artifact, timing = _evaluate_objective(
        row, data_root, centroids, registry,
        data_batch_rows=int(objective_config["data_batch_rows"]),
        centroid_batch_rows=int(objective_config["centroid_batch_rows"]),
        objective_backend=str(objective_config["backend"]),
        compute_source_sha256=compute_source_sha256,
        hash_cache=hash_cache,
        gpu_resources=gpu_resources,
    )
    recall_config = protocol["ivf_recall"]
    recall_started = time.perf_counter()
    if recall_config["enabled"]:
        try:
            recall = _validated_ivf_recall(
                row, data_root, centroids, registry,
                nprobe_values=list(recall_config["nprobe"]), topk=int(recall_config["topk"]),
                add_batch_rows=int(recall_config["add_batch_rows"]),
                query_batch_rows=int(recall_config["query_batch_rows"]),
                max_index_memory_gib=float(recall_config["max_index_memory_gib"]),
                ivf_backend=str(recall_config["backend"]),
                gpu_resources=gpu_resources,
            )
        except Exception as exc:
            # Retrieval evidence is nested and optional.  A recall failure must
            # never erase a successfully established assignment objective.
            recall = {"status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "metrics": {}}
    else:
        recall = {"status": "NOT_RUN_DISABLED", "metrics": {}}
    timing["centroid_load_seconds"] = centroid_load_seconds
    timing["ivf_recall_total_seconds"] = time.perf_counter() - recall_started
    return {
        "objective": objective,
        "assignment_sse": objective["sum_assignment_cost"],
        "assignment_cost_mean": objective["mean_assignment_cost"],
        "objective_name": objective["name"],
        "data_slice": slice_artifact,
        "ivf_recall": recall,
        "timing": timing,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact deterministic-slice K-Means objective plus validated-GT IVF recall"
    )
    parser.add_argument("--results-jsonl", type=Path, nargs="+", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path, default=DEFAULT_REGISTRY_PATH,
        help="training or formal registry instance; content is evidence, not protocol identity",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results" / "paper_v2" / "quality" / "quality_results_v2.jsonl",
    )
    parser.add_argument(
        "--protocol-output", type=Path,
        default=ROOT / "results" / "paper_v2" / "quality" / "quality_protocol_v2.json",
        help="standalone create-once/verify-always frozen protocol envelope",
    )
    parser.add_argument("--data-batch-rows", type=int, default=2048)
    parser.add_argument("--centroid-batch-rows", type=int, default=256)
    parser.add_argument(
        "--objective-backend", choices=["faiss_flat", "faiss_gpu_flat", "numpy_exact"],
        default="faiss_flat",
        help="all choices are exhaustive; faiss_gpu_flat requires admitted GPU execution",
    )
    parser.add_argument(
        "--ivf-backend", choices=["faiss_cpu_ivf", "faiss_gpu_ivf"],
        default="faiss_cpu_ivf",
    )
    parser.add_argument(
        "--gpu", type=int,
        help="physical GPU index; required by either GPU backend and remapped to logical device 0",
    )
    parser.add_argument("--nprobe", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--add-batch-rows", type=int, default=8192)
    parser.add_argument("--query-batch-rows", type=int, default=1024)
    parser.add_argument("--max-index-memory-gib", type=float, default=16.0)
    parser.add_argument(
        "--skip-source-sha256", action="store_true",
        help="explicitly trade source artifact integrity for speed; recorded in the protocol",
    )
    parser.add_argument("--skip-ivf-recall", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    positive = [
        args.data_batch_rows, args.centroid_batch_rows, args.topk,
        args.add_batch_rows, args.query_batch_rows,
    ] + list(args.nprobe)
    if any(int(value) <= 0 for value in positive):
        parser.error("batch sizes, nprobe, and topk must be positive")
    if not math.isfinite(args.max_index_memory_gib) or args.max_index_memory_gib <= 0:
        parser.error("--max-index-memory-gib must be a positive finite number")
    if gpu_backend_requested(args) and args.gpu is None:
        parser.error("--gpu is required when faiss_gpu_flat or faiss_gpu_ivf is selected")
    if args.gpu is not None and args.gpu < 0:
        parser.error("--gpu must be non-negative")
    return args


def main() -> int:
    args = _parse_args()
    try:
        gpu_admission_report = admit_physical_gpu(args)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    try:
        registry = load_registry(args.registry)
    except ValueError as exc:
        raise SystemExit(f"quality registry validation refused execution: {exc}") from exc
    todo = [
        row for row in read_jsonl(args.results_jsonl)
        if row.get("status") == "OK" and row.get("centroids_file")
    ]
    registry_linkage_errors = []
    for position, row in enumerate(todo):
        try:
            validate_recorded_registry_evidence(
                row, registry_evidence_for_row(row, registry)
            )
        except ValueError as exc:
            registry_linkage_errors.append({
                "position": position, "run_id": row.get("run_id"),
                "case_id": row.get("case_id"), "error": str(exc),
            })
    if registry_linkage_errors:
        raise SystemExit(
            "quality evaluation refused candidate rows with mismatched registry evidence: "
            + json.dumps(registry_linkage_errors[:20], ensure_ascii=False)
        )
    protocol = build_protocol(args)
    protocol_sha256 = stable_hash(protocol)
    try:
        gpu_resources = create_faiss_gpu_resources(gpu_backend_requested(args))
    except RuntimeError as exc:
        raise SystemExit(f"GPU quality backend initialization failed: {exc}") from exc
    try:
        envelope, protocol_artifact = freeze_protocol_envelope(args.protocol_output, protocol)
    except ValueError as exc:
        raise SystemExit(f"quality protocol freeze refused execution: {exc}") from exc
    existing: set[str] = set()
    if args.output.exists():
        existing_rows = read_jsonl([args.output])
        mismatched_rows = [
            {
                "quality_key": entry.get("quality_key"),
                "protocol_sha256": entry.get("protocol_sha256"),
            }
            for entry in existing_rows
            if entry.get("quality_version") == QUALITY_VERSION
            and entry.get("protocol_sha256") != protocol_sha256
        ]
        if mismatched_rows:
            raise SystemExit(
                "quality output already contains rows from a different protocol; refusing to mix: "
                + json.dumps(mismatched_rows[:10], ensure_ascii=False)
            )
        if not args.force:
            existing = {
                str(entry.get("quality_key")) for entry in existing_rows
                if entry.get("quality_key")
            }
    hash_cache: dict[Path, str] = {}
    evaluated = skipped = failed = 0
    for row in todo:
        row_seeds = seed_provenance(row)
        registry_evidence = registry_evidence_for_row(row, registry)
        registry_linkage = validate_recorded_registry_evidence(row, registry_evidence)
        centroids_path = Path(str(row["centroids_file"]))
        centroid_artifact: dict[str, Any]
        try:
            centroid_artifact = {
                **file_identity(centroids_path),
                "sha256": file_sha256(centroids_path),
            }
        except Exception as exc:
            centroid_artifact = {"path": str(centroids_path), "sha256": None}
            centroid_error = exc
        else:
            centroid_error = None
        recorded_centroids = row.get("centroids_identity")
        recorded_centroids = recorded_centroids if isinstance(recorded_centroids, dict) else {}
        recorded_centroids_sha256 = recorded_centroids.get("sha256")
        centroid_artifact["run_recorded_sha256"] = recorded_centroids_sha256
        if isinstance(recorded_centroids_sha256, str) and len(recorded_centroids_sha256) == 64:
            centroid_artifact["provenance_status"] = "VERIFIED_AGAINST_RUN_RECORDED_SHA256"
            if centroid_artifact.get("sha256") != recorded_centroids_sha256:
                centroid_error = ValueError(
                    "centroid artifact SHA256 changed since candidate execution: "
                    f"{centroid_artifact.get('sha256')} != {recorded_centroids_sha256}"
                )
        else:
            centroid_artifact["provenance_status"] = "LEGACY_NO_RUN_RECORDED_SHA256"
        key = stable_hash({
            "result_key": result_key(row),
            "protocol_sha256": protocol_sha256,
            "centroids_sha256": centroid_artifact.get("sha256"),
            "data_fingerprint_sha256": stable_hash(row.get("data_fingerprint", {})),
            "seed_provenance": row_seeds,
            "registry_file_sha256": registry_evidence["registry_artifact"]["sha256"],
            "registry_dataset_entry_sha256": registry_evidence["dataset_entry_sha256"],
        })
        if key in existing:
            continue
        entry: dict[str, Any] = {
            "quality_version": QUALITY_VERSION,
            "quality_key": key,
            "result_key": result_key(row),
            "evaluated_at": utc_now(),
            "protocol": protocol,
            "protocol_sha256": protocol_sha256,
            "protocol_envelope_id": envelope["id"],
            "protocol_envelope_payload_sha256": protocol_artifact["payload_sha256"],
            "protocol_artifact": protocol_artifact,
            "protocol_artifact_metadata": {
                "protocol_file_sha256": protocol_artifact["file_sha256"],
                "protocol_envelope_payload_sha256": protocol_artifact["payload_sha256"],
                "registry_contract_version": registry["contract_version"],
                "registry_file_sha256": registry_evidence["registry_artifact"]["sha256"],
                "registry_dataset_entry_sha256": registry_evidence["dataset_entry_sha256"],
                "source_artifact_sha256": None,
                "source_artifact_kind": None,
                "gpu_admission_report_sha256": (
                    gpu_admission_report.get("report_sha256")
                    if gpu_admission_report is not None else None
                ),
            },
            "gpu_admission_report": gpu_admission_report,
            "registry_evidence": registry_evidence,
            "candidate_registry_linkage": registry_linkage,
            "run_id": row.get("run_id"),
            "case_id": row.get("case_id"),
            "track": row.get("track"),
            "algorithm": row.get("algorithm"),
            "dtype": row.get("dtype"),
            "device": row.get("device"),
            "seed": row_seeds["run_seed"],
            "data_seed": row_seeds["data_seed"],
            "init_seed": row_seeds["init_seed"],
            "repeat_id": row_seeds["repeat_id"],
            "order_block": row_seeds["order_block"],
            "order_index": row_seeds["order_index"],
            "warmup_state": row_seeds["warmup_state"],
            "dataset": row.get("dataset"),
            "centroids_artifact": centroid_artifact,
            "status": "OK",
            "error": None,
            "quality": {},
        }
        quality_started = time.perf_counter()
        try:
            if centroid_error is not None:
                raise centroid_error
            entry["quality"] = evaluate_centroids_v2(
                row, args.data_root, registry, protocol=protocol,
                compute_source_sha256=not args.skip_source_sha256,
                hash_cache=hash_cache,
                gpu_resources=gpu_resources,
            )
            data_slice = entry["quality"].get("data_slice", {})
            source_artifact = data_slice.get("source_artifact", {}) if isinstance(data_slice, dict) else {}
            source_sha256 = source_artifact.get("sha256") if isinstance(source_artifact, dict) else None
            source_kind = source_artifact.get("kind") if isinstance(source_artifact, dict) else None
            if source_sha256 is None and source_kind == "synthetic_generator":
                source_sha256 = data_slice.get("reconstructed_slice_sha256")
                source_kind = "synthetic_generator_reconstructed_slice"
            entry["protocol_artifact_metadata"].update(
                source_artifact_sha256=source_sha256,
                source_artifact_kind=source_kind,
            )
            evaluated += 1
        except (FileNotFoundError, MemoryError, ValueError) as exc:
            entry.update(status="SKIPPED_CONFIG", error=f"{type(exc).__name__}: {exc}")
            skipped += 1
        except Exception as exc:
            entry.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
            failed += 1
        entry["quality_total_seconds"] = time.perf_counter() - quality_started
        entry["entry_payload_sha256"] = stable_hash(entry)
        append_jsonl(args.output, entry)
        existing.add(key)
        print(json.dumps({
            "case": entry["case_id"], "algorithm": entry["algorithm"],
            "seed": entry["seed"], "status": entry["status"],
            "ivf_recall_status": entry.get("quality", {}).get("ivf_recall", {}).get("status"),
        }, ensure_ascii=False))
    print(json.dumps({
        "output": str(args.output), "protocol_sha256": protocol_sha256,
        "protocol_output": str(args.protocol_output),
        "protocol_envelope_payload_sha256": protocol_artifact["payload_sha256"],
        "registry": str(args.registry),
        "registry_file_sha256": registry["artifact"]["sha256"],
        "gpu_admission_report_sha256": (
            gpu_admission_report.get("report_sha256")
            if gpu_admission_report is not None else None
        ),
        "evaluated": evaluated, "skipped": skipped, "failed": failed,
        "source_files_hashed": len(hash_cache),
    }, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
