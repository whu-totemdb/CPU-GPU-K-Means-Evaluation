#!/usr/bin/env python3
"""Serial, guarded runner for comparable high-dimensional GPU K-Means runs.

Every algorithm runs in a fresh child process.  The parent admits one physical
GPU immediately before each case; the child sees exactly one logical GPU via
``CUDA_VISIBLE_DEVICES``.  SuperKMeans uses a second, bounded-memory preparation
subprocess so its raw-f32 input never coexists with a Python n×d training array.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from preflight import admission_report
from registry_utils import load_dataset_registry, registry_reference, same_registry_reference
from result_contract import (
    config_sha256,
    exception_status,
    file_identity,
    file_sha256,
    make_run_context,
    new_result_row,
    ProcessTreeRssSampler,
    process_peak_rss_gib,
    stable_hash,
    validate_config,
    validate_result_row,
)
from super_sampling_contract import sampling_cardinality

ROOT = Path(__file__).resolve().parent
DEFAULT_REGISTRY_PATH = ROOT / "data_registry.json"
REGISTRY, REGISTRY_IDENTITY = load_dataset_registry(DEFAULT_REGISTRY_PATH)
GIB = 1024 ** 3


def configure_registry(path: Path) -> dict[str, dict[str, Any]]:
    """Select one registry before validation/loading in either parent or child."""
    global REGISTRY, REGISTRY_IDENTITY
    REGISTRY, REGISTRY_IDENTITY = load_dataset_registry(path)
    return REGISTRY


def case_registry_reference(case: dict[str, Any]) -> dict[str, Any]:
    dataset = case.get("dataset") if case.get("source") == "dataset" else None
    return registry_reference(REGISTRY, REGISTRY_IDENTITY, dataset)


class ConfigSkip(RuntimeError):
    """A deliberate non-run: record it, do not turn it into an implementation failure."""

    def __init__(self, message: str, *, failure_class: str | None = None,
                 failure_code: str | None = None, failure_phase: str | None = None,
                 failure_scope: str | None = None,
                 candidate_attributable: str | None = None,
                 retry_disposition: str | None = None,
                 root_cause: str | None = None) -> None:
        super().__init__(message)
        self.taxonomy = {
            key: value for key, value in {
                "failure_class": failure_class,
                "failure_code": failure_code,
                "failure_phase": failure_phase,
                "failure_scope": failure_scope,
                "candidate_attributable": candidate_attributable,
                "retry_disposition": retry_disposition,
                "root_cause": root_cause,
            }.items() if value is not None
        }


class GpuMemorySampler:
    """Measure peak device-memory delta without enumerating GPU processes.

    The child sees one physical card in ``CUDA_VISIBLE_DEVICES``.  Polling the
    aggregate ``memory.used`` counter avoids inspecting any process identity;
    the conservative admission gate has already required an idle card.  Native
    library allocations (FAISS/cuML/SuperKMeans) are therefore measurable even
    when their Python APIs expose no allocator statistics.
    """

    def __init__(self, interval_sec: float = 0.25, gpu: str | int | None = None):
        self.interval_sec = interval_sec
        self.gpu = (str(gpu) if gpu is not None
                    else (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",", 1)[0].strip())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_mib: float | None = None
        self.peak_used_mib: float | None = None
        self.samples = 0
        self.error: str | None = None

    def _read_used_mib(self) -> float:
        query = subprocess.run(
            ["nvidia-smi", "-i", self.gpu, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, check=False, timeout=5,
        )
        if query.returncode != 0:
            raise RuntimeError(query.stderr.strip() or query.stdout.strip() or "nvidia-smi memory query failed")
        return float(query.stdout.splitlines()[0].strip())

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                value = self._read_used_mib()
                self.samples += 1
                self.peak_used_mib = value if self.peak_used_mib is None else max(self.peak_used_mib, value)
            except Exception as exc:  # measurement failure must not abort an algorithm run
                self.error = f"{type(exc).__name__}: {exc}"
                return
            self._stop.wait(self.interval_sec)

    def __enter__(self) -> "GpuMemorySampler":
        try:
            self.baseline_mib = self._read_used_mib()
            self.peak_used_mib = self.baseline_mib
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return self
        self._thread = threading.Thread(target=self._poll, name="gpu-memory-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 4))

    def result(self) -> dict[str, Any]:
        peak_delta = None
        if self.baseline_mib is not None and self.peak_used_mib is not None:
            peak_delta = max(0.0, self.peak_used_mib - self.baseline_mib) / 1024.0
        return {
            "method": "nvidia_smi_total_used_delta_v1", "physical_gpu": self.gpu,
            "interval_sec": self.interval_sec, "samples": self.samples,
            "baseline_used_mib": self.baseline_mib, "peak_used_mib": self.peak_used_mib,
            "peak_delta_gib": peak_delta, "error": self.error,
        }


@dataclass(frozen=True)
class AlgorithmPlugin:
    name: str
    implementation_family: str
    supports_shared_initialization: bool
    uses_python_matrix: bool
    runner: Callable[..., dict[str, Any]]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def dtype_bytes(dtype: str) -> int:
    return {"fp16": 2, "fp32": 4}[dtype]


def estimate_memory_gib(case: dict, workspace_multiplier: float) -> dict:
    """Conservative per-case GPU estimate; it rejects rather than overcommits."""
    n, d, k = int(case["n"]), int(case.get("d") or 0), int(case["k"])
    if not d:
        d = int(REGISTRY[case["dataset"]]["expected_d"])
    elem = dtype_bytes(case["dtype"])
    data = n * d * elem
    centers = k * d * elem
    labels = n * 4
    accumulators = k * d * 4 + k * 4
    total = data * workspace_multiplier + centers * 2 + labels * 2 + accumulators
    return {
        "d": d, "data_gib": round(data / GIB, 3), "centers_gib": round(centers / GIB, 3),
        "estimated_peak_gib": round(total / GIB, 3), "workspace_multiplier": workspace_multiplier,
    }


def make_row(case: dict, algorithm: str, *, d: int, hardware: str, gpu_index: int, context: dict) -> dict:
    row = new_result_row(case=case, algorithm=algorithm, d=d, hardware=hardware, gpu_index=gpu_index, context=context)
    row["registry_evidence"] = case_registry_reference(case)
    return row


def run_seeds(config: dict) -> list[int]:
    raw = config.get("seeds", [config.get("seed", 42)])
    if not isinstance(raw, list) or not raw:
        raise ValueError("seeds must be a non-empty list when provided")
    try:
        seeds = [int(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("seeds must contain integers") from exc
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return seeds


def run_specs(config: dict) -> list[dict[str, Any]]:
    """Return explicit, independently seeded repetitions.

    Legacy configs keep their existing ``seeds`` behaviour.  A v2 controlled
    config can instead freeze one data slice while varying shared initial
    centres through ``run_plan``.  ``run_seed`` remains the append-only cell
    identity; ``data_seed`` and ``init_seed`` make the causal factors explicit.
    """
    plan = config.get("run_plan")
    default_init = None
    if isinstance(config.get("initialization"), dict):
        default_init = config["initialization"].get("seed")
    if plan is None:
        return [
            {
                "run_seed": seed,
                "data_seed": seed,
                "init_seed": int(default_init) if default_init is not None else seed,
                "repeat_id": index,
                "order_block": None,
                "order_index": None,
                "warmup_state": config.get("warmup_state", "unspecified"),
            }
            for index, seed in enumerate(run_seeds(config))
        ]
    if not isinstance(plan, list) or not plan:
        raise ValueError("run_plan must be a non-empty list")
    result: list[dict[str, Any]] = []
    for index, item in enumerate(plan):
        if not isinstance(item, dict):
            raise ValueError(f"run_plan[{index}] must be an object")
        try:
            run_seed = int(item.get("run_seed", item.get("seed")))
            data_seed = int(item.get("data_seed", run_seed))
            init_seed = int(item.get("init_seed", default_init if default_init is not None else run_seed))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"run_plan[{index}] requires integer run_seed/data_seed/init_seed") from exc
        result.append({
            "run_seed": run_seed,
            "data_seed": data_seed,
            "init_seed": init_seed,
            "repeat_id": item.get("repeat_id", index),
            "order_block": item.get("order_block"),
            "order_index": item.get("order_index"),
            "warmup_state": item.get("warmup_state", config.get("warmup_state", "unspecified")),
        })
    identities = [item["run_seed"] for item in result]
    if len(identities) != len(set(identities)):
        raise ValueError("run_plan.run_seed values must be unique")
    return result


def algorithm_options(config: dict, algorithm: str) -> dict:
    options = config.get("algorithm_options", {})
    if not isinstance(options, dict):
        raise ValueError("algorithm_options must be an object")
    value = options.get(algorithm, {})
    if not isinstance(value, dict):
        raise ValueError(f"algorithm_options.{algorithm} must be an object")
    return value


def planned_case_metadata(case: dict, data_root: Path, data_seed: int) -> dict[str, Any]:
    """Build task identity before dispatch so timeout/OOM rows remain usable."""
    n = int(case["n"])
    if case["source"] == "synthetic":
        d = int(case["d"])
        from synthetic_controls import normalized_spec

        controlled = normalized_spec(case)
        generator = ({
            "generator": "controlled_kmeans_geometry_v1",
            "seed": int(data_seed), "n": n, "d": d,
            "parameters": controlled,
            "controlled_feature": case.get("controlled_feature", "baseline"),
            "feature_level": case.get("feature_level", "baseline"),
            "control_block": case.get("control_block", "reference_scale_v1"),
        } if controlled is not None else {
            "generator": "numpy.default_rng.standard_normal.float32.v1",
            "seed": int(data_seed), "n": n, "d": d,
        })
        return {
            "d": d,
            "metric": str(controlled.get("metric", "l2")) if controlled is not None else "l2",
            "normalized": bool(controlled.get("normalize", False)) if controlled is not None else False,
            "data_fingerprint": {
                "kind": "synthetic_controlled_v1" if controlled is not None else "synthetic_v1",
                "generator_spec": generator,
                "generator_spec_sha256": stable_hash(generator),
            },
        }
    spec = REGISTRY[case["dataset"]]
    path = data_root / spec["filename"]
    total = int(spec["expected_train_rows"])
    sampling = ({
        "method": "prefix_first_n", "seed": None,
        "requested_rows": n, "source_train_rows": total,
    } if case.get("input_selection") == "prefix_rows_v1" else {
        "method": "all_rows" if n == total else "sorted_choice_without_replacement",
        "seed": int(data_seed), "requested_rows": n, "source_train_rows": total,
    })
    return {
        "d": int(spec["expected_d"]), "metric": spec["metric"],
        "normalized": bool(spec.get("normalize")),
        "data_fingerprint": {
            "kind": "hdf5_train_sample_v1", "dataset_key": case["dataset"],
            "dataset_family": spec.get("dataset_family"), "registry_sha256": spec.get("sha256"),
            "registry_sha256_status": spec.get("sha256_status", "UNDECLARED"),
            "observed_file_identity": file_identity(path) if path.is_file() else {"path": str(path), "state": "missing_at_dispatch"},
            "sampling": sampling, "normalization_applied": bool(spec.get("normalize")),
        },
    }


def load_case_data(case: dict, data_root: Path, seed: int) -> tuple[Any, dict]:
    """Load only algorithms that need an in-process Python matrix."""
    import numpy as np

    n = int(case["n"])
    if case["source"] == "synthetic":
        d = int(case["d"])
        from synthetic_controls import generate, normalized_spec

        controlled = normalized_spec(case)
        if controlled is not None:
            matrix, synthetic_spec = generate(case, seed)
            return matrix, {
                "metric": str(controlled.get("metric", "l2")),
                "normalize": bool(controlled.get("normalize", False)),
                "d": d,
                "data_fingerprint": {
                    "kind": "synthetic_controlled_v1",
                    "generator_spec": synthetic_spec,
                    "generator_spec_sha256": stable_hash(synthetic_spec),
                },
            }
        rng = np.random.default_rng(seed)
        synthetic_spec = {
            "generator": "numpy.default_rng.standard_normal.float32.v1", "seed": seed, "n": n, "d": d,
        }
        return rng.standard_normal((n, d), dtype=np.float32), {
            "metric": "l2", "normalize": False, "d": d,
            "data_fingerprint": {
                "kind": "synthetic_v1", "generator_spec": synthetic_spec,
                "generator_spec_sha256": stable_hash(synthetic_spec),
            },
        }

    spec = REGISTRY[case["dataset"]]
    import h5py

    path = data_root / spec["filename"]
    if not path.exists():
        raise FileNotFoundError(f"dataset missing: {path}; see README.md / download_datasets.py")
    with h5py.File(path, "r") as handle:
        if "train" not in handle:
            raise KeyError(f"{path} has no 'train' dataset")
        total, d = handle["train"].shape
        if n > total:
            raise ValueError(f"case asks for n={n}, but {path.name} has only {total} training vectors")
        prefix_rows = case.get("input_selection") == "prefix_rows_v1"
        indices = np.arange(n, dtype=np.int64) if prefix_rows else (np.arange(total) if n == total else np.sort(np.random.default_rng(seed).choice(total, n, replace=False)))
        x = np.asarray(handle["train"][indices], dtype=np.float32)
    if d != spec["expected_d"]:
        raise ValueError(f"{path.name}: d={d}, expected {spec['expected_d']}; registry/file mismatch")
    if spec.get("normalize"):
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        x /= np.maximum(norms, 1e-12)
    sampling = ({
        "method": "prefix_first_n", "seed": None,
        "requested_rows": n, "source_train_rows": int(total),
    } if case.get("input_selection") == "prefix_rows_v1" else {
        "method": "all_rows" if n == total else "sorted_choice_without_replacement",
        "seed": seed, "requested_rows": n, "source_train_rows": int(total),
    })
    return np.ascontiguousarray(x), {
        "metric": spec["metric"], "normalize": bool(spec.get("normalize")), "d": int(d),
        "data_fingerprint": {
            "kind": "hdf5_train_sample_v1", "dataset_key": case["dataset"],
            "dataset_family": spec.get("dataset_family"), "registry_sha256": spec.get("sha256"),
            "registry_sha256_status": spec.get("sha256_status", "UNDECLARED"),
            "observed_file_identity": file_identity(path), "sampling": sampling,
            "normalization_applied": bool(spec.get("normalize")),
        },
    }


def _torch_elapsed_ms(torch: Any, operation: Callable[[], None]) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    operation()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def _check_initial_centers(initial_centers: Any, *, case: dict, d: int):
    import numpy as np

    centres = np.asarray(initial_centers, dtype=np.float32)
    expected = (int(case["k"]), d)
    if centres.shape != expected:
        raise ValueError(f"shared initial centers have shape {centres.shape}, expected {expected}")
    if not np.isfinite(centres).all():
        raise ValueError("shared initial centers contain NaN or infinity")
    return np.ascontiguousarray(centres)


def _configured_python_paths(config: dict, algorithm: str) -> list[str]:
    """Prepend only explicitly configured source/build directories."""
    values = algorithm_options(config, algorithm).get("python_paths", [])
    if not isinstance(values, list):
        raise ValueError(f"algorithm_options.{algorithm}.python_paths must be a list")
    result = []
    for value in values:
        path = str(value)
        if path and path not in sys.path:
            sys.path.insert(0, path)
        if path:
            result.append(path)
    return result


def _configured_source_evidence(config: dict, algorithm: str) -> list[dict[str, Any]]:
    """Hash named source artifacts so a binary/module result is auditable."""
    evidence = []
    for raw in algorithm_options(config, algorithm).get("source_artifacts", []):
        path = Path(str(raw)).expanduser()
        item: dict[str, Any] = {"path": str(path)}
        if path.is_file():
            item.update(file_identity(path), sha256=file_sha256(path))
        else:
            item["state"] = "missing"
        evidence.append(item)
    return evidence


def run_fastkmeans_triton(*, x: Any, case: dict, seed: int,
                          initialization: dict | None, config: dict, **_: Any) -> dict:
    """Run FastKMeans 0.5 on the full input in FP32 through its Triton path."""
    import numpy as np
    import torch

    if initialization is not None:
        raise ConfigSkip("FastKMeans 0.5 exposes random initialization only; operational track only")
    _configured_python_paths(config, "fastkmeans_triton")
    try:
        import fastkmeans as fastkmeans_module
        from fastkmeans import FastKMeans
    except Exception as exc:
        raise RuntimeError(
            "FastKMeans is unavailable; set algorithm_options.fastkmeans_triton.python_paths"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")

    options = algorithm_options(config, "fastkmeans_triton")
    chunk_rows = int(options.get("chunk_size_data", 51200))
    chunk_centroids = int(options.get("chunk_size_centroids", 10240))
    if chunk_rows <= 0 or chunk_centroids <= 0:
        raise ValueError("FastKMeans chunk sizes must be positive")

    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = FastKMeans(
        d=int(x.shape[1]),
        k=int(case["k"]),
        niter=int(case["max_iter"]),
        tol=float(options.get("tol", 1e-4)),
        gpu=True,
        seed=int(seed),
        max_points_per_centroid=None,  # mandatory: no hidden FAISS-style subsampling
        chunk_size_data=chunk_rows,
        chunk_size_centroids=chunk_centroids,
        device=torch.device("cuda:0"),
        dtype=torch.float32,
        pin_gpu_memory=bool(options.get("pin_gpu_memory", True)),
        verbose=False,
        nredo=1,
        use_triton=True,
    )
    if not bool(getattr(model, "use_triton", False)):
        raise RuntimeError("FastKMeans did not activate Triton; refusing a fallback result")
    started = time.perf_counter()
    model.fit(np.ascontiguousarray(x, dtype=np.float32))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    centers = np.asarray(model.centroids, dtype=np.float32)
    if centers.shape != (int(case["k"]), int(x.shape[1])):
        raise RuntimeError(f"FastKMeans returned {centers.shape}, expected {(int(case['k']), int(x.shape[1]))}")
    module_path = Path(inspect.getfile(FastKMeans))
    return {
        "implementation": "fastkmeans_0_5_triton_cuda_fp32_full_data",
        "centroids": centers,
        "n_iter": None,
        "configured_max_iter": int(case["max_iter"]),
        "iteration_count_semantics": "unobserved_configured_max_only",
        "converged": None,
        "peak_device_memory_gib": float(torch.cuda.max_memory_allocated()) / GIB,
        "t_alloc": None,
        "t_h2d": None,
        "t_kernel": elapsed,
        "t_d2h": None,
        "iteration_timing_mode": "aggregate_host_wall",
        "t_iteration_mean_sec": None,
        "compute_timing_semantics": "FastKMeans.fit_host_wall_including_chunk_transfers",
        "initialization_mode": "implementation_seeded_random_full_data",
        "full_data_enforced": True,
        "triton_enforced": True,
        "native_module": str(module_path),
        "native_module_identity": {**file_identity(module_path), "sha256": file_sha256(module_path)},
        "package_version": getattr(fastkmeans_module, "__version__", "0.5.0-source"),
        "source_artifacts": _configured_source_evidence(config, "fastkmeans_triton"),
    }


def run_kmcuda_yinyang(*, x: Any, case: dict, seed: int,
                        initialization: dict | None, config: dict, **_: Any) -> dict:
    """Run KMCUDA Yinyang, preserving its native reassignment stopping rule.

    Upstream KMCUDA does not expose a maximum-iteration parameter or iteration
    count.  The result is therefore useful for operational runtime/quality
    evaluation but is explicitly not a strict fixed-20-iteration comparison.
    """
    import numpy as np

    _configured_python_paths(config, "kmcuda_yinyang")
    try:
        import libKMCUDA
        from libKMCUDA import kmeans_cuda
    except Exception as exc:
        raise RuntimeError(
            "libKMCUDA is unavailable; set algorithm_options.kmcuda_yinyang.python_paths"
        ) from exc
    options = algorithm_options(config, "kmcuda_yinyang")
    tolerance = float(options.get("reassignment_tolerance", 1e-4))
    yinyang_t = float(options.get("yinyang_t", 0.1))
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("KMCUDA reassignment_tolerance must be in [0, 1)")
    if not 0.0 < yinyang_t <= 0.5:
        raise ValueError("KMCUDA yinyang_t must be in (0, 0.5]")
    metric = (
        "L2"
        if case["source"] == "synthetic" or REGISTRY[case["dataset"]]["metric"] == "l2"
        else "cos"
    )
    init_value: Any = (
        _check_initial_centers(initialization["centers"], case=case, d=int(x.shape[1]))
        if initialization is not None
        else "random"
    )
    started = time.perf_counter()
    result = kmeans_cuda(
        np.ascontiguousarray(x, dtype=np.float32),
        int(case["k"]),
        tolerance=tolerance,
        init=init_value,
        yinyang_t=yinyang_t,
        metric=metric,
        average_distance=False,
        seed=int(seed),
        device=1,  # parent exposes one physical card as logical CUDA device 0
        verbosity=int(options.get("verbosity", 0)),
    )
    elapsed = time.perf_counter() - started
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError("libKMCUDA returned an unexpected result")
    centers = np.asarray(result[0], dtype=np.float32)
    if centers.shape != (int(case["k"]), int(x.shape[1])):
        raise RuntimeError(f"KMCUDA returned {centers.shape}, expected {(int(case['k']), int(x.shape[1]))}")
    module_path = Path(str(getattr(libKMCUDA, "__file__", "")))
    return {
        "implementation": "kmcuda_yinyang_cuda_fp32",
        "centroids": centers,
        "n_iter": None,
        "configured_max_iter": int(case["max_iter"]),
        "iteration_count_semantics": "upstream_unobserved_native_reassignment_threshold",
        "converged": None,
        "peak_device_memory_gib": None,
        "t_alloc": None,
        "t_h2d": None,
        "t_kernel": elapsed,
        "t_d2h": None,
        "iteration_timing_mode": "aggregate_host_wall",
        "t_iteration_mean_sec": None,
        "compute_timing_semantics": "libKMCUDA_kmeans_cuda_host_wall_including_transfers",
        "initialization_mode": (
            "shared_centroids_import" if initialization is not None
            else "implementation_seeded_random"
        ),
        "native_stopping_rule": {
            "kind": "relative_reassignment_count",
            "tolerance": tolerance,
            "fixed_max_iter_enforced": False,
        },
        "strict_fixed_iteration_comparable": False,
        "yinyang_t": yinyang_t,
        "metric_api": metric,
        "native_module": str(module_path),
        "native_module_identity": (
            {**file_identity(module_path), "sha256": file_sha256(module_path)}
            if module_path.is_file() else {"path": str(module_path), "state": "unresolved"}
        ),
        "source_artifacts": _configured_source_evidence(config, "kmcuda_yinyang"),
    }


def run_flash_kmeans(*, x: Any, case: dict, seed: int, initialization: dict | None, **_: Any) -> dict:
    """Run FlashKMeans only on its real CUDA+Triton path.

    The public class does not expose initial centres.  For the controlled track
    we invoke the package's Triton implementation directly with the persisted
    centre tensor; this remains the same Triton assignment/update implementation.
    """
    import numpy as np
    import torch
    import flash_kmeans as flash_kmeans_module
    from flash_kmeans import FlashKMeans

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    # The parent exposes exactly one physical card through CUDA_VISIBLE_DEVICES,
    # so this logical ordinal is always zero.  This H800 PyTorch 2.7 build
    # rejects *both* a torch.device and integer device argument for the peak
    # memory APIs.  With one visible CUDA device, their no-argument form is
    # unambiguous and refers to the current logical cuda:0 device.
    device_index = 0
    device = torch.device(f"cuda:{device_index}")
    dtype = torch.float16 if case["dtype"] == "fp16" else torch.float32
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    original_d = int(x.shape[1])
    kernel_d = 1 << (original_d - 1).bit_length()
    # Flash-KMeans 0.3.0 emits ``tl.arange(0, D)`` and Triton requires this
    # range to be a power of two.  Appending identically-zero coordinates is
    # algebraically exact: pairwise L2 distances, inner products and every
    # Lloyd centroid update in the original coordinates are unchanged.  Export
    # only the original dimensions so all candidates share the same (k, d).
    preprocess_started = time.perf_counter()
    if kernel_d != original_d:
        padded = np.zeros((x.shape[0], kernel_d), dtype=np.float32)
        padded[:, :original_d] = x
        x_kernel = padded
        input_transform = {
            "id": "zero_pad_to_power_of_two_v1", "original_d": original_d,
            "kernel_d": kernel_d, "added_zero_dimensions": kernel_d - original_d,
            "invariance": "zero coordinates preserve L2 distances, dot products, and Lloyd updates",
        }
    else:
        x_kernel = x
        input_transform = {"id": "identity_v1", "original_d": original_d, "kernel_d": kernel_d}
    t_preprocess = time.perf_counter() - preprocess_started
    host = torch.from_numpy(x_kernel)
    payload: dict[str, Any] = {}

    def h2d() -> None:
        payload["x_device"] = host.to(device=device, dtype=dtype, non_blocking=False)

    t_h2d = _torch_elapsed_ms(torch, h2d) / 1000.0
    if initialization is None:
        def allocate() -> None:
            payload["model"] = FlashKMeans(
                d=kernel_d, k=int(case["k"]), niter=int(case["max_iter"]), tol=1e-4,
                use_triton=True, seed=seed, verbose=False, dtype=dtype, device=device,
            )

        t_alloc = _torch_elapsed_ms(torch, allocate) / 1000.0
        model = payload["model"]
        if not model.use_triton:
            raise RuntimeError("FlashKMeans did not load Triton; refusing a fallback result")
        t_kernel = _torch_elapsed_ms(torch, lambda: model.fit(payload["x_device"])) / 1000.0
        centroids = model.centroids_b.squeeze(0)
        n_iter = None
        init_mode = "implementation_seeded_default"
        iteration_count_semantics = "unobserved_configured_max_only"
    else:
        try:
            from flash_kmeans.kmeans_triton_impl import batch_kmeans_Euclid
            from flash_kmeans.interface import _HAS_TRITON_IMPL
        except Exception as exc:
            raise RuntimeError("FlashKMeans controlled Triton API is unavailable") from exc
        if not _HAS_TRITON_IMPL:
            raise RuntimeError("FlashKMeans Triton implementation is unavailable")
        centres = _check_initial_centers(initialization["centers"], case=case, d=original_d)
        if kernel_d != original_d:
            padded_centres = np.zeros((centres.shape[0], kernel_d), dtype=np.float32)
            padded_centres[:, :original_d] = centres
            centres = padded_centres
        init_host = torch.from_numpy(centres).unsqueeze(0)

        def allocate() -> None:
            payload["initial_device"] = init_host.to(device=device, dtype=dtype, non_blocking=False)

        t_alloc = _torch_elapsed_ms(torch, allocate) / 1000.0
        result: dict[str, Any] = {}

        def fit_controlled() -> None:
            labels, centres_out, iterations = batch_kmeans_Euclid(
                payload["x_device"].unsqueeze(0), int(case["k"]), max_iters=int(case["max_iter"]),
                tol=1e-4, init_centroids=payload["initial_device"], verbose=False,
            )
            result.update(labels=labels, centroids=centres_out.squeeze(0), iterations=int(iterations))

        t_kernel = _torch_elapsed_ms(torch, fit_controlled) / 1000.0
        centroids, n_iter = result["centroids"], result["iterations"]
        init_mode = "shared_centroids_triton_direct_api"
        iteration_count_semantics = "observed_kernel_return_value"

    def d2h() -> None:
        payload["centroids"] = centroids.float().cpu().numpy()[:, :original_d]

    t_d2h = _torch_elapsed_ms(torch, d2h) / 1000.0
    return {
        "implementation": "flash_kmeans_triton_cuda", "centroids": np.asarray(payload["centroids"], dtype=np.float32),
        "n_iter": n_iter, "configured_max_iter": int(case["max_iter"]),
        "iteration_count_semantics": iteration_count_semantics,
        "converged": None, "peak_device_memory_gib": torch.cuda.max_memory_allocated() / GIB,
        "t_preprocess": t_preprocess, "t_alloc": t_alloc, "t_h2d": t_h2d, "t_kernel": t_kernel, "t_d2h": t_d2h,
        "iteration_timing_mode": "aggregate_gpu_event",
        "t_iteration_mean_sec": t_kernel / max(1, n_iter) if n_iter is not None else None,
        "compute_timing_semantics": "flash_triton_fit_gpu_event",
        "initialization_mode": init_mode,
        "flash_kmeans_runtime": {
            "module_file": getattr(flash_kmeans_module, "__file__", None),
            "module_version": getattr(flash_kmeans_module, "__version__", None),
            "class_init_signature": str(inspect.signature(FlashKMeans.__init__)),
        },
        "input_transform": input_transform,
    }


def run_faiss_gpu(*, x: Any, case: dict, seed: int, initialization: dict | None, **_: Any) -> dict:
    import faiss
    import numpy as np

    if initialization is not None:
        raise ConfigSkip("faiss_gpu does not expose a verified external-centroid API in this runner; operational track only")
    if not hasattr(faiss, "StandardGpuResources") or faiss.get_num_gpus() != 1:
        raise RuntimeError("FAISS GPU requires exactly one visible CUDA device")
    start = time.perf_counter()
    km = faiss.Kmeans(x.shape[1], int(case["k"]), niter=int(case["max_iter"]), nredo=1, verbose=False,
                      seed=seed, gpu=True, min_points_per_centroid=1, max_points_per_centroid=max(len(x), 1))
    alloc_end = time.perf_counter()
    index = km.index
    index_types = [cls.__name__ for cls in type(index).__mro__]
    if not any(name.startswith("GpuIndex") for name in index_types) or index.getDevice() != 0:
        raise RuntimeError(f"FAISS GPU index on logical device 0 was not confirmed: {index_types}")
    train_started = time.perf_counter()
    km.train(np.ascontiguousarray(x, dtype=np.float32))
    end = time.perf_counter()
    stats = getattr(km, "iteration_stats", None)
    n_iter = len(stats) if stats is not None else None
    return {
        "implementation": "faiss_gpu_kmeans_fp32", "centroids": np.asarray(km.centroids, dtype=np.float32),
        "n_iter": n_iter, "configured_max_iter": int(case["max_iter"]),
        "iteration_count_semantics": "observed_iteration_stats_length" if n_iter is not None else "unobserved_configured_max_only",
        "converged": None, "peak_device_memory_gib": None,
        "t_alloc": alloc_end - start, "t_h2d": None, "t_kernel": end - train_started, "t_d2h": None,
        "iteration_timing_mode": "aggregate_host_wall",
        "t_iteration_mean_sec": (end - train_started) / max(1, n_iter) if n_iter is not None else None,
        "compute_timing_semantics": "faiss_kmeans_train_opaque_host_wall",
        "initialization_mode": "implementation_seeded_default",
    }


def run_cuml_kmeans(*, x: Any, case: dict, seed: int, initialization: dict | None, **_: Any) -> dict:
    import cupy as cp
    import numpy as np
    from cuml.cluster import KMeans

    if initialization is not None:
        raise ConfigSkip("cuml_kmeans external-centroid initialization is not verified for this version; operational track only")
    cp.cuda.Device(0).use()
    cp.get_default_memory_pool().free_all_blocks()
    start = time.perf_counter()
    model = KMeans(n_clusters=int(case["k"]), max_iter=int(case["max_iter"]), n_init=1,
                   random_state=seed, output_type="cupy")
    alloc_end = time.perf_counter()
    model.fit(np.ascontiguousarray(x, dtype=np.float32))
    cp.cuda.Stream.null.synchronize()
    end = time.perf_counter()
    centers = cp.asnumpy(model.cluster_centers_)
    cp.cuda.Stream.null.synchronize()
    observed_n_iter = getattr(model, "n_iter_", None)
    n_iter = int(observed_n_iter) if observed_n_iter is not None else None
    return {
        "implementation": "cuml_kmeans_cuda_fp32", "centroids": np.asarray(centers, dtype=np.float32),
        "n_iter": n_iter, "configured_max_iter": int(case["max_iter"]),
        "iteration_count_semantics": "observed_model_n_iter" if n_iter is not None else "unobserved_configured_max_only",
        "converged": None, "peak_device_memory_gib": None,
        "t_alloc": alloc_end - start, "t_h2d": None, "t_kernel": end - alloc_end, "t_d2h": None,
        "iteration_timing_mode": "aggregate_host_wall",
        "t_iteration_mean_sec": (end - alloc_end) / max(1, n_iter) if n_iter is not None else None,
        "compute_timing_semantics": "cuml_fit_plus_internal_transfers_opaque_host_wall",
        "initialization_mode": "implementation_seeded_default",
    }


def _last_json(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(f"subprocess returned no JSON; last output: {stdout[-1000:]}")


def _super_runner_path(options: dict, explicit_runner: Path | None) -> Path:
    raw = explicit_runner or options.get("runner") or os.environ.get("SUPERKMEANS_GPU_RUNNER")
    if not raw:
        raise ConfigSkip("superkmeans_gpu requires --super-runner, algorithm_options.superkmeans_gpu.runner, or SUPERKMEANS_GPU_RUNNER")
    runner = Path(raw).expanduser()
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise ConfigSkip(f"SuperKMeans GPU runner is not an executable file: {runner}")
    return runner


def run_superkmeans_gpu(*, x: Any, case: dict, seed: int, data_seed: int | None = None,
                         initialization: dict | None,
                         config: dict, data_root: Path, output_dir: Path, super_runner: Path | None, **_: Any) -> dict:
    """Run the native C++ CUDA binary without retaining an n×d Python matrix."""
    import numpy as np

    if initialization is not None:
        raise ConfigSkip("native SuperKMeans runner has no verified external-centroid entry point; operational track only")
    options = algorithm_options(config, "superkmeans_gpu")
    data_seed = int(seed if data_seed is None else data_seed)
    runner = _super_runner_path(options, super_runner)
    threads = int(options.get("threads", 4))
    if threads <= 0 or threads > 8:
        raise ConfigSkip("superkmeans_gpu.threads must be in [1, 8] to protect the shared host")
    sampling_fraction = float(options.get("sampling_fraction", 0.3))
    if not 0.0 < sampling_fraction <= 1.0:
        raise ValueError("superkmeans_gpu.sampling_fraction must be in (0, 1]")
    sampling_proof = sampling_cardinality(
        n=int(case["n"]), k=int(case["k"]), sampling_fraction=sampling_fraction,
    )
    if sampling_proof["unsupported"]:
        root_cause = (
            f"configured sampling_fraction={sampling_fraction:g} provides at most "
            f"{sampling_proof['configured_sample_count']:g} sampled points for k={int(case['k'])}"
        )
        raise ConfigSkip(
            f"SuperKMeans unsupported configuration: {root_cause}",
            failure_class="ALGORITHM_CAPABILITY",
            failure_code="UNSUPPORTED_CONFIG",
            failure_phase="INITIALISATION",
            failure_scope="CANDIDATE_TASK_CONFIG",
            candidate_attributable="YES",
            retry_disposition="DO_NOT_RETRY_IDENTICAL_CONFIG",
            root_cause=root_cause,
        )
    chunk_rows = int(options.get("prepare_chunk_rows", 2048))
    if chunk_rows <= 0:
        raise ValueError("superkmeans_gpu.prepare_chunk_rows must be positive")
    d = int(case.get("d") or REGISTRY[case["dataset"]]["expected_d"])
    metric = "l2" if case["source"] == "synthetic" else REGISTRY[case["dataset"]]["metric"]
    normalise = bool(case["source"] == "dataset" and REGISTRY[case["dataset"]].get("normalize"))
    work_dir = output_dir / "work" / "superkmeans_gpu"
    work_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{case['id']}__dataseed{data_seed}__initseed{seed}"
    raw_input = work_dir / f"{stem}.f32"
    raw_centroids = work_dir / f"{stem}.centroids.f32"
    prepared_metadata = raw_input.with_suffix(raw_input.suffix + ".json")
    command_prepare = [sys.executable, str(ROOT / "prepare_superkmeans_input.py"), "--output", str(raw_input),
                       "--n", str(case["n"]), "--seed", str(data_seed), "--chunk-rows", str(chunk_rows)]
    if case["source"] == "dataset":
        command_prepare.extend(["--input", str(data_root / REGISTRY[case["dataset"]]["filename"])])
        if case.get("input_selection") == "prefix_rows_v1":
            command_prepare.append("--prefix-rows")
        if normalise:
            command_prepare.append("--normalize")
    else:
        command_prepare.extend(["--synthetic-d", str(d)])
    prep_started = time.perf_counter()
    prep = subprocess.run(command_prepare, text=True, capture_output=True, check=False)
    t_preprocess = time.perf_counter() - prep_started
    if prep.returncode != 0:
        raise RuntimeError(f"SuperKMeans input preparation failed: {prep.stderr[-1000:] or prep.stdout[-1000:]}")
    prep_metadata = _last_json(prep.stdout)
    command = [str(runner), "--input", str(raw_input), "--n", str(case["n"]), "--d", str(d), "--k", str(case["k"]),
               "--iters", str(case["max_iter"]), "--seed", str(seed), "--threads", str(threads),
               "--sampling-fraction", str(sampling_fraction), "--centroids-out", str(raw_centroids)]
    if metric == "ip":
        command.append("--angular")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)})
    try:
        process = subprocess.run(command, text=True, capture_output=True, env=env, check=False)
        if process.returncode != 0:
            raise RuntimeError(f"native SuperKMeans failed: {process.stderr[-1000:] or process.stdout[-1000:]}")
        native = _last_json(process.stdout)
        if native.get("status") != "OK":
            raise RuntimeError(f"native SuperKMeans reported {native}")
        expected_values = int(case["k"]) * d
        centroids = np.fromfile(raw_centroids, dtype=np.float32)
        if centroids.size != expected_values:
            raise RuntimeError(f"native centroid file has {centroids.size} values, expected {expected_values}")
        n_iter = int(native.get("n_iter", case["max_iter"]))
        train_wall = float(native.get("t_train_wall", native.get("t_total", 0.0)))
        return {
            "implementation": "native_gpu_superkmeans_cuda", "centroids": centroids.reshape(int(case["k"]), d),
            "n_iter": n_iter, "converged": None, "peak_device_memory_gib": native.get("peak_device_memory_gib"),
            "t_alloc": native.get("t_model_init"), "t_h2d": native.get("t_h2d"), "t_kernel": train_wall,
            "t_d2h": native.get("t_d2h"), "t_preprocess": t_preprocess,
            "iteration_timing_mode": "aggregate_native_wall", "t_iteration_mean_sec": train_wall / max(1, n_iter),
            "compute_timing_semantics": "native_superkmeans_train_wall",
            "initialization_mode": "implementation_seeded_default", "superkmeans_native": native,
            "superkmeans_command": command, "superkmeans_runner_identity": file_identity(runner),
            "data_fingerprint": {"kind": "superkmeans_raw_f32_prepared_v1", "prepared": prep_metadata},
        }
    finally:
        if not bool(options.get("keep_prepared_input", False)):
            for path in (raw_input, prepared_metadata, raw_centroids):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass


ALGORITHMS: dict[str, AlgorithmPlugin] = {
    "flash_kmeans": AlgorithmPlugin("flash_kmeans", "flash_kmeans_triton_cuda", True, True, run_flash_kmeans),
    "faiss_gpu": AlgorithmPlugin("faiss_gpu", "faiss_gpu_kmeans", False, True, run_faiss_gpu),
    "cuml_kmeans": AlgorithmPlugin("cuml_kmeans", "cuml_kmeans_cuda", False, True, run_cuml_kmeans),
    "superkmeans_gpu": AlgorithmPlugin("superkmeans_gpu", "native_gpu_superkmeans_cuda", False, False, run_superkmeans_gpu),
    "fastkmeans_triton": AlgorithmPlugin(
        "fastkmeans_triton", "fastkmeans_0_5_triton_cuda", False, True, run_fastkmeans_triton
    ),
    "kmcuda_yinyang": AlgorithmPlugin(
        "kmcuda_yinyang", "kmcuda_yinyang_cuda", True, True, run_kmcuda_yinyang
    ),
}


class InvalidCentroidOutput(RuntimeError):
    """The candidate returned an artifact that cannot represent a K-Means result."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def persist_validated_centroids(
    centroids: Any,
    *,
    expected_shape: tuple[int, int],
    valid_path: Path,
    invalid_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Persist finite centroids and quarantine invalid candidate output."""
    import numpy as np

    matrix = np.asarray(centroids)
    finite = np.isfinite(matrix)
    diagnostics = {
        "expected_shape": list(expected_shape),
        "observed_shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
        "value_count": int(matrix.size),
        "finite_count": int(finite.sum()),
        "nonfinite_count": int(matrix.size - finite.sum()),
        "nan_count": int(np.isnan(matrix).sum()),
        "positive_inf_count": int(np.isposinf(matrix).sum()),
        "negative_inf_count": int(np.isneginf(matrix).sum()),
    }
    shape_valid = tuple(matrix.shape) == tuple(expected_shape)
    finite_valid = diagnostics["nonfinite_count"] == 0
    if not shape_valid or not finite_valid:
        invalid_dir.mkdir(parents=True, exist_ok=True)
        invalid_path = invalid_dir / valid_path.name
        np.save(invalid_path, matrix)
        diagnostics["quarantined_artifact"] = {
            **file_identity(invalid_path),
            "sha256": file_sha256(invalid_path),
        }
        reasons = []
        if not shape_valid:
            reasons.append(
                f"centroids have shape {tuple(matrix.shape)}, expected {tuple(expected_shape)}"
            )
        if not finite_valid:
            reasons.append(
                f"centroids contain {diagnostics['nonfinite_count']} NaN or infinity values"
            )
        raise InvalidCentroidOutput("; ".join(reasons), diagnostics=diagnostics)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(valid_path, matrix)
    return valid_path, diagnostics


def resolve_initialization(config: dict, case: dict, data_seed: int, init_seed: int, config_path: Path) -> dict | None:
    if config.get("track") != "controlled":
        return None
    init = config.get("initialization")
    if not isinstance(init, dict) or init.get("policy") != "shared_forgy_v1":
        raise ConfigSkip("controlled track requires initialization.policy='shared_forgy_v1'")
    centres_dir = config_path.parent / str(init.get("centers_dir", "initial_centers"))
    filename = f"{case['id']}__runseed{data_seed}__initseed{init_seed}.npy"
    path = centres_dir / filename
    metadata_path = path.with_suffix(path.suffix + ".json")
    if not path.is_file() or not metadata_path.is_file():
        raise ConfigSkip(f"missing shared initial centers; run generate_initial_centers.py for {case['id']} data_seed {data_seed}")
    import numpy as np

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_registry = case_registry_reference(case)
    if not same_registry_reference(metadata.get("registry_evidence"), expected_registry):
        raise ConfigSkip("shared initial centers were generated from a different dataset registry")
    centres = np.load(path, mmap_mode="r")
    expected_shape = (int(case["k"]), int(case.get("d") or REGISTRY[case["dataset"]]["expected_d"]))
    if tuple(centres.shape) != expected_shape:
        raise ConfigSkip(f"shared initial centers have shape {centres.shape}, expected {expected_shape}")
    if not np.isfinite(centres).all():
        raise ConfigSkip("shared initial centers contain NaN or infinity")
    expected_sha = metadata.get("output_sha256")
    observed_sha = file_sha256(path)
    if not isinstance(expected_sha, str) or expected_sha != observed_sha:
        raise ConfigSkip("shared initial centers SHA256 does not match its metadata")
    checks = {
        "case_id": case["id"], "run_seed": int(data_seed),
        "initialization_seed": int(init_seed), "n": int(case["n"]),
        "d": expected_shape[1], "k": expected_shape[0],
    }
    mismatches = {key: {"expected": expected, "observed": metadata.get(key)}
                  for key, expected in checks.items() if metadata.get(key) != expected}
    if mismatches:
        raise ConfigSkip(f"shared initial centers metadata mismatch: {mismatches}")
    return {
        "centers": centres, "file": str(path), "metadata_file": str(metadata_path),
        "metadata": metadata, "identity": {**file_identity(path), "sha256": observed_sha},
    }


def run_one(config: dict, config_path: Path, case_index: int, algorithm: str, seed: int, data_root: Path,
            output_dir: Path, hardware: str, context: dict, super_runner: Path | None,
            *, data_seed: int | None = None, init_seed: int | None = None,
            repeat_id: Any = None, order_block: Any = None, order_index: Any = None,
            warmup_state: Any = None) -> dict:
    case = config["cases"][case_index]
    data_seed = int(seed if data_seed is None else data_seed)
    init_seed = int(seed if init_seed is None else init_seed)
    d = int(case.get("d") or REGISTRY[case["dataset"]]["expected_d"])
    row = make_row(case, algorithm, d=d, hardware=hardware, gpu_index=0, context=context)
    row.update(seed=seed, data_seed=data_seed, init_seed=init_seed, repeat_id=repeat_id,
               order_block=order_block, order_index=order_index, warmup_state=warmup_state)
    started = time.perf_counter()
    t_input_load: float | None = None
    t_algorithm_wall: float | None = None
    t_centroid_write: float | None = None
    try:
        plugin = ALGORITHMS[algorithm]
        row["algorithm_config"] = {
            "max_iter": int(case["max_iter"]), "options": algorithm_options(config, algorithm),
        }
        initialization = resolve_initialization(config, case, data_seed, init_seed, config_path)
        if initialization is not None and not plugin.supports_shared_initialization:
            raise ConfigSkip(f"{algorithm} is operational-only: no verified external-centroid entry point")
        row["initialization"] = ({
            "track": "controlled", "mode": "shared_centroids", "centers_file": initialization["file"],
            "centers_identity": initialization["identity"], "metadata": initialization["metadata"],
        } if initialization is not None else {
            "track": config.get("track"), "mode": "implementation_seeded_default", "seed": init_seed,
        })
        if plugin.uses_python_matrix:
            input_started = time.perf_counter()
            x, data_info = load_case_data(case, data_root, data_seed)
            t_input_load = time.perf_counter() - input_started
            row["metric"] = data_info["metric"]
            row["data_fingerprint"] = data_info["data_fingerprint"]
        else:
            x = None
        algorithm_started = time.perf_counter()
        if os.environ.get("KMEANS_PARENT_RESOURCE_MONITOR") == "1":
            result = plugin.runner(x=x, case=case, seed=init_seed, data_seed=data_seed,
                                   initialization=initialization, config=config,
                                   data_root=data_root, output_dir=output_dir, super_runner=super_runner)
            memory_measurement = {
                "method": "delegated_to_parent_process_v1", "peak_delta_gib": None,
                "samples": 0, "error": None,
            }
        else:
            with GpuMemorySampler() as memory_sampler:
                result = plugin.runner(x=x, case=case, seed=init_seed, data_seed=data_seed,
                                       initialization=initialization, config=config,
                                       data_root=data_root, output_dir=output_dir, super_runner=super_runner)
            memory_measurement = memory_sampler.result()
        t_algorithm_wall = time.perf_counter() - algorithm_started
        measured_peak = memory_measurement.get("peak_delta_gib")
        plugin_peak = result.get("peak_device_memory_gib")
        try:
            measured_peak = float(measured_peak) if measured_peak is not None else None
        except (TypeError, ValueError):
            measured_peak = None
        try:
            plugin_peak = float(plugin_peak) if plugin_peak is not None else None
        except (TypeError, ValueError):
            plugin_peak = None
        # The baseline-adjusted external sampler is the only cross-framework
        # primary metric. Framework/native allocator counters remain diagnostic
        # because their semantics differ (reserved vs allocated vs whole-card).
        result["allocator_peak_device_memory_gib"] = plugin_peak
        result["peak_device_memory_gib"] = measured_peak if measured_peak is not None else plugin_peak
        result["device_memory_primary_method"] = (
            "nvidia_smi_total_used_delta_v1" if measured_peak is not None
            else "framework_or_native_fallback_noncomparable"
        )
        result["device_memory_measurement"] = memory_measurement
        if not plugin.uses_python_matrix:
            row["metric"] = "l2" if case["source"] == "synthetic" else REGISTRY[case["dataset"]]["metric"]
            row["data_fingerprint"] = result.pop("data_fingerprint")
        centroids_dir = output_dir / "centroids"
        attempt_id = stable_hash({"run_id": context["run_id"], "case": case["id"], "algorithm": algorithm,
                                  "seed": seed, "run_at": row["run_at"], "pid": os.getpid()})[:12]
        centroid_path = centroids_dir / f"{case['id']}__{algorithm}__seed{seed}__attempt{attempt_id}.npy"
        write_started = time.perf_counter()
        returned_centroids = result.pop("centroids")
        row.update(result)
        centroid_path, centroid_validation = persist_validated_centroids(
            returned_centroids,
            expected_shape=(int(case["k"]), d),
            valid_path=centroid_path,
            invalid_dir=output_dir / "invalid_centroids",
        )
        t_centroid_write = time.perf_counter() - write_started
        # Reaching this point means the plugin's guarded, GPU-only entry point
        # completed and exported centroids.  Keep the original default on every
        # skipped/error row so a claimed GPU result can never be inferred there.
        row["verdict"] = "GPU_GENUINE"
        row["centroids_file"] = str(centroid_path)
        row["centroids_identity"] = {**file_identity(centroid_path), "sha256": file_sha256(centroid_path)}
        row["centroid_validation"] = centroid_validation
        row["status"] = "OK"
    except InvalidCentroidOutput as exc:
        row["status"] = "ERROR"
        row["failure_kind"] = "invalid_algorithm_output"
        row["failure_code"] = "NONFINITE_OR_MALFORMED_CENTROIDS"
        row["candidate_attribution"] = "PENDING_AUDIT"
        row["retry_disposition"] = "NO_AUTOMATIC_RETRY"
        row["invalid_centroid_diagnostics"] = exc.diagnostics
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=4)
    except ConfigSkip as exc:
        row["status"] = "SKIPPED_CONFIG"
        row["failure_kind"] = "configuration"
        row["error"] = str(exc)
        row.update(exc.taxonomy)
    except Exception as exc:
        row["status"] = exception_status(exc)
        row["failure_kind"] = "out_of_memory" if row["status"] == "OOM" else "implementation_error"
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc(limit=4)
    row["t_total"] = time.perf_counter() - started
    row["t_input_load"] = t_input_load
    row["t_algorithm_wall"] = t_algorithm_wall
    row["t_candidate_e2e"] = t_algorithm_wall
    row["t_centroid_write"] = t_centroid_write
    row["peak_host_rss_gib"] = process_peak_rss_gib()
    if row["status"] != "OK":
        row["elapsed_to_failure_seconds"] = row["t_total"]
    return row


def append_jsonl(path: Path, row: dict) -> None:
    errors = validate_result_row(row)
    if errors:
        raise ValueError(f"refusing to append contract-invalid result row: {errors}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def print_case_outcome(row: dict) -> None:
    """Echo a compact outcome while retaining the complete JSONL as evidence."""
    print(json.dumps({
        "case": row.get("case_id"), "algorithm": row.get("algorithm"), "seed": row.get("seed"),
        "status": row.get("status"), "t_total": row.get("t_total"), "error": row.get("error"),
    }, ensure_ascii=False))


def parse_child_output(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value:
            return value
    raise ValueError(f"child returned no JSON result; last output: {stdout[-1000:]}")


def completed_case_keys(path: Path, *, terminal: bool = False) -> set[tuple[str, str, int]]:
    """Return recorded task keys eligible for skip logic.

    The interactive runner retains the old default (only ``OK`` is skipped),
    because a researcher may deliberately want to retry a failed case.  The
    unattended pipeline instead requests ``terminal=True``: an implementation
    error, timeout or configuration skip becomes an auditable terminal record
    and the queue advances to the next task.  ``SKIPPED_SAFETY`` is never
    terminal, so it always returns to admission polling rather than silently
    losing a GPU experiment.
    """
    if not path.exists():
        return set()
    completed: set[tuple[str, str, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            statuses = {"OK"}
            if terminal:
                statuses.update({"ERROR", "OOM", "TIMEOUT", "SKIPPED_CONFIG"})
            if row.get("status") in statuses and row.get("case_sha256") and row.get("algorithm") and row.get("seed") is not None:
                completed.add((str(row["case_sha256"]), str(row["algorithm"]), int(row["seed"])))
    return completed


def validate_runner_config(config: dict) -> list[str]:
    errors = validate_config(config)
    try:
        run_specs(config)
    except ValueError as exc:
        errors.append(str(exc))
    for index, case in enumerate(config.get("cases", [])):
        for algorithm in case.get("algorithms", []):
            if algorithm not in ALGORITHMS:
                errors.append(f"cases[{index}] uses unknown algorithm {algorithm!r}")
    if config.get("track") == "controlled":
        init = config.get("initialization")
        if not isinstance(init, dict) or init.get("policy") != "shared_forgy_v1":
            errors.append("controlled track requires initialization.policy='shared_forgy_v1'")
    return errors


def run_parent(args: argparse.Namespace) -> int:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config_errors = validate_runner_config(config)
    if config_errors:
        raise ValueError(f"invalid experiment config: {config_errors}")
    cfg_hash = config_sha256(args.config)
    run_id = args.run_id or f"{config['experiment_name']}__{utc_now().replace(':', '').replace('+00:00', 'Z')}"
    context = make_run_context(config, cfg_hash, run_id)
    safety = config["safety"]
    output_dir = args.output_dir / config["experiment_name"]
    result_path = output_dir / "results.jsonl"
    scheduled = []
    for idx, case in enumerate(config["cases"]):
        if case.get("preflight_ready") is False:
            continue
        if not case.get("enabled", False) and not args.include_disabled:
            continue
        algorithms = [algorithm for algorithm in case["algorithms"]
                      if not args.algorithm or algorithm == args.algorithm]
        estimate = estimate_memory_gib(case, float(safety["workspace_multiplier"]))
        specs = run_specs(config)
        if config.get("schedule_policy") == "balanced_rotation_v1" and algorithms:
            for repeat_index, raw_spec in enumerate(specs):
                offset = repeat_index % len(algorithms)
                ordered = algorithms[offset:] + algorithms[:offset]
                for order_index, algorithm in enumerate(ordered):
                    spec = dict(raw_spec)
                    spec["order_block"] = spec.get("order_block") or f"{case['id']}__repeat{repeat_index}"
                    spec["order_index"] = order_index
                    scheduled.append((idx, case, algorithm, spec, estimate))
        else:
            for algorithm in algorithms:
                for spec in specs:
                    scheduled.append((idx, case, algorithm, spec, estimate))
    if args.skip_existing or args.skip_terminal:
        completed = completed_case_keys(result_path, terminal=args.skip_terminal)
        scheduled = [item for item in scheduled
                     if (stable_hash(item[1]), item[2], item[3]["run_seed"]) not in completed]
    if args.limit:
        scheduled = scheduled[:args.limit]
    plan = [{"case": case["id"], "algorithm": algo, **spec, "estimate": estimate}
            for _, case, algo, spec, estimate in scheduled]
    print(json.dumps({"mode": "execute" if args.execute else "dry_run", "scheduled": plan}, ensure_ascii=False, indent=2))
    if not args.execute:
        return 0

    for idx, case, algorithm, spec, estimate in scheduled:
        seed = int(spec["run_seed"])
        data_seed = int(spec["data_seed"])
        init_seed = int(spec["init_seed"])
        report = admission_report(args.gpu, min_free_fraction=float(safety["min_free_memory_fraction"]),
                                  max_existing_util=int(safety["max_existing_gpu_util_percent"]),
                                  refuse_if_busy=bool(safety["refuse_if_compute_process_exists"]))
        case_context = {
            **context, "hardware_fingerprint": {"gpu": report["selected_gpu"], "physical_gpu_index": args.gpu},
            "environment": report["environment_fingerprint"],
        }
        planned = planned_case_metadata(case, args.data_root, data_seed)
        row = make_row(case, algorithm, d=planned["d"], hardware=report["selected_gpu"]["name"],
                       gpu_index=args.gpu, context=case_context)
        row.update(seed=seed, data_seed=data_seed, init_seed=init_seed,
                   repeat_id=spec.get("repeat_id"), order_block=spec.get("order_block"),
                   order_index=spec.get("order_index"), warmup_state=spec.get("warmup_state"),
                   metric=planned["metric"], data_fingerprint=planned["data_fingerprint"],
                   algorithm_config={"max_iter": int(case["max_iter"]),
                                     "options": algorithm_options(config, algorithm)},
                   initialization={"track": config.get("track"), "mode": "planned_shared_centroids"
                                   if config.get("track") == "controlled" else "implementation_seeded_default",
                                   "data_seed": data_seed, "init_seed": init_seed},
                   memory_estimate=estimate, gpu_admission=report)
        allowed_gib = report["selected_gpu"]["memory_free_mib"] / 1024 * float(safety["max_case_fraction_of_free_memory"])
        if not report["safe_to_start"]:
            row.update(status="SKIPPED_SAFETY", error="; ".join(report["blockers"]))
            append_jsonl(result_path, row)
            print_case_outcome(row)
            continue
        if estimate["estimated_peak_gib"] > allowed_gib:
            row.update(status="SKIPPED_SAFETY", error=f"estimated peak {estimate['estimated_peak_gib']} GiB > allowed {allowed_gib:.2f} GiB")
            append_jsonl(result_path, row)
            print_case_outcome(row)
            continue
        command = [sys.executable, str(Path(__file__).resolve()), "--run-one", str(idx), "--algorithm", algorithm,
                   "--seed", str(seed), "--config", str(args.config), "--data-root", str(args.data_root),
                   "--registry", str(args.registry),
                   "--output-dir", str(output_dir), "--hardware", report["selected_gpu"]["name"],
                   "--run-id", run_id, "--config-sha256", cfg_hash,
                   "--data-seed", str(data_seed), "--init-seed", str(init_seed),
                   "--repeat-id-json", json.dumps(spec.get("repeat_id")),
                   "--order-block-json", json.dumps(spec.get("order_block")),
                   "--order-index-json", json.dumps(spec.get("order_index")),
                   "--warmup-state-json", json.dumps(spec.get("warmup_state"))]
        if args.super_runner:
            command.extend(["--super-runner", str(args.super_runner)])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  # prohibit accidental multi-GPU paths
        env["KMEANS_PARENT_RESOURCE_MONITOR"] = "1"
        process: subprocess.Popen[str] | None = None
        tree_sampler: ProcessTreeRssSampler | None = None
        tree_measurement: dict[str, Any] | None = None
        gpu_sampler: GpuMemorySampler | None = None
        gpu_measurement: dict[str, Any] | None = None
        stdout = ""
        stderr = ""
        dispatch_started = time.perf_counter()
        try:
            with GpuMemorySampler(gpu=args.gpu) as gpu_sampler:
                process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           env=env, start_new_session=True)
                with ProcessTreeRssSampler(process.pid) as tree_sampler:
                    try:
                        stdout, stderr = process.communicate(timeout=int(safety["case_timeout_sec"]))
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.communicate()
                        raise
            tree_measurement = tree_sampler.result()
            gpu_measurement = gpu_sampler.result()
            child_row = parse_child_output(stdout)
            child_memory_measurement = child_row.get("device_memory_measurement")
            measured_device_peak = gpu_measurement.get("peak_delta_gib")
            child_row.update(gpu_index=args.gpu, memory_estimate=estimate,
                             hardware_fingerprint=case_context["hardware_fingerprint"],
                             environment=case_context["environment"], gpu_admission=report,
                             process_tree_peak_host_rss_gib=tree_measurement.get("peak_rss_gib"),
                             process_tree_rss_measurement=tree_measurement,
                             child_device_memory_measurement=child_memory_measurement,
                             device_memory_measurement=gpu_measurement,
                             peak_device_memory_gib=(measured_device_peak if measured_device_peak is not None
                                                     else child_row.get("peak_device_memory_gib")),
                             device_memory_primary_method=("nvidia_smi_total_used_delta_v1_parent"
                                                           if measured_device_peak is not None
                                                           else child_row.get("device_memory_primary_method")))
            if process.returncode != 0 and child_row["status"] == "OK":
                child_row.update(status="ERROR", failure_kind="child_exit",
                                 elapsed_to_failure_seconds=child_row.get("t_total"),
                                 error=f"child exited {process.returncode}: {stderr[-500:]}")
            append_jsonl(result_path, child_row)
            print_case_outcome(child_row)
        except subprocess.TimeoutExpired:
            if tree_sampler is not None:
                tree_measurement = tree_sampler.result()
            if gpu_sampler is not None:
                gpu_measurement = gpu_sampler.result()
            row.update(status="TIMEOUT", failure_kind="timeout",
                       t_total=float(safety["case_timeout_sec"]),
                       elapsed_to_failure_seconds=float(safety["case_timeout_sec"]),
                       process_tree_peak_host_rss_gib=(tree_measurement or {}).get("peak_rss_gib"),
                       process_tree_rss_measurement=tree_measurement,
                       peak_device_memory_gib=(gpu_measurement or {}).get("peak_delta_gib"),
                       device_memory_measurement=gpu_measurement,
                       device_memory_primary_method="nvidia_smi_total_used_delta_v1_parent",
                       error=f"exceeded {safety['case_timeout_sec']} seconds")
            append_jsonl(result_path, row)
            print_case_outcome(row)
        except Exception as exc:
            # The unattended dispatcher must be able to advance after a
            # malformed child response.  Preserve the full failure evidence as
            # an append-only terminal row instead of crashing before a result
            # is recorded and selecting the same case forever.
            status = exception_status(RuntimeError(f"{exc}; {stderr[-1000:]}"))
            elapsed = time.perf_counter() - dispatch_started
            if tree_sampler is not None:
                tree_measurement = tree_sampler.result()
            if gpu_sampler is not None:
                gpu_measurement = gpu_sampler.result()
            row.update(status=status,
                       failure_kind="out_of_memory" if status == "OOM" else "dispatcher_error",
                       t_total=elapsed, elapsed_to_failure_seconds=elapsed,
                       process_tree_peak_host_rss_gib=(tree_measurement or {}).get("peak_rss_gib"),
                       process_tree_rss_measurement=tree_measurement,
                       peak_device_memory_gib=(gpu_measurement or {}).get("peak_delta_gib"),
                       device_memory_measurement=gpu_measurement,
                       device_memory_primary_method="nvidia_smi_total_used_delta_v1_parent",
                       error=f"{type(exc).__name__}: {exc}")
            append_jsonl(result_path, row)
            print_case_outcome(row)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH,
                        help="dataset registry JSON (passed unchanged to child processes)")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--gpu", type=int, default=0, help="physical GPU index for parent admission")
    parser.add_argument("--execute", action="store_true", help="required to launch a GPU workload")
    parser.add_argument("--include-disabled", action="store_true", help="explicitly opt into disabled cases")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true", help="do not rerun a case/algorithm/seed already recorded as OK")
    parser.add_argument("--skip-terminal", action="store_true",
                        help="also skip recorded ERROR/TIMEOUT/SKIPPED_CONFIG rows; intended for unattended queues")
    parser.add_argument("--super-runner", type=Path, help="native gpu_generic_superkmeans.out executable")
    parser.add_argument("--run-one", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS))
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--data-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--init-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat-id-json", help=argparse.SUPPRESS)
    parser.add_argument("--order-block-json", help=argparse.SUPPRESS)
    parser.add_argument("--order-index-json", help=argparse.SUPPRESS)
    parser.add_argument("--warmup-state-json", help=argparse.SUPPRESS)
    parser.add_argument("--hardware", default="unknown")
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--config-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure_registry(args.registry)
    args.registry = args.registry.resolve()
    if args.run_one is not None:
        if not args.algorithm or args.seed is None:
            parser.error("--run-one requires --algorithm and --seed")
        config = json.loads(args.config.read_text(encoding="utf-8"))
        errors = validate_runner_config(config)
        if errors:
            parser.error(f"invalid experiment config: {errors}")
        cfg_hash = args.config_sha256 or config_sha256(args.config)
        run_id = args.run_id or f"{config['experiment_name']}__direct_child"
        context = make_run_context(config, cfg_hash, run_id)
        row = run_one(config, args.config, args.run_one, args.algorithm, args.seed, args.data_root,
                      args.output_dir, args.hardware, context, args.super_runner,
                      data_seed=args.data_seed, init_seed=args.init_seed,
                      repeat_id=json.loads(args.repeat_id_json) if args.repeat_id_json is not None else None,
                      order_block=json.loads(args.order_block_json) if args.order_block_json is not None else None,
                      order_index=json.loads(args.order_index_json) if args.order_index_json is not None else None,
                      warmup_state=json.loads(args.warmup_state_json) if args.warmup_state_json is not None else None)
        print(json.dumps(row, ensure_ascii=False, default=float))
        return 0 if row["status"] == "OK" else 1
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
