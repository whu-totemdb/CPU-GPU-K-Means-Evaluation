"""Deterministic synthetic generators for controlled K-Means geometry studies."""
from __future__ import annotations

import copy
from typing import Any


VERSION = "controlled_kmeans_geometry_v1"


def normalized_spec(case: dict[str, Any]) -> dict[str, Any] | None:
    raw = case.get("synthetic_generator")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("synthetic_generator must be an object")
    spec = copy.deepcopy(raw)
    if spec.get("version") != VERSION:
        raise ValueError(f"synthetic_generator.version must be {VERSION!r}")
    defaults = {
        "true_clusters": int(case["k"]),
        "effective_rank": int(case["d"]),
        "separation": 4.0,
        "axis_scale": 1.0,
        "sparsity": 0.0,
        "norm_log_std": 0.0,
        "duplicate_fraction": 0.0,
        "imbalance_ratio": 1.0,
        "normalize": False,
        "metric": "l2",
    }
    for key, value in defaults.items():
        spec.setdefault(key, value)
    n, d = int(case["n"]), int(case["d"])
    clusters = int(spec["true_clusters"])
    rank = int(spec["effective_rank"])
    if not 1 <= clusters <= n:
        raise ValueError("synthetic_generator.true_clusters must be in [1,n]")
    if not 1 <= rank <= d:
        raise ValueError("synthetic_generator.effective_rank must be in [1,d]")
    if float(spec["separation"]) < 0 or float(spec["axis_scale"]) <= 0:
        raise ValueError("separation must be non-negative and axis_scale positive")
    for key in ("sparsity", "duplicate_fraction"):
        if not 0.0 <= float(spec[key]) < 1.0:
            raise ValueError(f"synthetic_generator.{key} must be in [0,1)")
    if float(spec["norm_log_std"]) < 0.0:
        raise ValueError("synthetic_generator.norm_log_std must be non-negative")
    if float(spec["imbalance_ratio"]) < 1.0:
        raise ValueError("synthetic_generator.imbalance_ratio must be >= 1")
    if spec["metric"] not in {"l2", "ip"}:
        raise ValueError("synthetic_generator.metric must be l2 or ip")
    return spec


def generate(case: dict[str, Any], seed: int):
    """Return a controlled float32 matrix and a frozen generator description.

    Each task changes one explicit generative intervention while preserving the
    task-scale block (n,d,k), seed policy, total latent variance, and remaining
    generator parameters.  Measured profile features can still co-move; causal
    claims therefore apply to the intervention, not to an allegedly isolated
    observed statistic.
    """
    import numpy as np

    spec = normalized_spec(case)
    if spec is None:
        raise ValueError("controlled generator requested without synthetic_generator")
    n, d = int(case["n"]), int(case["d"])
    clusters = int(spec["true_clusters"])
    rank = int(spec["effective_rank"])
    rng = np.random.default_rng(int(seed))

    ratio = float(spec["imbalance_ratio"])
    if clusters == 1 or ratio == 1.0:
        probabilities = np.full(clusters, 1.0 / clusters, dtype=np.float64)
    else:
        probabilities = ratio ** (-np.arange(clusters, dtype=np.float64) / (clusters - 1))
        probabilities /= probabilities.sum()
        probabilities = probabilities[rng.permutation(clusters)]
    labels = rng.choice(clusters, size=n, replace=True, p=probabilities)

    centers = rng.standard_normal((clusters, rank), dtype=np.float32)
    center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
    centers /= np.maximum(center_norms, 1e-12)
    latent = rng.standard_normal((n, rank), dtype=np.float32)
    latent += np.float32(spec["separation"]) * centers[labels]

    axis_scale = float(spec["axis_scale"])
    if axis_scale != 1.0:
        latent[:, 0] *= np.float32(axis_scale)
        # Preserve expected total energy while changing covariance anisotropy.
        latent *= np.float32((rank / (axis_scale ** 2 + rank - 1)) ** 0.5)

    if rank == d:
        matrix = latent
    else:
        projection_raw = rng.standard_normal((d, rank), dtype=np.float32)
        projection, _ = np.linalg.qr(projection_raw, mode="reduced")
        matrix = latent @ np.asarray(projection.T, dtype=np.float32)

    sparsity = float(spec["sparsity"])
    if sparsity > 0.0:
        keep = rng.random(matrix.shape) >= sparsity
        matrix *= keep
        matrix /= np.float32(max(1.0 - sparsity, 1e-12) ** 0.5)

    norm_log_std = float(spec["norm_log_std"])
    if norm_log_std > 0.0:
        # A log-normal radial multiplier changes row-norm dispersion while
        # preserving E[scale^2]=1, so expected total squared energy remains
        # comparable to the baseline.  Geometry statistics can still co-move.
        radial = np.exp(
            np.float32(norm_log_std) * rng.standard_normal(n, dtype=np.float32)
            - np.float32(norm_log_std ** 2)
        )
        matrix *= radial[:, None]

    duplicate_fraction = float(spec["duplicate_fraction"])
    duplicate_count = int(round(n * duplicate_fraction))
    if duplicate_count:
        targets = np.arange(n - duplicate_count, n)
        sources = rng.integers(0, max(1, n - duplicate_count), size=duplicate_count)
        matrix[targets] = matrix[sources]

    if bool(spec["normalize"]):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix /= np.maximum(norms, 1e-12)
    return np.ascontiguousarray(matrix, dtype=np.float32), {
        "generator": VERSION,
        "seed": int(seed),
        "n": n,
        "d": d,
        "parameters": spec,
        "controlled_feature": case.get("controlled_feature", "baseline"),
        "feature_level": case.get("feature_level", "baseline"),
        "control_block": case.get("control_block", "reference_scale_v1"),
    }
