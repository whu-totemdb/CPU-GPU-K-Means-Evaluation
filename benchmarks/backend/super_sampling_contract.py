"""Dependency-free SuperKMeans sampling preconditions shared by runners and audits."""
from __future__ import annotations

from typing import Any


def sampling_cardinality(*, n: int, k: int, sampling_fraction: float) -> dict[str, Any]:
    n_value = int(n)
    k_value = int(k)
    fraction = float(sampling_fraction)
    if n_value <= 0 or k_value <= 0:
        raise ValueError("n and k must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("sampling_fraction must be in (0, 1]")
    configured_sample_count = n_value * fraction
    return {
        "kind": "super_sampling_cardinality_v1",
        "n": n_value,
        "k": k_value,
        "sampling_fraction": fraction,
        "configured_sample_count": configured_sample_count,
        "required_minimum_sample_count": k_value,
        "unsupported": configured_sample_count < k_value,
    }
