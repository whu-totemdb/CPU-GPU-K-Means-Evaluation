"""Canonical CPU--GPU comparison identities for the unified RQ1/RQ2 study.

Raw result rows remain append-only and intentionally keep their runner-specific
provenance.  This module creates *derived* identities that make device and
compute precision candidate attributes rather than task attributes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from result_contract import stable_hash


# A safety skip reflects transient shared-GPU availability, not an algorithm
# outcome. It must be retried and cannot satisfy a paper-matrix cell.
TERMINAL_STATUSES = frozenset({"OK", "ERROR", "OOM", "TIMEOUT", "SKIPPED_CONFIG"})


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_version") != "unified_candidate_manifest_v1":
        raise ValueError("unsupported unified candidate manifest")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("manifest needs a non-empty candidates list")
    ids = [item.get("candidate_id") for item in candidates]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("candidate_id values must be non-empty and unique")
    return payload


def _options(raw: dict[str, Any]) -> dict[str, Any]:
    config = raw.get("algorithm_config", {})
    result = dict(config.get("options", {})) if isinstance(config, dict) and isinstance(config.get("options", {}), dict) else {}
    hardware = raw.get("hardware_fingerprint", {})
    if isinstance(hardware, dict) and hardware.get("cpu_threads") is not None:
        result.setdefault("threads", hardware["cpu_threads"])
    return result


def _matches(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    return all(observed.get(key) == value for key, value in expected.items())


def candidate_for_raw(raw: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any] | None:
    """Map a raw runner row to exactly one frozen candidate descriptor."""
    raw_algorithm = raw.get("algorithm")
    device = raw.get("device")
    dtype = raw.get("dtype")
    matches: list[dict[str, Any]] = []
    for candidate in manifest["candidates"]:
        if candidate.get("raw_algorithm") != raw_algorithm or candidate.get("device") != device:
            continue
        expected_dtype = candidate.get("compute_dtype")
        if expected_dtype and expected_dtype != dtype:
            # Raw ``dtype`` describes the runner's input/result representation
            # and is historically fp32 for every CPU runner.  A binding may
            # nevertheless declare an explicitly recorded higher internal
            # compute precision (mlpack's Armadillo ``arma::mat`` is fp64).
            # Permit that narrow, auditable distinction without rewriting
            # frozen raw rows or mislabelling the candidate in the manifest.
            declared_compute = _options(raw).get("compute_precision")
            if expected_dtype != declared_compute:
                continue
        expected_options = candidate.get("options_match", {})
        observed_options = _options(raw)
        if not _matches(expected_options, observed_options):
            # Historical parent-produced terminal records can predate normal
            # child provenance. Recover them only when the frozen structural
            # descriptor still maps uniquely; never guess between candidates.
            terminal_missing_options = raw.get("status") in TERMINAL_STATUSES and not observed_options
            if not terminal_missing_options:
                continue
        matches.append(candidate)
    if len(matches) > 1:
        raise ValueError(f"ambiguous candidate mapping for {raw_algorithm!r}/{device!r}/{dtype!r}")
    return matches[0] if matches else None


def sampling_descriptor(raw: dict[str, Any]) -> dict[str, Any]:
    """Return representation-independent data slicing provenance."""
    fingerprint = raw.get("data_fingerprint", {}) if isinstance(raw.get("data_fingerprint"), dict) else {}
    prepared = fingerprint.get("prepared", {}) if isinstance(fingerprint.get("prepared"), dict) else {}
    sampling = fingerprint.get("sampling") or prepared.get("sampling")
    if isinstance(sampling, dict):
        return {key: sampling.get(key) for key in ("method", "seed", "requested_rows", "source_train_rows")}
    generator = (fingerprint.get("generator_spec") or fingerprint.get("generator")
                 or prepared.get("generator_spec") or prepared.get("generator"))
    if isinstance(generator, dict):
        generator = generator.get("generator")
    data_seed = raw.get("data_seed") if raw.get("data_seed") is not None else raw.get("seed")
    return {"method": "synthetic_generator" if generator else "unknown", "seed": data_seed, "generator": generator}


def task_descriptor(raw: dict[str, Any]) -> dict[str, Any]:
    """Fields shared by CPU/GPU candidates; excludes device and compute dtype."""
    fingerprint = raw.get("data_fingerprint", {}) if isinstance(raw.get("data_fingerprint"), dict) else {}
    prepared = fingerprint.get("prepared", {}) if isinstance(fingerprint.get("prepared"), dict) else {}
    config = raw.get("algorithm_config", {}) if isinstance(raw.get("algorithm_config"), dict) else {}
    is_synthetic = raw.get("source") == "synthetic" or raw.get("dataset") == "synthetic"
    normalized = (False if is_synthetic else
                  fingerprint.get("normalization_applied", fingerprint.get("normalized", prepared.get("normalized"))))
    return {
        "dataset": raw.get("dataset"), "n": raw.get("n"), "d": raw.get("d"), "k": raw.get("k"),
        "metric": raw.get("metric"), "normalized": normalized,
        "source": "synthetic" if is_synthetic else raw.get("source", "dataset"),
        "max_iter": config.get("max_iter"), "track": raw.get("track"),
        "scenario": "host_resident_cold_start_v1", "input_dtype": "fp32",
    }


def task_spec_id(raw: dict[str, Any]) -> str:
    return stable_hash(task_descriptor(raw))


def data_slice_id(raw: dict[str, Any]) -> str:
    return stable_hash({"task": task_descriptor(raw), "sampling": sampling_descriptor(raw)})


def platform_id(manifest: dict[str, Any]) -> str:
    value = manifest.get("platform", {}).get("platform_id")
    if not isinstance(value, str) or not value:
        raise ValueError("manifest.platform.platform_id is required")
    return value


def terminal(row: dict[str, Any]) -> bool:
    return row.get("status") in TERMINAL_STATUSES
