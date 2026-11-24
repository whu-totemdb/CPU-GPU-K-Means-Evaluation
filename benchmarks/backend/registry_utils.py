"""Immutable provenance helpers for selectable dataset registry files.

Paths and timestamps are retained for diagnostics, but cross-stage consistency
is deliberately based on registry content and the selected entry content.
"""
from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path
from typing import Any

from result_contract import stable_hash


SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def load_dataset_registry(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load one registry and return datasets plus diagnostic artifact evidence."""
    path = path.resolve()
    try:
        with path.open("rb") as handle:
            content = handle.read()
            stat = Path(handle.name).stat()
        payload = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load dataset registry {path}: {exc}") from exc
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if not isinstance(datasets, dict):
        raise ValueError(f"dataset registry {path} must contain a datasets object")
    for key, value in datasets.items():
        if not isinstance(key, str) or not key or not isinstance(value, dict):
            raise ValueError(f"dataset registry {path} has an invalid entry {key!r}")
    artifact = {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    return datasets, artifact


def registry_reference(
    datasets: dict[str, dict[str, Any]], artifact: dict[str, Any], dataset: str | None
) -> dict[str, Any]:
    """Build the common ``dataset_registry_evidence_v2`` stage contract."""
    if dataset in (None, "synthetic"):
        return {
            "contract_version": "dataset_registry_evidence_v2",
            "registry_artifact": dict(artifact),
            "dataset_id": "synthetic",
            "dataset_entry_sha256": None,
            "dataset_entry_state": {
                "sha256_status": "NOT_APPLICABLE",
                "ground_truth_status": "NOT_APPLICABLE",
                "license_status": "NOT_APPLICABLE",
            },
        }
    if dataset not in datasets:
        raise KeyError(f"dataset {dataset!r} is absent from registry {artifact.get('path')}")
    spec = datasets[dataset]
    return {
        "contract_version": "dataset_registry_evidence_v2",
        "registry_artifact": dict(artifact),
        "dataset_id": dataset,
        "dataset_entry_sha256": stable_hash(spec),
        "dataset_entry_state": {
            "sha256_status": spec.get("sha256_status") or "UNDECLARED",
            "ground_truth_status": spec.get("ground_truth_status") or "UNVERIFIED",
            "license_status": spec.get("license_status") or "UNDECLARED",
        },
    }


def same_registry_reference(left: Any, right: Any) -> bool:
    """Compare registry identities without invalidating registry-free synthetic data.

    Real datasets retain the strict whole-registry and selected-entry hashes.
    Synthetic rows do not read a registry entry, so an unrelated registry-file
    update must not invalidate their downstream quality/profile linkage.  Both
    sides must still carry a well-formed v2 artifact reference and the exact
    NOT_APPLICABLE entry contract.
    """
    # ``dataset_profile._row_registry_reference`` returns
    # ``(reference, compatibility_reason)`` in the v2 derived-recovery path.
    # Accept that enriched return value at legacy comparison call sites while
    # keeping the actual registry contract dictionary-only below.
    if isinstance(left, tuple) and len(left) == 2 and isinstance(left[0], dict):
        left = left[0]
    if isinstance(right, tuple) and len(right) == 2 and isinstance(right[0], dict):
        right = right[0]
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if (
        left.get("contract_version") != "dataset_registry_evidence_v2"
        or right.get("contract_version") != "dataset_registry_evidence_v2"
        or left.get("dataset_id") != right.get("dataset_id")
    ):
        return False
    left_artifact = left.get("registry_artifact")
    right_artifact = right.get("registry_artifact")
    if not isinstance(left_artifact, dict) or not isinstance(right_artifact, dict):
        return False
    left_sha = left_artifact.get("sha256")
    right_sha = right_artifact.get("sha256")
    if (
        not isinstance(left_sha, str)
        or not isinstance(right_sha, str)
        or SHA256_HEX.fullmatch(left_sha) is None
        or SHA256_HEX.fullmatch(right_sha) is None
    ):
        return False
    if left.get("dataset_id") == "synthetic":
        not_applicable = {
            "sha256_status": "NOT_APPLICABLE",
            "ground_truth_status": "NOT_APPLICABLE",
            "license_status": "NOT_APPLICABLE",
        }
        return (
            left.get("dataset_entry_sha256") is None
            and right.get("dataset_entry_sha256") is None
            and left.get("dataset_entry_state") == not_applicable
            and right.get("dataset_entry_state") == not_applicable
        )
    return (
        left_sha == right_sha
        and left.get("dataset_entry_sha256") == right.get("dataset_entry_sha256")
    )


def synthetic_registry_reference_compatible(value: Any) -> bool:
    """Recognise a registry-free synthetic evidence record from either contract era.

    Controlled synthetic rows do not load a dataset-registry entry.  A later
    addition of real datasets to the whole registry therefore cannot change
    their generated matrix.  We still require a well-formed automatically
    recorded registry artifact and reject any value that purports to bind a
    selected real-dataset entry.  This helper is deliberately *not* used for
    real datasets, whose artifact and selected-entry identities stay strict.
    """
    if not isinstance(value, dict):
        return False
    if value.get("contract_version") != "dataset_registry_evidence_v2":
        return False
    if value.get("dataset_id") != "synthetic":
        return False
    # Controlled synthetic input never consumes a selected registry entry, but
    # its versioned evidence envelope and automatic artifact identity remain
    # mandatory.  Only the *global* artifact SHA may differ after unrelated
    # registry edits; selected real-data entries remain strictly checked.
    if value.get("dataset_entry_sha256") not in (None, ""):
        return False
    entry_state = value.get("dataset_entry_state")
    if entry_state != {
        "sha256_status": "NOT_APPLICABLE",
        "ground_truth_status": "NOT_APPLICABLE",
        "license_status": "NOT_APPLICABLE",
    }:
        return False
    artifact = value.get("registry_artifact")
    if not isinstance(artifact, dict):
        return False
    digest = artifact.get("sha256")
    if not isinstance(digest, str) or SHA256_HEX.fullmatch(digest) is None:
        return False
    return True
