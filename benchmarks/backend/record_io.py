"""Record I/O and identity used by the quality evaluator."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from result_contract import stable_hash


def read_jsonl(paths: Iterable[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number} is not a JSON object")
                    rows.append(value)
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def result_key(row: dict[str, Any]) -> str:
    """Stable join key for one algorithm/case/seed result, independent of paths."""
    return stable_hash({
        "run_id": row.get("run_id"), "case_id": row.get("case_id"),
        "algorithm": row.get("algorithm"), "seed": row.get("seed"),
        "case_sha256": row.get("case_sha256"), "centroids_file": row.get("centroids_file"),
    })
