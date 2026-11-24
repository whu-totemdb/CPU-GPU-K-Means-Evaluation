#!/usr/bin/env python3
"""Stream a deterministic K-Means input into SuperKMeans' raw-f32 format.

The script deliberately runs as a separate process.  It never materializes the
entire selected HDF5 subset: only ``chunk_rows × d`` float32 values are live at
once, then the parent runner can release this process before starting CUDA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterator


def stable_hash(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_identity(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def sampled_indices(total: int, n: int, seed: int):
    """Return the exact sorted sampling rule used by ``run_experiments.py``."""
    import numpy as np

    if n <= 0 or n > total:
        raise ValueError(f"n must be in [1, {total}], got {n}")
    if n == total:
        return np.arange(total, dtype=np.int64)
    return np.sort(np.random.default_rng(seed).choice(total, n, replace=False))


def normalise_rows(matrix) -> None:
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, 1e-12)


def iter_hdf5_subset(path: Path, *, n: int, seed: int, prefix_rows: bool, chunk_rows: int) -> tuple[int, int, object, Iterator[object]]:
    """Return shape metadata and a generator that closes the HDF5 file itself."""
    import h5py

    with h5py.File(path, "r") as handle:
        if "train" not in handle:
            raise KeyError(f"{path} has no train array")
        total, d = handle["train"].shape
    import numpy as np
    indices = np.arange(n, dtype=np.int64) if prefix_rows else sampled_indices(int(total), n, seed)

    def chunks() -> Iterator[object]:
        import h5py
        import numpy as np

        with h5py.File(path, "r") as handle:
            train = handle["train"]
            for start in range(0, n, chunk_rows):
                rows = indices[start:start + chunk_rows]
                yield np.ascontiguousarray(train[rows], dtype=np.float32)

    return int(d), int(total), indices, chunks()


def write_hdf5_subset(
    *, input_path: Path, output_path: Path, n: int, seed: int, prefix_rows: bool, normalize: bool, chunk_rows: int
) -> dict:
    """Write a row-major f32 sample, bounded by ``chunk_rows × d × 4`` bytes."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    d, total, indices, chunks = iter_hdf5_subset(input_path, n=n, seed=seed, prefix_rows=prefix_rows, chunk_rows=chunk_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output_path.open("wb") as output:
        for matrix in chunks:
            if normalize:
                normalise_rows(matrix)
            payload = matrix.tobytes(order="C")
            output.write(payload)
            digest.update(payload)
    expected_size = n * d * 4
    if output_path.stat().st_size != expected_size:
        raise RuntimeError(f"wrote {output_path.stat().st_size} bytes, expected {expected_size}")
    sampling = {
        "method": "prefix_first_n" if prefix_rows else ("all_rows" if n == total else "sorted_choice_without_replacement"),
        "seed": None if prefix_rows else int(seed), "requested_rows": int(n), "source_train_rows": int(total),
    }
    # The stable sampling signature, rather than the full index vector, is enough
    # to reproduce the subset without storing a large array in every JSON row.
    return {
        "input": str(input_path), "input_identity": file_identity(input_path),
        "output": str(output_path), "n": int(n), "d": int(d), "seed": int(seed),
        "normalized": bool(normalize), "dtype": "float32", "layout": "row_major",
        "chunk_rows": int(chunk_rows), "sampling": sampling,
        "sampling_signature_sha256": stable_hash({
            "method": sampling["method"], "n": n, "seed": sampling["seed"], "source_train_rows": total,
            "rng": None if prefix_rows else "numpy.default_rng.choice_without_replacement.v1",
        }),
        "output_sha256": digest.hexdigest(),
    }


def write_synthetic_subset(
    *, output_path: Path, n: int, d: int, seed: int, chunk_rows: int
) -> dict:
    """Stream the documented synthetic generator without allocating ``n × d``."""
    import numpy as np

    if n <= 0 or d <= 0 or chunk_rows <= 0:
        raise ValueError("n, d and chunk_rows must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    with output_path.open("wb") as output:
        for start in range(0, n, chunk_rows):
            matrix = rng.standard_normal((min(chunk_rows, n - start), d), dtype=np.float32)
            payload = matrix.tobytes(order="C")
            output.write(payload)
            digest.update(payload)
    return {
        "input": None, "output": str(output_path), "n": int(n), "d": int(d), "seed": int(seed),
        "normalized": False, "dtype": "float32", "layout": "row_major", "chunk_rows": int(chunk_rows),
        "generator": "numpy.default_rng.standard_normal.float32.v1",
        "output_sha256": digest.hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="ANN-Benchmarks HDF5 file containing train")
    source.add_argument("--synthetic-d", type=int, help="stream deterministic Gaussian synthetic data")
    parser.add_argument("--output", required=True, type=Path, help="output row-major float32 .f32 file")
    parser.add_argument("--n", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--prefix-rows", action="store_true", help="select the first n HDF5 train rows")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize HDF5 rows before writing")
    parser.add_argument("--chunk-rows", type=int, default=2048, help="bounded host working set in rows")
    args = parser.parse_args()

    if args.input:
        metadata = write_hdf5_subset(
            input_path=args.input, output_path=args.output, n=args.n, seed=args.seed, prefix_rows=args.prefix_rows,
            normalize=args.normalize, chunk_rows=args.chunk_rows,
        )
    else:
        metadata = write_synthetic_subset(
            output_path=args.output, n=args.n, d=args.synthetic_d, seed=args.seed, chunk_rows=args.chunk_rows,
        )
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
