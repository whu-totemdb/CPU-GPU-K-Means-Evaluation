"""Versioned, dependency-free contract for K-Means experiment records.

The contract is intentionally additive: old CPU JSONL files remain readable,
while new GPU and SeekDB records carry enough provenance to be compared only
when their environment and quality policy are compatible.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import socket
import sys
import threading
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any


CONTRACT_VERSION = "1.0.0"
RECORD_TYPE = "kmeans_experiment"
VALID_STATUSES = frozenset({"OK", "ERROR", "OOM", "TIMEOUT", "SKIPPED_SAFETY", "SKIPPED_CONFIG"})
VALID_TRACKS = frozenset({"screening", "controlled", "operational", "final_holdout", "validation"})
SHA256_LENGTH = 64


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_hash(value: Any) -> str:
    return sha256_text(canonical_json(value))


def file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a full-file SHA256. Call explicitly, never implicitly per run."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    """Cheap file identity used for each run; not a substitute for SHA256."""
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def config_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions(packages: dict[str, str] | None = None) -> dict[str, str | None]:
    packages = packages or {
        "numpy": "numpy",
        "h5py": "h5py",
        "torch": "torch",
        "flash_kmeans": "flash-kmeans",
        "faiss": "faiss-gpu",
        "cuml": "cuml",
        "cupy": "cupy-cuda12x",
    }
    versions: dict[str, str | None] = {}
    for module_name, distribution in packages.items():
        try:
            versions[module_name] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            # Some distributions use a non-standard project name. Keep the
            # absence explicit rather than importing a CUDA library here.
            versions[module_name] = None
    return versions


def cpu_model_name() -> str | None:
    """Read the Linux CPU model without invoking a command or touching CUDA."""
    cpuinfo = Path("/proc/cpuinfo")
    try:
        entries: dict[str, str] = {}
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if separator:
                entries.setdefault(key.strip().lower(), value.strip())
        # ``processor: 0`` is a CPU ordinal, not a model.  Prefer the explicit
        # model name and only use architecture labels on platforms that lack it.
        for key in ("model name", "hardware"):
            if entries.get(key):
                return entries[key]
    except OSError:
        pass
    return platform.processor() or None


def host_fingerprint() -> dict[str, Any]:
    """Stable-enough host metadata that is safe to collect without CUDA imports."""
    memory_total_bytes: int | None = None
    try:
        memory_total_bytes = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        pass
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": cpu_model_name(),
        "cpu_count_logical": os.cpu_count(),
        "memory_total_bytes": memory_total_bytes,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "packages": package_versions(),
    }


def process_peak_rss_gib() -> float | None:
    """Return this process's observed peak resident set, if the OS exposes it.

    Linux reports ``ru_maxrss`` in KiB while macOS reports bytes.  This value is
    provenance for a completed child process, not a scheduler admission input
    and therefore must never be used as a dynamic selector feature.
    """
    try:
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (AttributeError, OSError, ValueError):
        return None
    if peak < 0:
        return None
    bytes_per_unit = 1 if sys.platform == "darwin" else 1024
    return peak * bytes_per_unit / (1024 ** 3)


def _linux_process_tree_pids(root_pid: int) -> set[int]:
    """Return the live Linux process tree rooted at ``root_pid``.

    This reads only the benchmark subprocesses named by the caller.  It never
    enumerates users, commands, or unrelated processes, which keeps it suitable
    for the shared server policy used by this project.
    """
    pending = [int(root_pid)]
    observed: set[int] = set()
    while pending:
        pid = pending.pop()
        if pid in observed:
            continue
        observed.add(pid)
        children = Path(f"/proc/{pid}/task/{pid}/children")
        try:
            pending.extend(int(value) for value in children.read_text(encoding="ascii").split())
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, OSError):
            continue
    return observed


def linux_process_tree_rss_bytes(root_pid: int) -> int | None:
    """Snapshot RSS for one known benchmark process tree on Linux."""
    if not Path("/proc").is_dir():
        return None
    total_kib = 0
    found = False
    for pid in _linux_process_tree_pids(root_pid):
        try:
            lines = Path(f"/proc/{pid}/status").read_text(encoding="ascii", errors="replace").splitlines()
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        for line in lines:
            if line.startswith("VmRSS:"):
                try:
                    total_kib += int(line.split()[1])
                    found = True
                except (IndexError, ValueError):
                    pass
                break
    return total_kib * 1024 if found else None


class ProcessTreeRssSampler:
    """Poll peak RSS for a benchmark child and all of its descendants."""

    def __init__(self, root_pid: int, interval_sec: float = 0.25):
        self.root_pid = int(root_pid)
        self.interval_sec = float(interval_sec)
        self.peak_bytes: int | None = None
        self.samples = 0
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                value = linux_process_tree_rss_bytes(self.root_pid)
                if value is not None:
                    self.samples += 1
                    self.peak_bytes = value if self.peak_bytes is None else max(self.peak_bytes, value)
            except Exception as exc:  # diagnostics must not abort an experiment
                self.error = f"{type(exc).__name__}: {exc}"
                return
            self._stop.wait(self.interval_sec)

    def __enter__(self) -> "ProcessTreeRssSampler":
        try:
            value = linux_process_tree_rss_bytes(self.root_pid)
            if value is not None:
                self.samples = 1
                self.peak_bytes = value
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        self._thread = threading.Thread(target=self._poll, name="process-tree-rss-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 4))

    def result(self) -> dict[str, Any]:
        return {
            "method": "linux_proc_process_tree_rss_v1",
            "root_pid": self.root_pid,
            "interval_sec": self.interval_sec,
            "samples": self.samples,
            "peak_rss_gib": None if self.peak_bytes is None else self.peak_bytes / (1024 ** 3),
            "error": self.error,
        }


def exception_status(exc: BaseException) -> str:
    """Map a caught exception to a structured terminal status."""
    if isinstance(exc, MemoryError):
        return "OOM"
    text = f"{type(exc).__name__}: {exc}".lower()
    markers = ("out of memory", "outofmemory", "cuda_error_out_of_memory", "cublas_status_alloc_failed")
    return "OOM" if any(marker in text for marker in markers) else "ERROR"


def new_result_row(
    *,
    case: dict[str, Any],
    algorithm: str,
    d: int,
    hardware: str,
    gpu_index: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Create a complete nullable record before an experiment starts."""
    return {
        "contract_version": CONTRACT_VERSION,
        "record_type": RECORD_TYPE,
        "run_at": utc_now(),
        "run_id": context["run_id"],
        "experiment_name": context["experiment_name"],
        "track": context["track"],
        "config_sha256": context["config_sha256"],
        "case_sha256": stable_hash(case),
        "quality_policy": context["quality_policy"],
        "case_id": case["id"],
        "tier": "gpu_highdim",
        "dataset": case.get("dataset", "synthetic"),
        "algorithm": algorithm,
        "n": int(case["n"]),
        "d": int(d),
        "k": int(case["k"]),
        "seed": None,
        "data_seed": None,
        "init_seed": None,
        "repeat_id": None,
        "order_block": None,
        "order_index": None,
        "warmup_state": None,
        "dtype": case["dtype"],
        "device": "cuda",
        "hardware": hardware,
        "gpu_index": gpu_index,
        "hardware_fingerprint": context.get("hardware_fingerprint", {}),
        "environment": context.get("environment", {}),
        "data_fingerprint": {"state": "not_loaded"},
        "implementation": None,
        "verdict": "GPU_GENUINE_REQUIRED",
        "t_total": None,
        "t_candidate_e2e": None,
        "t_candidate_e2e_definition": "host_fp32_buffer_ready_to_host_fp32_centroids_materialized_v1",
        "t_input_load": None,
        "t_preprocess": None,
        "t_algorithm_wall": None,
        "t_centroid_write": None,
        "t_kernel": None,
        "t_alloc": None,
        "t_h2d": None,
        "t_d2h": None,
        "peak_device_memory_gib": None,
        "peak_host_rss_gib": None,
        "process_tree_peak_host_rss_gib": None,
        "inertia": None,
        "n_iter": None,
        "converged": None,
        "status": None,
        "failure_kind": None,
        "elapsed_to_failure_seconds": None,
        "error": None,
        "centroids_file": None,
        "centroids_identity": None,
        "metric": None,
    }


def make_run_context(config: dict[str, Any], config_hash: str, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "experiment_name": config["experiment_name"],
        "track": config.get("track", "validation"),
        "config_sha256": config_hash,
        "quality_policy": config.get("quality_policy", {"id": "unversioned"}),
    }


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("contract_version") != CONTRACT_VERSION:
        errors.append(f"contract_version must be {CONTRACT_VERSION!r}")
    if not isinstance(config.get("experiment_name"), str) or not config["experiment_name"].strip():
        errors.append("experiment_name must be a non-empty string")
    if config.get("track") not in VALID_TRACKS:
        errors.append(f"track must be one of {sorted(VALID_TRACKS)}")
    if not isinstance(config.get("quality_policy"), dict) or not config["quality_policy"].get("id"):
        errors.append("quality_policy.id is required")
    safety = config.get("safety")
    if not isinstance(safety, dict):
        errors.append("safety object is required")
    elif int(safety.get("parallel_cases", 1)) != 1:
        errors.append("parallel_cases must be 1 for this shared-GPU runner")
    cases = config.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("at least one case is required")
    else:
        seen: set[str] = set()
        for index, case in enumerate(cases):
            prefix = f"cases[{index}]"
            if not isinstance(case, dict):
                errors.append(f"{prefix} must be an object")
                continue
            case_id = case.get("id")
            if not isinstance(case_id, str) or not case_id:
                errors.append(f"{prefix}.id is required")
            elif case_id in seen:
                errors.append(f"duplicate case id {case_id!r}")
            else:
                seen.add(case_id)
            for name in ("n", "k", "max_iter"):
                try:
                    if int(case[name]) <= 0:
                        raise ValueError
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{prefix}.{name} must be a positive integer")
            # The formal portfolio includes the explicitly declared
            # ``cpu:mlpack_hamerly:fp64:t4`` candidate.  Rejecting fp64 at
            # this generic schema layer makes its frozen configuration
            # impossible to execute, even though the CPU runner supports it.
            if case.get("dtype") not in {"fp16", "fp32", "fp64"}:
                errors.append(f"{prefix}.dtype must be fp16, fp32 or fp64")
            if case.get("source") not in {"synthetic", "dataset"}:
                errors.append(f"{prefix}.source must be synthetic or dataset")
            if case.get("source") == "synthetic" and not case.get("d"):
                errors.append(f"{prefix}.d is required for synthetic data")
            if case.get("source") == "dataset" and not case.get("dataset"):
                errors.append(f"{prefix}.dataset is required for dataset data")
            if not isinstance(case.get("algorithms"), list) or not case["algorithms"]:
                errors.append(f"{prefix}.algorithms must be a non-empty list")
    return errors


def validate_result_row(row: dict[str, Any]) -> list[str]:
    """Validate a record at append time without rejecting extensible fields."""
    errors: list[str] = []
    for key in ("contract_version", "record_type", "run_id", "experiment_name", "track", "config_sha256",
                "case_sha256", "case_id", "algorithm", "dataset", "status", "data_fingerprint"):
        if key not in row:
            errors.append(f"missing {key}")
    if row.get("contract_version") != CONTRACT_VERSION:
        errors.append(f"unexpected contract_version {row.get('contract_version')!r}")
    if row.get("record_type") != RECORD_TYPE:
        errors.append(f"unexpected record_type {row.get('record_type')!r}")
    if row.get("track") not in VALID_TRACKS:
        errors.append(f"invalid track {row.get('track')!r}")
    if row.get("status") not in VALID_STATUSES:
        errors.append(f"invalid status {row.get('status')!r}")
    for key in ("config_sha256", "case_sha256"):
        value = row.get(key)
        if not isinstance(value, str) or len(value) != SHA256_LENGTH:
            errors.append(f"{key} must be a SHA256 hex string")
    for key in ("n", "d", "k"):
        try:
            if int(row.get(key, 0)) <= 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{key} must be positive")
    if row.get("status") == "OK":
        try:
            if float(row.get("t_total")) < 0:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("OK result requires non-negative t_total")
        if not row.get("metric"):
            errors.append("OK result requires metric")
        if not row.get("centroids_file"):
            errors.append("OK result requires centroids_file")
        candidate_e2e = row.get("t_candidate_e2e")
        if candidate_e2e is not None:
            try:
                if float(candidate_e2e) < 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append("t_candidate_e2e must be non-negative when present")
    if not isinstance(row.get("data_fingerprint"), dict):
        errors.append("data_fingerprint must be an object")
    if not isinstance(row.get("hardware_fingerprint"), dict):
        errors.append("hardware_fingerprint must be an object")
    if not isinstance(row.get("environment"), dict):
        errors.append("environment must be an object")
    return errors
