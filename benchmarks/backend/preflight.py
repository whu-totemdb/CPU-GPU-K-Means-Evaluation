#!/usr/bin/env python3
"""Non-invasive GPU admission check for the high-dimensional K-Means runs."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from result_contract import CONTRACT_VERSION, host_fingerprint


def _run(command: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def gpu_inventory() -> list[dict]:
    """Read GPU state via nvidia-smi without importing a CUDA Python runtime."""
    if not shutil.which("nvidia-smi"):
        raise RuntimeError("nvidia-smi not found; this experiment package requires an NVIDIA CUDA server")
    fields = "index,uuid,name,driver_version,memory.total,memory.free,utilization.gpu"
    code, out, err = _run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if code != 0:
        raise RuntimeError(f"nvidia-smi GPU query failed: {err or out}")
    gpus = []
    for line in out.splitlines():
        cols = [col.strip() for col in line.split(",")]
        if len(cols) != 7:
            continue
        gpus.append({
            "index": int(cols[0]), "uuid": cols[1], "name": cols[2], "driver_version": cols[3],
            "memory_total_mib": int(float(cols[4])), "memory_free_mib": int(float(cols[5])),
            "utilization_gpu_percent": int(float(cols[6])),
        })
    if not gpus:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    return gpus


def compute_processes() -> list[dict]:
    """Return existing compute processes. Empty output is normal on an idle server."""
    fields = "gpu_uuid,pid,process_name,used_memory"
    code, out, err = _run(["nvidia-smi", f"--query-compute-apps={fields}", "--format=csv,noheader,nounits"])
    if code != 0 or not out:
        return []
    rows = []
    for line in out.splitlines():
        cols = [col.strip() for col in line.split(",")]
        if len(cols) == 4:
            try:
                rows.append({"gpu_uuid": cols[0], "pid": int(cols[1]), "process_name": cols[2], "used_memory_mib": int(float(cols[3]))})
            except ValueError:
                continue
    return rows


def python_capabilities() -> dict:
    caps = {name: importlib.util.find_spec(module) is not None for name, module in {
        "numpy": "numpy", "h5py": "h5py", "torch": "torch", "flash_kmeans": "flash_kmeans",
        "faiss_gpu": "faiss", "cuml": "cuml",
    }.items()}
    caps["python"] = sys.executable
    return caps


def cuda_toolchain() -> dict:
    """Collect toolchain identifiers without importing Torch/CuPy."""
    result: dict[str, str | None] = {"nvcc_version": None, "nvidia_smi_cuda_version": None}
    if shutil.which("nvcc"):
        code, out, _ = _run(["nvcc", "--version"])
        if code == 0:
            result["nvcc_version"] = next((line.strip() for line in out.splitlines() if "release" in line.lower()), out)
    if shutil.which("nvidia-smi"):
        code, out, _ = _run(["nvidia-smi"])
        if code == 0:
            for line in out.splitlines():
                if "CUDA Version" in line:
                    result["nvidia_smi_cuda_version"] = line.strip()
                    break
    return result


def environment_fingerprint() -> dict:
    """Versioned environment snapshot attached to every attempted case."""
    fingerprint = host_fingerprint()
    fingerprint["cuda"] = cuda_toolchain()
    fingerprint["captured_at"] = datetime.now(timezone.utc).isoformat()
    return fingerprint


def admission_report(gpu_index: int, *, min_free_fraction: float = 0.70,
                     max_existing_util: int = 10, refuse_if_busy: bool = True) -> dict:
    gpus = gpu_inventory()
    selected = next((gpu for gpu in gpus if gpu["index"] == gpu_index), None)
    if selected is None:
        raise RuntimeError(f"GPU {gpu_index} does not exist; available: {[g['index'] for g in gpus]}")
    processes = [p for p in compute_processes() if p["gpu_uuid"] == selected["uuid"]]
    free_fraction = selected["memory_free_mib"] / selected["memory_total_mib"]
    blockers = []
    if free_fraction < min_free_fraction:
        blockers.append(f"free memory {free_fraction:.1%} < required {min_free_fraction:.1%}")
    if selected["utilization_gpu_percent"] > max_existing_util:
        blockers.append(f"GPU utilization {selected['utilization_gpu_percent']}% > allowed {max_existing_util}%")
    if refuse_if_busy and processes:
        blockers.append(f"existing compute processes: {processes}")
    return {
        "contract_version": CONTRACT_VERSION,
        "checked_at": datetime.now(timezone.utc).isoformat(), "selected_gpu": selected,
        "all_gpus": gpus, "compute_processes": processes, "python_capabilities": python_capabilities(),
        "environment_fingerprint": environment_fingerprint(),
        "admission_policy": {
            "min_free_memory_fraction": min_free_fraction,
            "max_existing_gpu_util_percent": max_existing_util,
            "refuse_if_compute_process_exists": refuse_if_busy,
        },
        "safe_to_start": not blockers, "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only GPU server admission check")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--min-free-memory-fraction", type=float, default=0.70)
    parser.add_argument("--max-existing-gpu-util", type=int, default=10)
    parser.add_argument("--allow-shared-gpu", action="store_true", help="Do not reject existing compute processes")
    parser.add_argument("--require", nargs="*", default=[], choices=["flash_kmeans", "faiss_gpu", "cuml"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = admission_report(args.gpu, min_free_fraction=args.min_free_memory_fraction,
                                  max_existing_util=args.max_existing_gpu_util,
                                  refuse_if_busy=not args.allow_shared_gpu)
        missing = [item for item in args.require if not report["python_capabilities"].get(item)]
        if missing:
            report["safe_to_start"] = False
            report["blockers"].append(f"required Python packages unavailable: {missing}")
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        return 0 if report["safe_to_start"] else 2
    except Exception as exc:
        report = {"safe_to_start": False, "blockers": [str(exc)]}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
