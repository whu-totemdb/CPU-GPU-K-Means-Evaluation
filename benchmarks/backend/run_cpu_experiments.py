#!/usr/bin/env python3
"""Serial, bounded-thread CPU K-Means runner using the shared result contract.

This runner is deliberately independent of GPU admission.  It launches one
fresh child per case, caps numerical-library threads before imports occur, and
records CPU resource settings so CPU results remain in a separate hardware
domain from the H800 GPU runs.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from result_contract import (
    config_sha256,
    exception_status,
    file_identity,
    file_sha256,
    host_fingerprint,
    make_run_context,
    new_result_row,
    process_peak_rss_gib,
    ProcessTreeRssSampler,
    stable_hash,
    validate_config,
    validate_result_row,
)
from run_experiments import (
    ConfigSkip,
    DEFAULT_REGISTRY_PATH,
    case_registry_reference,
    configure_registry,
    load_case_data,
    planned_case_metadata,
    resolve_initialization,
    run_specs,
)


ROOT = Path(__file__).resolve().parent
MLPACK_ALGORITHMS = frozenset({
    "mlpack_naive",
    "mlpack_elkan",
    "mlpack_hamerly",
    "mlpack_pelleg_moore",
    "mlpack_dualtree",
    "mlpack_dualtree_covertree",
})
ALGORITHMS = frozenset({
    "faiss_cpu",
    "sklearn_lloyd",
    "sklearn_elkan",
    "sklearn_minibatch",
    "superkmeans_cpu",
    "mopbucket",
    *MLPACK_ALGORITHMS,
})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def seeds(config: dict[str, Any]) -> list[int]:
    values = config.get("seeds", [config.get("seed", 42)])
    if not isinstance(values, list) or not values:
        raise ValueError("seeds must be a non-empty list")
    result = [int(value) for value in values]
    if len(result) != len(set(result)):
        raise ValueError("seeds must be unique")
    return result


def cpu_threads(config: dict[str, Any]) -> int:
    value = int(config.get("cpu_threads", 4))
    if value < 1:
        raise ValueError("cpu_threads must be positive")
    return value


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    errors = validate_result_row(row)
    if errors:
        raise ValueError(f"refusing invalid result row: {errors}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")


def completed_keys(path: Path, *, terminal: bool = False) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        statuses = {"OK"}
        if terminal:
            statuses.update({"ERROR", "OOM", "TIMEOUT", "SKIPPED_CONFIG"})
        if row.get("status") in statuses and row.get("seed") is not None:
            result.add((str(row.get("case_sha256")), str(row.get("algorithm")), int(row["seed"])))
    return result


def set_thread_environment(threads: int) -> None:
    value = str(threads)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value
    # A CPU baseline must never initialize an accidental GPU backend.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _native_paths(options: dict[str, Any]) -> None:
    """Add only explicitly configured native source/build paths to ``sys.path``.

    There is no sklearn fallback for SuperKMeans or MopBucket: falling back
    would fabricate an evaluation of a different candidate.
    """
    for value in options.get("python_paths", []):
        path = str(value)
        if path and path not in sys.path:
            sys.path.insert(0, path)
    library_paths = [str(value) for value in options.get("library_paths", []) if str(value)]
    if library_paths:
        # mlpack's Python extension is dynamically linked against the isolated
        # Armadillo/OpenBLAS prefix.  Make that prefix explicit in the result
        # evidence rather than depending on a machine-wide ldconfig setting.
        previous = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(
            [*library_paths, *([previous] if previous else [])]
        )


def run_algorithm(name: str, x: Any, *, k: int, max_iter: int, seed: int,
                  threads: int, options: dict[str, Any],
                  initialization: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:
    import numpy as np

    if name in MLPACK_ALGORITHMS:
        _native_paths(options)
        try:
            import mlpack
        except Exception as exc:
            raise RuntimeError(
                f"{name} requires the mlpack 4.8.0 Python binding; "
                "set algorithm_options.<name>.python_paths"
            ) from exc
        mode = {
            "mlpack_naive": "naive",
            "mlpack_elkan": "elkan",
            "mlpack_hamerly": "hamerly",
            "mlpack_pelleg_moore": "pelleg-moore",
            "mlpack_dualtree": "dualtree",
            "mlpack_dualtree_covertree": "dualtree-covertree",
        }[name]
        kwargs: dict[str, Any] = {
            # The mlpack 4.8 Python binding exposes its required `input`
            # parameter as `input_` and converts to Armadillo `arma::mat`
            # (double precision).  Record that precision explicitly instead
            # of labelling this candidate as fp32.
            "input_": np.ascontiguousarray(x, dtype=np.float64),
            "clusters": int(k),
            "max_iterations": int(max_iter),
            "algorithm": mode,
            "seed": int(seed),
        }
        if initialization is not None:
            kwargs["initial_centroids"] = np.ascontiguousarray(
                initialization["centers"], dtype=np.float64
            )
        result = mlpack.kmeans(**kwargs)
        if not isinstance(result, dict) or "centroid" not in result:
            raise RuntimeError(f"mlpack.kmeans returned unexpected keys: {type(result)!r}")
        centers = np.asarray(result["centroid"], dtype=np.float32)
        if centers.shape == (x.shape[1], k):
            centers = centers.T
        if centers.shape != (k, x.shape[1]):
            raise RuntimeError(f"mlpack returned {centers.shape}, expected {(k, x.shape[1])}")
        module_path = Path(str(getattr(mlpack, "__file__", "")))
        return np.ascontiguousarray(centers), {
            "implementation": f"mlpack_4_8_0_{mode}_fp64_internal",
            "input_precision": "float64",
            "compute_precision": "float64_arma_mat",
            "n_iter": None,
            "configured_max_iter": int(max_iter),
            "iteration_count_semantics": "unobserved_configured_max_only",
            "converged": None,
            "inertia": None,
            "initialization_mode": (
                "shared_centroids" if initialization is not None
                else "implementation_seeded_sample_initialization"
            ),
            "compute_timing_semantics": "mlpack_python_binding_kmeans_host_wall",
            "mlpack_algorithm": mode,
            "mlpack_version": getattr(mlpack, "__version__", None),
            "native_module": str(module_path),
        }

    if name == "faiss_cpu":
        if initialization is not None:
            raise ConfigSkip("faiss_cpu has no verified external-centroid entry point in this runner")
        import faiss

        model = faiss.Kmeans(x.shape[1], k, niter=max_iter, nredo=1, verbose=False, seed=seed,
                             gpu=False, min_points_per_centroid=1, max_points_per_centroid=max(len(x), 1))
        model.train(np.ascontiguousarray(x, dtype=np.float32))
        objective = float(model.obj[-1]) if len(model.obj) else None
        n_iter = len(model.obj) if len(model.obj) else None
        return np.asarray(model.centroids, dtype=np.float32), {
            "implementation": "faiss_cpu_kmeans_fp32", "n_iter": n_iter,
            "configured_max_iter": max_iter,
            "iteration_count_semantics": "observed_objective_history_length" if n_iter is not None else "unobserved_configured_max_only",
            "converged": None,
            "inertia": objective, "initialization_mode": "implementation_seeded_default",
            "compute_timing_semantics": "faiss_kmeans_train_opaque_host_wall",
        }

    if name in {"sklearn_lloyd", "sklearn_elkan"}:
        from sklearn.cluster import KMeans

        init_value = (np.asarray(initialization["centers"], dtype=np.float32)
                      if initialization is not None else "k-means++")
        model = KMeans(n_clusters=k, init=init_value, n_init=1, max_iter=max_iter, tol=1e-4,
                       algorithm="lloyd" if name == "sklearn_lloyd" else "elkan", random_state=seed)
        model.fit(x)
        return np.asarray(model.cluster_centers_, dtype=np.float32), {
            "implementation": f"scikit_learn_kmeans_{model.algorithm}_fp32", "n_iter": int(model.n_iter_),
            "converged": int(model.n_iter_) < max_iter, "inertia": float(model.inertia_),
            "initialization_mode": "shared_centroids_n_init_1" if initialization is not None else "kmeans_plus_plus_n_init_1",
            "iteration_count_semantics": "observed_model_n_iter",
            "compute_timing_semantics": "sklearn_fit_host_wall",
        }

    if name == "sklearn_minibatch":
        from sklearn.cluster import MiniBatchKMeans

        init_value = (np.asarray(initialization["centers"], dtype=np.float32)
                      if initialization is not None else "k-means++")
        model = MiniBatchKMeans(n_clusters=k, init=init_value, n_init=1, max_iter=max_iter, tol=1e-4,
                                batch_size=min(4096, len(x)), random_state=seed, reassignment_ratio=0.0)
        model.fit(x)
        return np.asarray(model.cluster_centers_, dtype=np.float32), {
            "implementation": "scikit_learn_minibatch_kmeans_fp32", "n_iter": int(model.n_steps_),
            "converged": None, "inertia": float(model.inertia_),
            "initialization_mode": ("shared_centroids_n_init_1_batch4096" if initialization is not None
                                    else "kmeans_plus_plus_n_init_1_batch4096"),
            "iteration_count_semantics": "observed_model_n_steps",
            "compute_timing_semantics": "sklearn_minibatch_fit_host_wall",
        }
    if name == "superkmeans_cpu":
        if initialization is not None:
            raise ConfigSkip("superkmeans_cpu has no verified external-centroid entry point")
        _native_paths(options)
        try:
            from superkmeans import SuperKMeans
        except Exception as exc:
            raise RuntimeError("superkmeans_cpu native binding unavailable; set algorithm_options.superkmeans_cpu.python_paths") from exc
        kwargs = {
            "n_clusters": k, "dimensionality": int(x.shape[1]), "iters": max_iter,
            "n_threads": int(threads),
            "sample_queries": bool(options.get("sample_queries", False)),
            "early_termination": bool(options.get("early_termination", True)),
        }
        # A seeded run is mandatory for the full matrix.  C0 may explicitly
        # set allow_unverified_seed only to discover the native API; it still
        # remains a non-promoted admission result.
        try:
            model = SuperKMeans(**{**kwargs, "seed": seed})
            seed_mode = "native_seed_argument"
        except TypeError as exc:
            if not bool(options.get("allow_unverified_seed", False)):
                raise RuntimeError("SuperKMeans binding has no verified seed argument; refusing an unseeded main-matrix run") from exc
            model = SuperKMeans(**kwargs)
            seed_mode = "unverified_native_seed"
        centers = np.asarray(model.train(np.ascontiguousarray(x, dtype=np.float32)), dtype=np.float32)
        if centers.shape != (k, x.shape[1]):
            raise RuntimeError(f"SuperKMeans returned {centers.shape}, expected {(k, x.shape[1])}")
        stats = getattr(model, "iteration_stats", None)
        n_iter = len(stats) if stats is not None else max_iter
        return centers, {
            "implementation": "superkmeans_cpu_native_binding", "n_iter": int(n_iter), "converged": None,
            "inertia": None, "initialization_mode": seed_mode,
            "native_module": getattr(sys.modules.get("superkmeans"), "__file__", None),
        }
    if name == "mopbucket":
        if initialization is not None:
            raise ConfigSkip("mopbucket has no verified external-centroid entry point")
        _native_paths(options)
        try:
            import mopbucket_py as mop
        except Exception as exc:
            raise RuntimeError("mopbucket native pybind unavailable; set algorithm_options.mopbucket.python_paths") from exc
        graph = mop.kmeans_graph(np.ascontiguousarray(x, dtype=np.float32), int(k), int(threads))
        # Native MopBucket distributions do not expose one stable Python API.
        # Try only explicit seed entry points; never silently perform an
        # unseeded main-matrix run.  C0 may opt into an *explicitly labelled*
        # unverified discovery run, but that result cannot enter RQ1/RQ2.
        seed_mode = None
        for setter, receiver in (
            ("set_seed", mop), ("seed", mop), ("set_seed", graph), ("seed", graph),
        ):
            candidate = getattr(receiver, setter, None)
            if callable(candidate):
                try:
                    candidate(int(seed))
                    seed_mode = f"{type(receiver).__name__}.{setter}(seed)"
                    break
                except (TypeError, ValueError):
                    pass
        if seed_mode is None:
            for args, label in (((int(seed),), "init_random(seed)"), ((), "init_random()")):
                if not args and not bool(options.get("allow_unverified_seed", False)):
                    continue
                try:
                    graph.init_random(*args)
                    seed_mode = label if args else "unverified_init_random()"
                    break
                except TypeError:
                    pass
        else:
            graph.init_random()
        if seed_mode is None:
            raise RuntimeError("MopBucket binding exposes no verified seed entry point; refusing an unseeded main-matrix run")
        n_iter = 0
        for _ in range(max_iter):
            n_iter += 1
            changed, _ = graph.reassign()
            graph.recompute_centroids()
            if int(changed) == 0:
                break
        centers = np.asarray(graph.get_centroids(), dtype=np.float32)
        if centers.shape != (k, x.shape[1]):
            raise RuntimeError(f"MopBucket returned {centers.shape}, expected {(k, x.shape[1])}")
        score = graph.compute_score() if hasattr(graph, "compute_score") else None
        return centers, {
            "implementation": "mopbucket_native_pybind", "n_iter": n_iter, "converged": None,
            "inertia": float(score) if score is not None else None, "initialization_mode": seed_mode,
            "native_module": getattr(mop, "__file__", None),
        }
    raise ValueError(f"unknown CPU algorithm {name!r}")


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    case = config["cases"][args.case_index]
    thread_count = cpu_threads(config)
    set_thread_environment(thread_count)
    context = make_run_context(config, args.config_sha256, args.run_id)
    hardware = host_fingerprint()
    d = int(case.get("d") or 0)
    if not d and case.get("source") == "dataset":
        from run_experiments import REGISTRY
        d = int(REGISTRY[case["dataset"]]["expected_d"])
    row = new_result_row(case=case, algorithm=args.algorithm, d=d, hardware=hardware.get("processor") or "cpu",
                         gpu_index=-1, context=context)
    row["registry_evidence"] = case_registry_reference(case)
    data_seed = int(args.data_seed if args.data_seed is not None else args.seed)
    init_seed = int(args.init_seed if args.init_seed is not None else args.seed)
    options = dict(config.get("algorithm_options", {}).get(args.algorithm, {}))
    options.setdefault("threads", thread_count)
    row.update(tier="cpu_highdim", device="cpu", gpu_index=None, verdict="CPU_GENUINE",
               seed=args.seed, data_seed=data_seed, init_seed=init_seed,
               repeat_id=json.loads(args.repeat_id_json) if args.repeat_id_json is not None else None,
               order_block=json.loads(args.order_block_json) if args.order_block_json is not None else None,
               order_index=json.loads(args.order_index_json) if args.order_index_json is not None else None,
               warmup_state=json.loads(args.warmup_state_json) if args.warmup_state_json is not None else None,
               hardware_fingerprint={"cpu": hardware, "cpu_threads": thread_count},
               environment={"host": hardware, "thread_environment": {
                   name: os.environ[name] for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
               }, "platform_id": config.get("platform_id", "h800_host_v1__cpu4__1xh800")},
               algorithm_config={"max_iter": int(case["max_iter"]), "options": options})
    started = time.perf_counter()
    try:
        loaded = time.perf_counter()
        x, data_meta = load_case_data(case, args.data_root, data_seed)
        row["t_input_load"] = time.perf_counter() - loaded
        row.update(metric=data_meta["metric"], data_fingerprint=data_meta["data_fingerprint"])
        initialization = resolve_initialization(config, case, data_seed, init_seed, args.config)
        row["initialization"] = ({
            "track": "controlled", "mode": "shared_centroids", "centers_file": initialization["file"],
            "centers_identity": initialization["identity"], "metadata": initialization["metadata"],
        } if initialization is not None else {
            "track": config.get("track"), "mode": "implementation_seeded_default", "seed": init_seed,
        })
        algorithm_started = time.perf_counter()
        centroids, result = run_algorithm(args.algorithm, x, k=int(case["k"]), max_iter=int(case["max_iter"]),
                                           seed=init_seed, threads=thread_count, options=options,
                                           initialization=initialization)
        row["t_algorithm_wall"] = time.perf_counter() - algorithm_started
        row["t_candidate_e2e"] = row["t_algorithm_wall"]
        attempt_id = stable_hash({"run_id": args.run_id, "case": case["id"], "algorithm": args.algorithm,
                                  "seed": args.seed, "run_at": row["run_at"], "pid": os.getpid()})[:12]
        output = args.output_dir / "centroids" / f"{case['id']}__{args.algorithm}__seed{args.seed}__attempt{attempt_id}.npy"
        output.parent.mkdir(parents=True, exist_ok=True)
        write_started = time.perf_counter()
        import numpy as np
        np.save(output, centroids)
        row["t_centroid_write"] = time.perf_counter() - write_started
        row.update(result, centroids_file=str(output),
                   centroids_identity={**file_identity(output), "sha256": file_sha256(output)},
                   peak_host_rss_gib=process_peak_rss_gib(), status="OK")
    except ConfigSkip as exc:
        row.update(status="SKIPPED_CONFIG", failure_kind="configuration", error=str(exc),
                   peak_host_rss_gib=process_peak_rss_gib())
    except Exception as exc:
        status = exception_status(exc)
        row.update(status=status, failure_kind="out_of_memory" if status == "OOM" else "implementation_error",
                   error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
                   peak_host_rss_gib=process_peak_rss_gib())
    row["t_total"] = time.perf_counter() - started
    if row["status"] != "OK":
        row["elapsed_to_failure_seconds"] = row["t_total"]
    return row


def validate_cpu_config(config: dict[str, Any]) -> list[str]:
    errors = validate_config(config)
    try:
        cpu_threads(config)
        run_specs(config)
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
    for index, case in enumerate(config.get("cases", [])):
        for algorithm in case.get("algorithms", []):
            if algorithm not in ALGORITHMS:
                errors.append(f"cases[{index}] uses unknown CPU algorithm {algorithm!r}")
    if config.get("track") == "controlled":
        init = config.get("initialization")
        if not isinstance(init, dict) or init.get("policy") != "shared_forgy_v1":
            errors.append("controlled track requires initialization.policy='shared_forgy_v1'")
    return errors


def print_outcome(row: dict[str, Any]) -> None:
    print(json.dumps({key: row.get(key) for key in ("case_id", "algorithm", "seed", "status", "t_total", "error")},
                     ensure_ascii=False), flush=True)


def run_parent(args: argparse.Namespace) -> int:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    errors = validate_cpu_config(config)
    if errors:
        raise ValueError(f"invalid CPU experiment config: {errors}")
    out = args.output_dir / config["experiment_name"]
    raw = out / "results.jsonl"
    jobs = []
    for index, case in enumerate(config["cases"]):
        if not (case.get("enabled", False) or args.include_disabled):
            continue
        algorithms = [algorithm for algorithm in case["algorithms"]
                      if not args.algorithm or algorithm == args.algorithm]
        specs = run_specs(config)
        if config.get("schedule_policy") == "balanced_rotation_v1" and algorithms:
            for repeat_index, raw_spec in enumerate(specs):
                offset = repeat_index % len(algorithms)
                ordered = algorithms[offset:] + algorithms[:offset]
                for order_index, algorithm in enumerate(ordered):
                    spec = dict(raw_spec)
                    spec["order_block"] = spec.get("order_block") or f"{case['id']}__repeat{repeat_index}"
                    spec["order_index"] = order_index
                    jobs.append((index, case, algorithm, spec))
        else:
            jobs.extend((index, case, algorithm, spec) for algorithm in algorithms for spec in specs)
    if args.skip_existing or args.skip_terminal:
        finished = completed_keys(raw, terminal=args.skip_terminal)
        jobs = [job for job in jobs if (stable_hash(job[1]), job[2], job[3]["run_seed"]) not in finished]
    if args.limit:
        jobs = jobs[:args.limit]
    print(json.dumps({"mode": "execute" if args.execute else "dry_run", "cpu_threads": cpu_threads(config),
                      "scheduled": [{"case": item[1]["id"], "algorithm": item[2], **item[3]} for item in jobs]},
                     ensure_ascii=False, indent=2))
    if not args.execute:
        return 0
    cfg_hash = config_sha256(args.config)
    run_id = args.run_id or f"{config['experiment_name']}__{utc_now().replace(':', '').replace('+00:00', 'Z')}"
    host = host_fingerprint()
    context = make_run_context(config, cfg_hash, run_id)
    thread_count = cpu_threads(config)
    for index, case, algorithm, spec in jobs:
        seed = int(spec["run_seed"])
        data_seed = int(spec["data_seed"])
        init_seed = int(spec["init_seed"])
        planned = planned_case_metadata(case, args.data_root, data_seed)
        options = dict(config.get("algorithm_options", {}).get(algorithm, {}))
        options.setdefault("threads", thread_count)
        row = new_result_row(case=case, algorithm=algorithm, d=planned["d"],
                             hardware=host.get("processor") or "cpu", gpu_index=-1, context=context)
        row["registry_evidence"] = case_registry_reference(case)
        row.update(tier="cpu_highdim", device="cpu", gpu_index=None, verdict="CPU_GENUINE",
                   seed=seed, data_seed=data_seed, init_seed=init_seed,
                   repeat_id=spec.get("repeat_id"), order_block=spec.get("order_block"),
                   order_index=spec.get("order_index"), warmup_state=spec.get("warmup_state"),
                   metric=planned["metric"], data_fingerprint=planned["data_fingerprint"],
                   hardware_fingerprint={"cpu": host, "cpu_threads": thread_count},
                   environment={"host": host, "platform_id": config.get("platform_id", "h800_host_v1__cpu4__1xh800")},
                   algorithm_config={"max_iter": int(case["max_iter"]), "options": options},
                   initialization={"track": config.get("track"), "mode": "planned_shared_centroids"
                                   if config.get("track") == "controlled" else "implementation_seeded_default",
                                   "data_seed": data_seed, "init_seed": init_seed})
        command = [sys.executable, str(Path(__file__).resolve()), "--run-one", "--case-index", str(index),
                   "--algorithm", algorithm, "--seed", str(seed), "--config", str(args.config),
                   "--registry", str(args.registry),
                   "--data-root", str(args.data_root), "--output-dir", str(out), "--run-id", run_id,
                   "--config-sha256", cfg_hash, "--data-seed", str(data_seed), "--init-seed", str(init_seed),
                   "--repeat-id-json", json.dumps(spec.get("repeat_id")),
                   "--order-block-json", json.dumps(spec.get("order_block")),
                   "--order-index-json", json.dumps(spec.get("order_index")),
                   "--warmup-state-json", json.dumps(spec.get("warmup_state"))]
        process: subprocess.Popen[str] | None = None
        tree_sampler: ProcessTreeRssSampler | None = None
        tree_measurement: dict[str, Any] | None = None
        stdout = ""
        stderr = ""
        dispatch_started = time.perf_counter()
        try:
            process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       start_new_session=True)
            with ProcessTreeRssSampler(process.pid) as tree_sampler:
                try:
                    stdout, stderr = process.communicate(timeout=int(config["safety"].get("case_timeout_sec", 3600)))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.communicate()
                    raise
            tree_measurement = tree_sampler.result()
            row = json.loads(next(line for line in reversed(stdout.splitlines()) if line.strip().startswith("{")))
            row.update(process_tree_peak_host_rss_gib=tree_measurement.get("peak_rss_gib"),
                       process_tree_rss_measurement=tree_measurement)
            if process.returncode != 0 and row.get("status") == "OK":
                row.update(status="ERROR", failure_kind="child_exit",
                           elapsed_to_failure_seconds=row.get("t_total"),
                           error=f"child exited {process.returncode}: {stderr[-500:]}")
        except subprocess.TimeoutExpired:
            tree_measurement = tree_sampler.result() if tree_sampler is not None else None
            timeout = float(config["safety"].get("case_timeout_sec", 3600))
            row.update(status="TIMEOUT", failure_kind="timeout", t_total=timeout,
                       elapsed_to_failure_seconds=timeout,
                       process_tree_peak_host_rss_gib=(tree_measurement or {}).get("peak_rss_gib"),
                       process_tree_rss_measurement=tree_measurement,
                       error=f"exceeded {timeout:g} seconds")
        except Exception as exc:
            status = exception_status(RuntimeError(f"{exc}; {stderr[-1000:]}"))
            elapsed = time.perf_counter() - dispatch_started
            tree_measurement = tree_sampler.result() if tree_sampler is not None else tree_measurement
            row.update(status=status, failure_kind="out_of_memory" if status == "OOM" else "dispatcher_error",
                       t_total=elapsed, elapsed_to_failure_seconds=elapsed,
                       process_tree_peak_host_rss_gib=(tree_measurement or {}).get("peak_rss_gib"),
                       process_tree_rss_measurement=tree_measurement,
                       error=f"{type(exc).__name__}: {exc}; child stderr: {stderr[-1000:]}")
        append_jsonl(raw, row)
        print_outcome(row)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Serial, fixed-thread CPU K-Means experiment runner")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH,
                        help="dataset registry JSON (passed unchanged to child processes)")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--algorithm", choices=sorted(ALGORITHMS))
    parser.add_argument("--include-disabled", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-terminal", action="store_true",
                        help="also skip ERROR/OOM/TIMEOUT/SKIPPED_CONFIG rows in unattended matrices")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--data-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--init-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--repeat-id-json", help=argparse.SUPPRESS)
    parser.add_argument("--order-block-json", help=argparse.SUPPRESS)
    parser.add_argument("--order-index-json", help=argparse.SUPPRESS)
    parser.add_argument("--warmup-state-json", help=argparse.SUPPRESS)
    parser.add_argument("--config-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args()
    configure_registry(args.registry)
    args.registry = args.registry.resolve()
    if args.run_one:
        if args.case_index is None or not args.algorithm or args.seed is None or not args.config_sha256 or not args.run_id:
            parser.error("internal child arguments missing")
        print(json.dumps(run_one(args), ensure_ascii=False, default=float))
        return 0
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
