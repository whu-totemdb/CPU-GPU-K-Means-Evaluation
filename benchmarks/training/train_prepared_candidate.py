#!/usr/bin/env python3
"""One sealed V35 prepared-base training unit. Import/default CLI never trains.
The scheduler owns admission, external timeout and process-tree memory sampling.
This child verifies identities/binding and refuses implicit retries/overwrites.
"""
from __future__ import annotations
import argparse
import ast
import contextlib
import fcntl
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
        f.flush()
        os.fsync(f.fileno())


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_normalization_function(path):
    """Compile the exact frozen function only; avoid its unrelated FAISS GT imports."""
    import numpy as np
    tree = ast.parse(Path(path).read_text())
    functions = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'apply_frozen_normalization']
    require(len(functions) == 1, 'FROZEN_NORMALIZER_NOT_UNIQUE')
    namespace = {'np': np, 'require': require}
    code = compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec')
    exec(code, namespace)
    return namespace['apply_frozen_normalization']


def bound_file(manifest, path):
    path = str(Path(path))
    expected = manifest['source_sha256'].get(path)
    require(expected and sha(path) == expected, 'SOURCE_SHA_MISSING_OR_MISMATCH:' + path)
    return path


def resource_binding(manifest, unit):
    lane = unit.get('resource_lane', unit['device'])
    binding = manifest['resources'][lane]
    if 'resource_lane' in unit:
        require(lane == 'cuda' if unit['device'] == 'cuda' else lane in ('cpu0', 'cpu1', 'cpu2'), 'RESOURCE_LANE_DEVICE_MISMATCH')
        require(unit.get('assigned_cpu_list') == binding['cpus'] and unit.get('assigned_numa') == binding['numa'], 'JOB_RESOURCE_BINDING_MISMATCH')
    return binding


def validate(manifest, unit):
    require(manifest.get('namespace') == 'FINAL_VLDB_V35' and manifest.get('frozen') is True, 'FROZEN_MANIFEST_REQUIRED')
    jobs = manifest['jobs']
    require(len({j['unit_id'] for j in jobs}) == len(jobs), 'DUPLICATE_UNIT_ID')
    require(unit['status'] == 'READY', 'UNIT_NOT_READY')
    require(unit['seed'] == 20260805 and unit['device'] in ('cpu', 'cuda'), 'SEED_OR_DEVICE_SCOPE')
    require(unit['cpu_threads'] == 4 and 0 < unit['k'] <= unit['n'] and unit['d'] > 0, 'SHAPE_OR_THREADS')
    require(Path(unit['unit_id']).name == unit['unit_id'], 'UNSAFE_UNIT_ID')
    require(unit['metric'] in ('l2', 'ip') and isinstance(unit['normalize'], bool), 'METRIC_OR_NORMALIZATION')
    require(manifest.get('source_sha256'), 'SOURCE_FREEZE_REQUIRED')
    config = read(unit['original_config_path'])
    require(sha(unit['original_config_path']) == unit['original_config_sha256'], 'CONFIG_SHA_MISMATCH')
    require(config.get('track') != 'controlled' and config.get('initialization') is None, 'OPERATIONAL_INITIALIZATION_ONLY')
    require(config.get('seeds') == [20260805], 'CONFIG_SEED_DRIFT')
    require(config.get('safety', {}).get('case_timeout_sec') == unit['timeout_seconds'], 'TIMEOUT_DRIFT')
    require(config.get('algorithm_options', {}).get(unit['algorithm'], {}) == unit['algorithm_options'], 'ALGORITHM_OPTIONS_DRIFT')
    require(config.get('execution_binding', {}) == unit['execution_binding'], 'EXECUTION_BINDING_DRIFT')
    require(not unit['algorithm_options'].get('allow_unverified_seed', False), 'UNVERIFIED_NATIVE_SEED_FORBIDDEN')
    require(unit['candidate_id'].startswith('cpu:' if unit['device'] == 'cpu' else 'gpu:'), 'DEVICE_IDENTITY')
    binding = resource_binding(manifest, unit)
    require(len(binding['cpus']) == len(set(binding['cpus'])) >= 4 and binding['numa'] >= 0, 'RESOURCE_BINDING')
    if unit['device'] == 'cuda':
        require(binding.get('gpu_uuid', '').startswith('GPU-') and ',' not in binding['gpu_uuid'], 'ONE_GPU_UUID_REQUIRED')
    return config


def array_sha(array):
    require(array.flags.c_contiguous, 'CONTIGUOUS_ARRAY_REQUIRED')
    return hashlib.sha256(memoryview(array).cast('B')).hexdigest()


def load_prepared(unit, normalization):
    """No registry re-selection. Load every already-selected raw FP32 train row."""
    import h5py
    import numpy as np
    t = time.perf_counter()
    require(sha(unit['asset_receipt_path']) == unit['asset_receipt_sha256'], 'ASSET_RECEIPT_SHA')
    receipt = read(unit['asset_receipt_path'])
    require(receipt['asset_id'] == unit['asset_id'] and receipt['status'] == 'MATERIALIZED_EXACT_TOP10', 'ASSET_RECEIPT_IDENTITY')
    require((receipt['n_base'], receipt['d'], receipt['metric'], receipt['normalized']) == (unit['n'], unit['d'], unit['metric'], unit['normalize']), 'ASSET_SHAPE_SEMANTICS')
    require(receipt['raw_base_array_sha256'] == unit['base_raw_array_sha256'] and receipt['base_array_sha256'] == unit['base_effective_array_sha256'], 'ASSET_ARRAY_BINDING')
    require(receipt['base_source_ids_array_sha256'] == unit['base_source_ids_array_sha256'], 'ASSET_SOURCE_ROW_ORDER')
    require(receipt['files']['base.hdf5'] == unit['base_sha256'] and sha(unit['base_path']) == unit['base_sha256'], 'BASE_FILE_SHA')
    hash_s = time.perf_counter() - t
    t = time.perf_counter()
    with h5py.File(unit['base_path'], 'r') as f:
        require('train' in f and f['train'].shape == (unit['n'], unit['d']) and f['train'].dtype == np.dtype('float32'), 'BASE_DATASET_SHAPE_DTYPE')
        require(not bool(f.attrs.get('normalization_applied', True)), 'BASE_NOT_RAW')
        require(bool(f.attrs.get('normalization_required')) == unit['normalize'], 'NORMALIZATION_ATTRIBUTE')
        x = np.ascontiguousarray(f['train'][:], dtype=np.float32)
    load_s = time.perf_counter() - t
    t = time.perf_counter()
    require(np.isfinite(x).all() and array_sha(x) == unit['base_raw_array_sha256'], 'BASE_RAW_ARRAY_SHA_OR_FINITE')
    raw_check_s = time.perf_counter() - t
    t = time.perf_counter()
    norm_receipt = normalization(x, unit['normalize'], 'V35 prepared training base')
    normalize_s = time.perf_counter() - t
    t = time.perf_counter()
    require(np.isfinite(x).all() and array_sha(x) == unit['base_effective_array_sha256'], 'BASE_EFFECTIVE_ARRAY_SHA_OR_FINITE')
    return x, {'file_identity_check_seconds': hash_s, 'base_load_seconds': load_s, 'raw_array_check_seconds': raw_check_s, 'normalization_seconds': normalize_s, 'effective_array_check_seconds': time.perf_counter() - t, 'normalization': norm_receipt, 'raw_array_sha256': unit['base_raw_array_sha256'], 'effective_array_sha256': unit['base_effective_array_sha256'], 'selection': 'all rows from prepared train; no resampling', 'base_selection': receipt['base_selection'], 'source_ids_array_sha256': unit['base_source_ids_array_sha256']}


@contextlib.contextmanager
def cpu_threads(runner, unit):
    runner._native_paths(unit['algorithm_options'])
    module_name = {'faiss_cpu': 'faiss', 'sklearn_lloyd': 'sklearn.cluster', 'sklearn_elkan': 'sklearn.cluster', 'sklearn_minibatch': 'sklearn.cluster', 'minibatch_kmeans': 'sklearn.cluster', 'mlpack_hamerly': 'mlpack', 'superkmeans_cpu': 'superkmeans', 'mopbucket_cpu': 'mopbucket_py', 'mopbucket': 'mopbucket_py'}.get(unit['algorithm'])
    require(module_name, 'UNBOUND_CPU_IMPLEMENTATION')
    module = importlib.import_module(module_name)
    import threadpoolctl
    threads = unit['cpu_threads']
    if unit['algorithm'] == 'faiss_cpu':
        module.omp_set_num_threads(threads)
        require(module.omp_get_max_threads() == threads, 'FAISS_THREAD_READBACK')
    with threadpoolctl.threadpool_limits(limits=threads):
        pools = threadpoolctl.threadpool_info()
        require(pools and all(p['num_threads'] == threads for p in pools), 'NATIVE_THREAD_POOL_READBACK')
        evidence = {'requested': threads, 'before_pools': pools, 'native_options': unit['algorithm_options'], 'setup_excluded_from_adapter_timer': True}
        yield evidence
        evidence['after_pools'] = threadpoolctl.threadpool_info()
        require(all(p['num_threads'] == threads for p in evidence['after_pools']), 'AFTER_NATIVE_THREAD_POOL_READBACK')
        if unit['algorithm'] == 'faiss_cpu':
            require(module.omp_get_max_threads() == threads, 'AFTER_FAISS_THREAD_READBACK')


def native_super(runner, manifest, unit, x, output):
    """Only prepared-input adaptation; native command/training options unchanged."""
    import numpy as np
    options = unit['algorithm_options']
    binary = unit['execution_binding'].get('super_runner')
    require(binary, 'SUPER_NATIVE_BINARY_NOT_BOUND')
    bound_file(manifest, binary)
    fraction = float(options.get('sampling_fraction', 0.3))
    proof = runner.sampling_cardinality(n=unit['n'], k=unit['k'], sampling_fraction=fraction)
    require(0 < fraction <= 1 and not proof['unsupported'], 'SUPER_SAMPLING_UNSUPPORTED')
    threads = int(options.get('threads', 4))
    require(threads == unit['cpu_threads'] and 1 <= threads <= 8, 'SUPER_NATIVE_THREAD_DRIFT')
    raw_input, raw_centroids = output / 'prepared_effective.f32', output / 'native_centroids.f32'
    t = time.perf_counter()
    with raw_input.open('xb') as f:
        x.tofile(f)
    input_sha = sha(raw_input)
    require(input_sha == unit['base_effective_array_sha256'], 'SUPER_PREPARED_INPUT_SHA')
    prep_s = time.perf_counter() - t
    command = [binary, '--input', str(raw_input), '--n', str(unit['n']), '--d', str(unit['d']), '--k', str(unit['k']), '--iters', str(unit['max_iter']), '--seed', str(unit['seed']), '--threads', str(threads), '--sampling-fraction', str(fraction), '--centroids-out', str(raw_centroids)]
    if unit['metric'] == 'ip':
        command.append('--angular')
    proc = subprocess.run(command, text=True, capture_output=True, check=False)
    (output / 'native.stdout.log').write_text(proc.stdout)
    (output / 'native.stderr.log').write_text(proc.stderr)
    require(proc.returncode == 0, 'SUPER_NATIVE_FAILED:' + str(proc.returncode))
    native = runner._last_json(proc.stdout)
    require(native.get('status') == 'OK', 'SUPER_NATIVE_STATUS:' + str(native))
    centers = np.fromfile(raw_centroids, dtype=np.float32)
    require(centers.size == unit['k'] * unit['d'], 'SUPER_CENTROID_SHAPE')
    train_wall = native.get('t_train_wall', native.get('t_total'))
    require(train_wall is not None and np.isfinite(train_wall) and train_wall >= 0, 'SUPER_NATIVE_TRAIN_TIME_MISSING')
    detail = {'implementation': 'native_gpu_superkmeans_cuda', 'centroids': centers.reshape(unit['k'], unit['d']), 'n_iter': int(native.get('n_iter', unit['max_iter'])), 'converged': None, 't_alloc': native.get('t_model_init'), 't_h2d': native.get('t_h2d'), 't_kernel': train_wall, 't_d2h': native.get('t_d2h'), 't_preprocess': prep_s, 'compute_timing_semantics': 'native_superkmeans_train_wall', 'iteration_timing_mode': 'aggregate_native_wall', 'initialization_mode': 'implementation_seeded_default', 'superkmeans_native': native, 'superkmeans_command': command, 'superkmeans_runner_sha256': sha(binary), 'prepared_input_array_sha256': input_sha, 'sampling_proof': proof, 'native_centroids_sha256': sha(raw_centroids), 'peak_device_memory_gib': native.get('peak_device_memory_gib')}
    # Only our newly created disposable raw-input copy; raw centroid output remains.
    if not options.get('keep_prepared_input', False):
        raw_input.unlink()
    return detail


def classify_failure(exc, stage):
    if stage in ('binding', 'input_preparation', 'adapter_setup'):
        return 'MEASUREMENT_FAILED'
    # Use the original exception classifier if the adapter imported it. No
    # re-import here: import/binding failures must not run additional code.
    contract = sys.modules.get('result_contract')
    status = contract.exception_status(exc) if contract is not None else ('OOM' if isinstance(exc, MemoryError) else 'ERROR')
    if status in ('OOM', 'TIMEOUT'):
        return status
    return 'ERROR' if stage == 'adapter' else 'MEASUREMENT_FAILED'


def run_child(manifest, unit, config, output):
    began = time.perf_counter()
    record = {k: unit[k] for k in ('unit_id', 'task_id', 'candidate_id', 'algorithm', 'device', 'seed', 'n', 'd', 'k', 'metric', 'normalize')}
    record.update(status='ERROR', started_unix=time.time(), pid=os.getpid(), original_config_sha256=unit['original_config_sha256'], timeout_seconds=unit['timeout_seconds'], quality_and_resource_admission='NOT_EVALUATED_BY_TRAINING_WORKER', t_algorithm_wall=None, t_candidate_e2e=None, peak_device_memory_gib=None)
    stage = 'binding'
    try:
        threads = unit['cpu_threads']
        os.environ.update({key: str(threads) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
        binding = resource_binding(manifest, unit)
        os.environ['CUDA_VISIBLE_DEVICES'] = '' if unit['device'] == 'cpu' else binding['gpu_uuid']
        for key, value in unit['execution_binding'].get('environment', {}).items():
            require(os.environ.get(key) == str(value), 'EXECUTION_ENVIRONMENT_MISMATCH:' + key)
        executable = unit['execution_binding'].get('python_executable')
        require(executable and Path(sys.executable).absolute() == Path(executable).absolute(), 'ORIGINAL_PYTHON_ENVIRONMENT_REQUIRED')
        os.sched_setaffinity(0, set(binding['cpus']))
        require(set(os.sched_getaffinity(0)) == set(binding['cpus']), 'CPU_AFFINITY_READBACK')
        numa = load_module(bound_file(manifest, manifest['numa_binder']), '_v35_training_numa')
        numa_readback = numa.bind_memory(binding['numa'], numa.system_nodes())
        require(numa_readback[0] == 2 and numa_readback[1] == [binding['numa']], 'NUMA_POLICY_READBACK')
        record['resource_binding'] = {**binding, 'observed_cpu_affinity': sorted(os.sched_getaffinity(0)), 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'numa_bind_memory_call': 'completed', 'numa_policy_readback': numa_readback, 'process_uid': os.getuid()}
        sys.path.insert(0, manifest['repository'])
        normal_path = manifest.get('normalization_source_path', str(Path(__file__).resolve().parents[1] / 'assets/materialize_exact_gt.py'))
        normalizer = load_normalization_function(bound_file(manifest, normal_path))
        stage = 'input_preparation'
        x, preparation = load_prepared(unit, normalizer)
        preparation['normalization_source_sha256'] = sha(normal_path)
        preparation['normalization_function'] = 'apply_frozen_normalization; exact AST function body, no GT execution'
        record['input_preparation'] = preparation
        runner_name = 'run_cpu_experiments' if unit['device'] == 'cpu' else 'run_experiments'
        bound_file(manifest, Path(manifest['repository']) / (runner_name + '.py'))
        runner = importlib.import_module(runner_name)
        stage = 'adapter_setup'
        if unit['device'] == 'cpu':
            with cpu_threads(runner, unit) as evidence:
                stage = 'adapter'
                start = time.perf_counter()
                centers, detail = runner.run_algorithm(unit['algorithm'], x, k=unit['k'], max_iter=unit['max_iter'], seed=unit['seed'], threads=threads, options=unit['algorithm_options'], initialization=None)
                elapsed = time.perf_counter() - start
            record['thread_evidence'] = evidence
        else:
            require('dataset' in unit and 'registry_path' in manifest, 'GPU_DATASET_REGISTRY_BINDING_REQUIRED')
            runner.configure_registry(Path(bound_file(manifest, manifest['registry_path'])))
            entry = runner.REGISTRY[unit['dataset']]
            require(entry['metric'] == unit['metric'] and bool(entry.get('normalize')) == unit['normalize'], 'GPU_DATASET_SEMANTICS_MISMATCH')
            case = {'id': unit['task_id'], 'source': 'dataset', 'dataset': unit['dataset'], 'n': unit['n'], 'd': unit['d'], 'k': unit['k'], 'max_iter': unit['max_iter'], 'dtype': unit['dtype']}
            if unit['algorithm'] == 'faiss_gpu':
                contract_path = manifest['faiss_correction_contract_path']
                bound_file(manifest, contract_path)
                wrapper = Path(bound_file(manifest, manifest['faiss_correction_wrapper_path']))
                # Import dependency is read-only; it is not instantiated by corrected_faiss_gpu.
                bound_file(manifest, wrapper.parent / 'gpu_memory_evidence.py')
                sys.path.insert(0, str(wrapper.parent))
                os.environ['V35_CORRECTION_CONTRACT'] = contract_path
                function = load_module(wrapper, '_v35_prepared_corrected_faiss').corrected_faiss_gpu
            else:
                require(unit['algorithm'] in runner.ALGORITHMS, 'UNREGISTERED_GPU_ADAPTER')
                function = runner.ALGORITHMS[unit['algorithm']].runner
            stage = 'adapter'
            start = time.perf_counter()
            detail = native_super(runner, manifest, unit, x, output) if unit['algorithm'] == 'superkmeans_gpu' else function(x=x, case=case, seed=unit['seed'], initialization=None, config=config, data_root=Path(unit['base_path']).parent, output_dir=output, super_runner=None)
            elapsed = time.perf_counter() - start
            centers = detail.pop('centroids')
        stage = 'output_validation'
        record['t_algorithm_wall'] = elapsed
        record['t_candidate_e2e'] = elapsed
        record['timing_domain'] = 'original run_algorithm/registered GPU runner call; GPU Super prepared-input write and native train included; base read/normalization separately recorded'
        import numpy as np
        centers = np.asarray(centers)
        require(centers.shape == (unit['k'], unit['d']) and np.isfinite(centers).all(), 'CENTROID_SHAPE_OR_FINITE')
        path = output / 'centroids.npy'
        t = time.perf_counter()
        with path.open('xb') as f:
            np.save(f, centers, allow_pickle=False)
        record.update(detail, status='OK', centroids_file=str(path), centroids_sha256=sha(path), centroids_array_sha256=array_sha(np.ascontiguousarray(centers)), centroid_dtype=str(centers.dtype), t_centroid_write=time.perf_counter() - t)
    except Exception as exc:
        record.update(status=classify_failure(exc, stage), failure_stage=stage, error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc(), elapsed_to_failure_seconds=time.perf_counter() - began)
    record.update(t_total=time.perf_counter() - began, ended_unix=time.time(), peak_host_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2), host_rss_semantics='Linux RUSAGE_SELF high-water mark; parent process-tree trace is primary', memory_evidence_owner='external scheduler')
    write_new(output / 'child_result.json', record)
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--manifest-sha256', required=True)
    p.add_argument('--unit-id', required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    require(sha(a.manifest) == a.manifest_sha256, 'MANIFEST_SHA_MISMATCH')
    manifest = read(a.manifest)
    units = [j for j in manifest['jobs'] if j['unit_id'] == a.unit_id]
    require(len(units) == 1, 'UNIT_NOT_UNIQUELY_REGISTERED')
    unit = units[0]
    config = validate(manifest, unit)
    for path, digest in manifest['source_sha256'].items():
        require(sha(path) == digest, 'DEPENDENCY_SHA_MISMATCH:' + path)
    bound_file(manifest, Path(__file__).resolve())
    if not a.execute:
        print(json.dumps({'status': 'DRY_RUN', 'unit_id': unit['unit_id'], 'candidate_id': unit['candidate_id'], 'base_path': unit['base_path'], 'would_execute_adapter': unit['adapter_symbol'], 'experiments_started': False}))
        return
    require(sys.platform.startswith('linux'), 'LINUX_BINDING_REQUIRED')
    output = Path(manifest['output_root']).resolve() / unit['unit_id']
    require('FINAL_VLDB_V35' in output.parts and output.is_dir() and not output.is_symlink(), 'OWNED_PARENT_CREATED_OUTPUT_REQUIRED')
    with (output / '.worker.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = output / 'child_result.json'
        if result.exists():
            old = read(result)
            require(old['unit_id'] == unit['unit_id'] and old['candidate_id'] == unit['candidate_id'] and old['original_config_sha256'] == unit['original_config_sha256'], 'TERMINAL_IDENTITY_MISMATCH')
            print(json.dumps({'status': 'EXISTING_TERMINAL_NO_RETRAIN', 'result_status': old['status'], 'unit_id': unit['unit_id']}))
            return
        write_new(output / 'started.json', {'unit_id': unit['unit_id'], 'manifest_sha256': a.manifest_sha256, 'started_unix': time.time(), 'pid': os.getpid(), 'retry_policy': 'no implicit retry; interrupted start requires parent-reviewed new attempt identity'})
        result = run_child(manifest, unit, config, output)
        print(json.dumps({'status': result['status'], 'unit_id': unit['unit_id']}))
        if result['status'] != 'OK':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
