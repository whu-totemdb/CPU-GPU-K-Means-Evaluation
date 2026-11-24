"""Prepared-input public objective and frozen SW1 geometry, no SQL or model fit.

Exact objective functions are imported unchanged from the frozen public evaluator.
All 130 registered training identities are retained; only M members get profiles.
Evaluation query files are never opened by this entrypoint.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent

def require(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4194304), b''):
            digest.update(block)
    return digest.hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.flush()
        os.fsync(out.fileno())

def event(path, **value):
    value['time_unix'] = time.time()
    with Path(path).open('a') as out:
        out.write(json.dumps(value, allow_nan=False) + '\n')
        out.flush()
        os.fsync(out.fileno())

def load(name, path, digest):
    require(sha(path) == digest, 'source drift: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def validate(manifest, path, expected):
    require(sha(path) == expected and manifest['frozen'] is True, 'frozen quality manifest')
    require(manifest['namespace'] == 'FINAL_VLDB_V35' and manifest['seed'] == 20260805, 'scope')
    require(len(manifest['units']) == 130 and len({u['job']['unit_id'] for u in manifest['units']}) == 130, '130 unique units')
    require(sum(u['profile_required'] for u in manifest['units']) == 100, 'only 100 M profile slots')
    require(manifest['profile_config'] == dict(data_sample_max=8192, query_sample_max=128, sample_seed=20260911, distance_block=256,
            probe_grid=[1,2,4,8,16,32,64,128,256,512,1024,2048,4096]), 'frozen SW1 proxy unchanged')
    require(manifest['objective_backend'] == 'faiss_flat' and manifest['data_batch_rows'] == 2048 and manifest['centroid_batch_rows'] == 256, 'public objective protocol')
    for source, digest in manifest['source_sha256'].items():
        require(sha(source) == digest, 'source drift: ' + source)
    originals = []
    for path in manifest['training_manifests']:
        source = json.loads(Path(path).read_text())
        originals.extend(source['jobs'])
    require([u['job'] for u in manifest['units']] == originals, 'quality units differ from training freezes')

def prior_completed(manifest):
    root = Path(manifest.get('prior_output_root', ''))
    result = {}
    if root.exists():
        for path in root.glob('units/*/result.json'):
            row = json.loads(path.read_text())
            result[row['unit_id']] = path
    return result

def training_result(unit):
    job = unit['job']
    for root in map(Path, unit['search_roots']):
        directory = root / job['unit_id']
        if not directory.exists():
            continue
        path = directory / 'result.json'
        if not path.exists():
            return None
        row = json.loads(path.read_text())
        require(all(row.get(key) == job[key] for key in ('unit_id', 'task_id', 'candidate_id', 'seed')), 'training result identity')
        if row.get('no_new_physical'):
            index = json.loads((directory / 'reuse_index.json').read_text())
            target = Path(index['prior_result_identity']['path'])
            require(target.parent.parent.as_posix() in unit['search_roots'] and target.parent.name == job['unit_id'], 'reuse outside registered roots')
            require(sha(target) == index['prior_result_identity']['sha256'], 'reused result drift')
        else:
            require(json.loads((directory / 'job.json').read_text()) == job, 'physical training job drift')
        return path, row
    return None

def objective(x, centers, metric, evaluator):
    import numpy as np
    started = time.perf_counter()
    centers = evaluator._centroids_for_objective(centers, metric)
    index = evaluator._build_exact_assignment_index(centers, metric, 'faiss_flat')
    total = 0.0
    maximum = 0.0
    kernel = 0.0
    for start in range(0, len(x), 2048):
        block = np.ascontiguousarray(x[start:start+2048], dtype=np.float32).copy()
        if metric == 'ip':
            evaluator._normalise_angular_rows_or_raise(block, object_name='input vector')
        began = time.perf_counter()
        costs = evaluator._exact_assignment_costs(block, centers, metric=metric, backend='faiss_flat', centroid_batch_rows=256, faiss_index=index)
        kernel += time.perf_counter() - began
        require(np.isfinite(costs).all(), 'nonfinite public objective')
        total += float(costs.sum(dtype=np.float64))
        maximum = max(maximum, float(costs.max()))
    return dict(status='OK', name=evaluator.objective_name(metric), metric=metric, n_evaluated=len(x),
                sum_assignment_cost=total, mean_assignment_cost=total/len(x), max_assignment_cost=maximum,
                exact_assignment_backend='faiss_flat', ground_truth_independent=True,
                objective_seconds=time.perf_counter()-started, objective_kernel_seconds=kernel,
                representation='FP32 inputs/centroids, FP64 objective sum; original frozen public evaluator functions')

def geometry(manifest, unit, x, centers, profiler):
    import numpy as np
    job = unit['job']
    cfg = manifest['profile_config']
    started = time.perf_counter()
    receipt = json.loads(Path(job['asset_receipt_path']).read_text())
    root = Path(job['base_path']).parent
    paths = [root/'calibration/effective_queries.npy', root/'calibration/exact_top10.npz']
    for path in paths:
        require(sha(path) == receipt['files'][str(path.relative_to(root))], 'calibration asset drift')
    queries = np.load(paths[0], allow_pickle=False)
    with np.load(paths[1], allow_pickle=False) as truth:
        require(np.array_equal(truth['query_indices'], np.arange(500)), 'calibration split identity')
        gt_ids = truth['neighbor_ids'][:128].copy()
    require(queries.shape == (500, job['d']) and gt_ids.shape == (128, 10), 'profile query geometry')
    ids = np.sort(np.random.default_rng(cfg['sample_seed']).choice(job['n'], min(job['n'], 8192), replace=False))
    data = x[ids].copy()
    query = queries[:128].copy()
    gt_vectors = x[gt_ids].copy()
    centers = centers.astype(np.float32, copy=True)
    if job['metric'] == 'ip':
        norms = np.sqrt(np.sum(centers.astype(np.float64)**2, axis=1))
        require(np.all(norms > 0), 'undefined angular centroid for proxy')
        centers /= norms[:, None]
    read_seconds = time.perf_counter()-started
    result = profiler.profile(data, query, gt_vectors, centers, job['n'], job['metric'], cfg['probe_grid'], cfg['distance_block'])
    result.update(sample_seed=cfg['sample_seed'], data_sample_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                  query_ids=list(range(128)), query_asset_sha256=sha(paths[0]), gt_asset_sha256=sha(paths[1]),
                  sample_and_centroid_preparation_seconds=read_seconds, evaluation_queries_read=False)
    return result

def process(manifest, unit, source, worker, evaluator, profiler, binding):
    import numpy as np
    job = unit['job']
    path, row = source
    record = dict(unit_id=job['unit_id'], task_id=job['task_id'], candidate_id=job['candidate_id'], seed=job['seed'],
                  training_status=row['status'], training_result_path=str(path), training_result_sha256=sha(path),
                  base_effective_array_sha256=job['base_effective_array_sha256'], base_sha256=job['base_sha256'],
                  profile_required=unit['profile_required'], resource_binding=binding, started_unix=time.time())
    started = time.perf_counter()
    if row['status'] != 'OK':
        require(row['status'] in ('ERROR', 'OOM', 'TIMEOUT', 'MEASUREMENT_FAILED'), 'unknown training terminal')
        if row['status'] == 'MEASUREMENT_FAILED':
            require(row.get('failure_stage') == 'output_validation' and row.get('error') == 'ValueError: CENTROID_SHAPE_OR_FINITE', 'unresolved infrastructure failure')
        record.update(status='NO_VALID_TRAINING_OUTPUT', objective=None, profile=None,
                      evidence='training failure retained; no imputed quality, memory or utility')
    else:
        center_path = Path(row['centroids_file'])
        require(sha(center_path) == row['centroids_sha256'], 'centroid artifact drift')
        centers = np.load(center_path, allow_pickle=False)
        require(centers.shape == (job['k'], job['d']) and np.isfinite(centers).all(), 'certified training output changed')
        normalizer = worker.load_normalization_function(manifest['normalization_source'])
        x, preparation = worker.load_prepared(job, normalizer)
        record.update(centroids_path=str(center_path), centroids_sha256=row['centroids_sha256'], input_preparation=preparation)
        if job['metric'] == 'ip' and np.any(np.linalg.norm(centers.astype(np.float32), axis=1) <= 1e-12):
            record.update(status='INVALID_ANGULAR_ZERO_CENTROID', objective=None, profile=None)
        else:
            record['objective'] = objective(x, centers, job['metric'], evaluator)
            record['profile'] = geometry(manifest, unit, x, centers, profiler) if unit['profile_required'] else None
            record['status'] = 'OK'
    record.update(ended_unix=time.time(), wall_seconds=time.perf_counter()-started,
                  quality_profile_scope='pre-decision preparation, independent reference separately identified; no target QPS')
    return record

def lane(manifest, name):
    root = Path(manifest['output_root'])/'lanes'/name
    root.mkdir(parents=True, exist_ok=False)
    events = root/'events.jsonl'
    resource = manifest['resources'][name]
    try:
        sys.path.insert(0, manifest['repository'])
        worker = load('prepared_worker_quality', manifest['training_worker'], manifest['source_sha256'][manifest['training_worker']])
        binder = load('quality_numa', manifest['numa_binder'], manifest['source_sha256'][manifest['numa_binder']])
        helper = load('quality_resource_gate', manifest['gate_helper'], manifest['source_sha256'][manifest['gate_helper']])
        topology = helper.topology(resource['cpus'] + manifest['protected_cpus'])
        selected = set().union(*(set(topology[str(c)]) for c in resource['cpus']))
        protected = set().union(*(set(topology[str(c)]) for c in manifest['protected_cpus']))
        require(not selected & protected, 'quality/training HT overlap')
        os.sched_setaffinity(0, resource['cpus'])
        policy = binder.bind_memory(resource['numa'], binder.system_nodes())
        require(policy[0] == 2 and policy[1] == [resource['numa']], 'NUMA readback')
        require(sorted(os.sched_getaffinity(0)) == resource['cpus'], 'CPU affinity readback')
        import faiss
        import threadpoolctl
        faiss.omp_set_num_threads(4)
        require(faiss.omp_get_max_threads() == 4, 'FAISS thread binding')
        evaluator = load('quality_eval_frozen', manifest['public_evaluator'], manifest['source_sha256'][manifest['public_evaluator']])
        profiler = load('sw1_proxy_frozen', manifest['profiler'], manifest['source_sha256'][manifest['profiler']])
        binding = dict(cpus=resource['cpus'], numa=resource['numa'], policy=list(policy), pid=os.getpid(), uid=os.geteuid(), faiss_threads=4)
        write(root/'binding.json', binding)
        completed = prior_completed(manifest)
        pending = [u for u in manifest['units']
                   if u['lane'] == name and u['job']['unit_id'] not in completed]
        with (root/'dispatch.lock').open('a+') as lock, threadpoolctl.threadpool_limits(limits=4):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while pending:
                ready = next(((u, training_result(u)) for u in pending if training_result(u) is not None), None)
                if ready is None:
                    event(events, event='WAIT_TRAINING_OUTPUTS', remaining=len(pending))
                    time.sleep(15)
                    continue
                unit, source = ready
                before = helper.proc_snapshot(sorted(selected))
                time.sleep(2)
                gate = helper.assess(before, helper.proc_snapshot(sorted(selected)))
                event(events, event='RESOURCE_GATE', unit_id=unit['job']['unit_id'], evidence=gate)
                if not gate['pass']:
                    time.sleep(15)
                    continue
                event(events, event='QUALITY_PROFILE_START', unit_id=unit['job']['unit_id'])
                record = process(manifest, unit, source, worker, evaluator, profiler, binding)
                write(Path(manifest['output_root'])/'units'/unit['job']['unit_id']/'result.json', record)
                event(root/'results.jsonl', **record)
                event(events, event='UNIT_TERMINAL', unit_id=unit['job']['unit_id'], status=record['status'])
                pending.remove(unit)
        status = dict(status='COMPLETE_TERMINAL', device=name)
    except Exception as exc:
        status = dict(status='LANE_STOPPED', device=name, error=repr(exc), traceback=traceback.format_exc())
    write(root/'completion.json', status)
    return 0 if status['status'] == 'COMPLETE_TERMINAL' else 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--lane')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    validate(manifest, args.manifest, args.manifest_sha256)
    if not args.execute:
        print(json.dumps(dict(status='DRY_RUN', units=130, profile_slots=100, database_calls=0)))
        return 0
    if args.lane:
        require(args.lane in manifest['resources'], 'registered quality lane')
        return lane(manifest, args.lane)
    locks = []
    for path in manifest['lock_paths']:
        lock = Path(path).open('a+')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(lock)
    root = Path(manifest['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    write(root/'dispatch.json', dict(pid=os.getpid(), started_unix=time.time(), manifest_sha256=args.manifest_sha256))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4')
    children = []
    for name in manifest['resources']:
        command = [sys.executable, '-B', str(Path(__file__).resolve()), '--manifest', str(args.manifest), '--manifest-sha256', args.manifest_sha256, '--execute', '--lane', name]
        with (root/(name+'.stdout.log')).open('x') as stdout, (root/(name+'.stderr.log')).open('x') as stderr:
            child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                     start_new_session=True, pass_fds=tuple(lock.fileno() for lock in locks))
        children.append((name, child))
    outcomes = [dict(lane=name, pid=child.pid, exit_code=child.wait()) for name, child in children]
    write(root/'completion.json', dict(lanes=outcomes, ended_unix=time.time()))
    return int(any(row['exit_code'] != 0 for row in outcomes))

if __name__ == '__main__':
    raise SystemExit(main())
