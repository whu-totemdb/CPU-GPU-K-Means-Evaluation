"""Nested-prefix geometry profiles using the recovered original work formula.

These are estimates of routing and scanned vectors, never measured database
scans. Explicit immutable index prefixes replace any fresh sampling RNG.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def routes(x, centroids, metric, block=256):
    """FP64 distances, stable centroid-ID ties, matching original profile code."""
    x, c = np.asarray(x), np.asarray(centroids)
    if metric not in ('l2', 'ip') or x.ndim != 2 or c.ndim != 2 or x.shape[1] != c.shape[1]:
        raise ValueError('invalid metric or vector shape')
    if not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError('nonfinite vectors')
    if not len(c) or not len(x) or block < 1:
        raise ValueError('nonempty arrays and positive block size required')
    b = c.astype(np.float64)
    out = []
    for start in range(0, len(x), block):
        a = x[start:start + block].astype(np.float64)
        distances = -(a @ b.T) if metric == 'ip' else (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
        out.append(np.argsort(distances, axis=1, kind='stable'))
    return np.concatenate(out)


def _unit_norm(name, a, tolerance):
    norms = np.linalg.norm(np.asarray(a, dtype=np.float64), axis=-1)
    if np.any(norms == 0) or not np.all(np.isfinite(norms)):
        raise ValueError(name + ': zero/nonfinite norm in normalized-IP condition')
    if np.max(np.abs(norms - 1)) > tolerance:
        raise ValueError(name + ': not normalized; profile never silently transforms assets')


def geometry_profile(data, queries, gt_vectors, centroids, n, metric, grid,
                     target=.95, block=256, normalized_ip=False, norm_tolerance=1e-5):
    start = time.perf_counter()
    data, queries, gt_vectors, centroids = map(np.asarray, (data, queries, gt_vectors, centroids))
    if len(data) > n or gt_vectors.shape != (len(queries), 10, centroids.shape[1]):
        raise ValueError('profile requires actual N and exact top10 neighbor vectors')
    if not 0 < target <= 1 or sorted(set(grid)) != list(grid) or any(p < 1 or p > len(centroids) for p in grid):
        raise ValueError('invalid target/probe grid')
    if metric == 'ip':
        if not normalized_ip:
            raise ValueError('IP protocol must explicitly confirm normalized-IP semantics')
        for name, a in [('base', data), ('queries', queries), ('GT neighbors', gt_vectors), ('centroids', centroids)]:
            _unit_norm(name, a, norm_tolerance)
    assignments = routes(data, centroids, metric, block)[:, 0]
    counts = np.bincount(assignments, minlength=len(centroids)).astype(float)
    sizes = counts / len(data) * n
    query_routes = routes(queries, centroids, metric, block)
    gt_lists = routes(gt_vectors.reshape(-1, centroids.shape[1]), centroids, metric, block)[:, 0].reshape(len(queries), 10)
    curves = []
    inverse = np.empty_like(query_routes)
    np.put_along_axis(inverse, query_routes, np.arange(len(centroids))[None, :], axis=1)
    gt_ranks = np.take_along_axis(inverse, gt_lists, axis=1)
    for probe in grid:
        selected = query_routes[:, :probe]
        # Search the rank of each GT list to avoid Q x probe x 10 temporary.
        hit = gt_ranks < probe
        work = sizes[selected].sum(1)
        curves.append({'probe': int(probe), 'route_recall': float(hit.mean()),
                       'scan_mean': float(work.mean()), 'scan_p90': float(np.quantile(work, .90)),
                       'scan_p99': float(np.quantile(work, .99))})
    point = next((r for r in curves if r['route_recall'] >= target), None)
    k, d = centroids.shape
    sorted_sizes = np.sort(sizes)
    gini = float(np.sum((2 * np.arange(1, k + 1) - k - 1) * sorted_sizes) / (k * sorted_sizes.sum()))
    top_count = max(1, math.ceil(.01 * k))
    result = {'kind': 'SAMPLED_GEOMETRY_PROXY_UNVERIFIED_ROUTING_EQUIVALENCE',
              'centroid_payload_sha256': hashlib.sha256(centroids.tobytes()).hexdigest(),
              'n': int(n), 'd': int(d), 'nlist': int(k), 'metric': metric,
              'data_sample_count': len(data), 'query_count': len(queries),
              'target': target, 'tie_policy': 'ascending_centroid_id_proxy',
              'list_cv': float(sizes.std() / sizes.mean()),
              'list_max_over_mean': float(sizes.max() / sizes.mean()),
              'sample_empty_fraction': float((counts == 0).mean()), 'sample_list_gini': gini,
              'sample_top1pct_mass': float(sorted_sizes[-top_count:].sum() / sizes.sum()),
              'top1pct_list_count': top_count, 'top1pct_actual_fraction': top_count / k,
              'curves': curves, 'proxy_point': point, 'actual_database_scan': None,
              'status': 'PROFILE_QUALIFIED' if point else 'PROFILE_RECALL_NOT_REACHED',
              'compute_seconds': time.perf_counter() - start}
    result.update({'probe': point['probe'] if point else None,
                   'route_recall': point['route_recall'] if point else None,
                   'estimated_scan': point['scan_mean'] if point else None,
                   'W': k * d + point['probe'] + d * point['scan_mean'] if point else None})
    return result


def nested_profiles(base, calibration_queries, ground_truth, centroids, base_order,
                    query_order, base_counts, query_counts, metric, grid, target=.95,
                    normalized_ip=False, evaluation_query_ids=None, calibration_query_ids=None):
    """All prefixes are user-frozen unique row indices; never create a seed."""
    bo, qo = np.asarray(base_order), np.asarray(query_order)
    for name, order, limit in [('base', bo, len(base)), ('query', qo, len(calibration_queries))]:
        if order.ndim != 1 or order.dtype.kind not in 'iu' or len(np.unique(order)) != len(order) or np.any(order < 0) or np.any(order >= limit):
            raise ValueError('invalid frozen ' + name + ' index prefix')
    if max(base_counts) > len(bo) or max(query_counts) > len(qo):
        raise ValueError('nested sample budget exceeds available frozen prefix')
    if any(v < 1 for v in [*base_counts, *query_counts]):
        raise ValueError('empty nested profile condition')
    if evaluation_query_ids is not None:
        if calibration_query_ids is None or len(calibration_query_ids) != len(calibration_queries):
            raise ValueError('query-ID split evidence required')
        if set(map(str, calibration_query_ids)).intersection(map(str, evaluation_query_ids)):
            raise ValueError('calibration and final-evaluation queries overlap')
    gt = np.asarray(ground_truth)
    if gt.shape != (len(calibration_queries), 10) or gt.dtype.kind not in 'iu' or np.any(gt < 0) or np.any(gt >= len(base)):
        raise ValueError('GT must be exact-top10 base row IDs for all calibration queries')
    rows = []
    for nb, nq in [(b, q) for b in base_counts for q in query_counts]:
        qb = qo[:nq]
        row = geometry_profile(base[bo[:nb]], calibration_queries[qb], base[gt[qb]], centroids,
                               len(base), metric, grid, target, normalized_ip=normalized_ip)
        row['base_prefix_sha256'] = hashlib.sha256(bo[:nb].astype('<i8').tobytes()).hexdigest()
        row['query_prefix_sha256'] = hashlib.sha256(qb.astype('<i8').tobytes()).hexdigest()
        row['sampling_rule'] = 'frozen_nested_prefix_no_new_rng'
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve earlier profiles')
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    root = args.config.resolve().parent

    def load(key):
        asset = root / cfg[key]
        if 'sha256' not in cfg or key not in cfg['sha256']:
            raise ValueError('frozen asset SHA256 required for ' + key)
        digest = None
        if hasattr(hashlib, 'file_digest'):
            with asset.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
        if digest is None:
            h = hashlib.sha256()
            with asset.open('rb') as f:
                for chunk in iter(lambda: f.read(8 << 20), b''):
                    h.update(chunk)
            digest = h.hexdigest()
        if digest != cfg['sha256'][key]:
            raise ValueError('asset SHA256 mismatch: ' + key)
        return np.load(asset, mmap_mode='r', allow_pickle=False)

    rows = nested_profiles(load('base'), load('calibration_queries'), load('ground_truth'), load('centroids'),
                           load('base_order'), load('query_order'), cfg['base_counts'], cfg['query_counts'],
                           cfg['metric'], cfg['probe_grid'], cfg['target'], cfg.get('normalized_ip', False),
                           evaluation_query_ids=cfg['evaluation_query_ids'], calibration_query_ids=cfg['calibration_query_ids'])
    for row in rows:
        row.update(candidate_id=cfg['candidate_id'], task_id=cfg['task_id'], eligibility=cfg['eligibility'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'profiles': rows, 'new_training_calls': 0, 'source_asset_sha256': cfg['sha256']}, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
