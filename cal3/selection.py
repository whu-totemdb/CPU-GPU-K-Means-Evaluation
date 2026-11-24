"""Frozen metadata, profile, and calibration policies.

No routine here trains a clustering model or creates a sampling seed. CLI inputs
are JSON/CSV; every relative path is relative to the JSON configuration file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
from pathlib import Path


class Blocked(ValueError):
    """An explicit evidence/protocol prerequisite is unavailable."""


def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _unique(rows):
    result = {}
    for row in rows:
        cid = row['candidate_id']
        if cid in result:
            raise Blocked('duplicate candidate row: ' + cid)
        result[cid] = row
    return result


def _order(config):
    ids = config['registry_order']
    if not ids or len(set(ids)) != len(ids):
        raise Blocked('registry_order must contain distinct frozen IDs')
    baseline = config['baseline_order']
    if set(baseline) != set(ids) or len(baseline) != len(ids):
        raise Blocked('baseline_order must be the full frozen development ranking')
    return ids


def screen_candidates(metadata_rows, config):
    """Only compatibility + frozen metadata_score are read, never outcomes.

    Rows must provide compatible=true/false, explicitly. The first statically
    compatible development baseline is protected. Outcome eligibility is checked
    later; a failed trained baseline cannot be silently replaced by another one.
    """
    ids = _order(config)
    rows = _unique(metadata_rows)
    if set(rows) != set(ids):
        raise Blocked('metadata rows must cover the explicitly named candidate universe')
    for row in rows.values():
        if not isinstance(row.get('compatible'), bool):
            raise Blocked('compatibility evidence missing for ' + row['candidate_id'])
    allowed = [cid for cid in ids if rows[cid]['compatible']]
    fixed_baseline = config.get('fixed_baseline')
    if fixed_baseline is not None:
        if fixed_baseline not in allowed:
            raise Blocked('the specified fixed baseline is not compatible')
        baseline = fixed_baseline
    else:
        baseline = next((c for c in config['baseline_order'] if c in allowed), None)
    if baseline is None:
        raise Blocked('no statically compatible development baseline')
    if any(not _finite(rows[c].get('metadata_score')) for c in allowed):
        raise Blocked('missing frozen-model prediction; no substitute ranking')
    m = int(config['M'])
    if m < 1:
        raise Blocked('M must be positive')
    ranked = sorted(allowed, key=lambda c: (-float(rows[c]['metadata_score']), ids.index(c)))
    selected = [baseline] + [c for c in ranked if c != baseline][:m - 1]
    return {'baseline': baseline, 'metadata_order': ranked, 'screened': selected,
            'requested_M': m, 'actual_M': len(selected), 'candidate_universe': ids,
            'compatible_candidate_count': len(allowed), 'compatible_candidate_ids': allowed,
            'screen_exhausts_compatible_pool': len(selected) == len(allowed),
            'requested_M_exceeds_compatible_pool': m > len(allowed),
            'identity_group': ('ALL_' + str(len(allowed)) + '_COMPATIBLE_CANDIDATES') if m >= len(allowed) else None,
            'budget_reduction_claim': 'ACTION_COUNT_ONLY_REFERENCE_TRAINING_NOT_YET_ACCOUNTED',
            'scope': config.get('candidate_scope', 'EXPLICIT_CONFIG_UNIVERSE')}


def select_sources(metadata_rows, profile_rows, config):
    """Fix source trials before calibration; profile rows carry quality status.

    Profile row: candidate_id, eligibility='ELIGIBLE', W, probe,
    route_recall, optional training_seconds/objective/estimated_scan.
    Missing required quality evidence blocks; recorded ineligibility remains a
    negative record. Strong baselines operate in their declared available pool.
    """
    pre = screen_candidates(metadata_rows, config)
    ids = pre['candidate_universe']
    prof = _unique(profile_rows)
    missing = [c for c in pre['screened'] if c not in prof or
               prof[c].get('eligibility') in (None, '', 'NEEDS_EVIDENCE', 'NOT_MEASURED')]
    if missing:
        raise Blocked('missing quality/profile evidence: ' + ','.join(missing))
    b = pre['baseline']
    if prof[b]['eligibility'] != 'ELIGIBLE':
        raise Blocked('protected baseline is outcome-ineligible; no post-hoc substitution')
    target = float(config['target'])
    ranking_field = config.get('ranking_field', 'W')
    if ranking_field not in ('W', 'estimated_scan'):
        raise Blocked('unknown frozen ranking field')
    valid = [c for c in pre['screened'] if prof[c]['eligibility'] == 'ELIGIBLE'
             and _finite(prof[c].get(ranking_field)) and _finite(prof[c].get('probe'))
             and _finite(prof[c].get('route_recall'))
             and float(prof[c]['route_recall']) >= target]
    ranking = sorted(valid, key=lambda c: (float(prof[c][ranking_field]), ids.index(c)))
    budget = int(config.get('index_trials', 3))
    if budget < 1:
        raise Blocked('index_trials must be positive')
    trials = [b] + [c for c in ranking if c != b][:budget - 1]
    p = ranking[0] if ranking else None
    eligible = [c for c in ids if c in prof and prof[c].get('eligibility') == 'ELIGIBLE']
    baselines = {}
    for policy, field in [('fastest_qualified_training', 'training_seconds'),
                          ('minimum_objective', 'objective'),
                          ('minimum_estimated_scan', 'estimated_scan')]:
        available = [c for c in eligible if _finite(prof[c].get(field))]
        # Missing labels cannot be discarded while calling the remaining minimum full.
        baselines[policy] = {'candidate_id': min(available, key=lambda c: (float(prof[c][field]), ids.index(c))) if available and len(available) == len(eligible) else None,
                             'pool': eligible, 'missing': sorted(set(eligible) - set(available)),
                             'scope': pre['scope']}
        if policy == 'fastest_qualified_training':
            domains = {prof[c].get('training_time_hardware_domain') for c in eligible}
            matched = len(domains) == 1 and None not in domains and '' not in domains
            baselines[policy].update(training_time_hardware_domains=sorted(d for d in domains if d),
                matched_training_hardware=matched,
                training_time_scopes={c: prof[c].get('training_time_scope') for c in eligible})
            if not matched:
                baselines[policy]['candidate_id'] = None
                baselines[policy]['blocked_reason'] = 'MIXED_OR_UNKNOWN_TRAINING_TIME_HARDWARE'
    for policy, cid in config.get('fixed_sources', {}).items():
        baselines[policy] = {'candidate_id': cid if cid in eligible else None,
                             'requested_candidate': cid, 'pool': eligible, 'scope': pre['scope']}
    metadata_trials = [b] + [c for c in pre['metadata_order'] if c in eligible and c in pre['screened'] and c != b][:budget - 1]
    pre.update(ranking_field=ranking_field, profile_order=ranking, index_trials=trials, eligible_sources=eligible,
               eligibility_by_candidate={c: prof.get(c, {}).get('eligibility', 'UNKNOWN') for c in ids},
               profile_action={'candidate_id': p, 'probe': int(prof[p]['probe'])} if p else None,
               metadata_trials=metadata_trials, strong_baselines=baselines,
               exclusions=[dict(candidate_id=c, eligibility=prof[c].get('eligibility'))
                           for c in pre['screened'] if prof[c].get('eligibility') != 'ELIGIBLE'],
               profile_rows_used=[prof[c] for c in pre['screened']],
               target=target, evaluation_inputs_read=False)
    pre['precalibration_sha256'] = hashlib.sha256(json.dumps(pre, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return pre


def _calibration_choices(rows, config):
    """First recall-qualified grid probe, with every timed round qualifying.

    Flat summaries must contain `split=calibration`, `minimum_pass_recall`,
    and the frozen latency metric. A failing timed-round status is retained and
    does not cause switching to a later probe after the original grid qualifies.
    """
    field = config.get('latency_field', 'latency')
    target = float(config['target'])
    grid = [int(p) for p in config['probe_grid']]
    if grid != sorted(set(grid)) or not grid or grid[0] < 1:
        raise Blocked('probe_grid must be frozen, positive and ascending')
    grouped = {}
    for row in rows:
        if row.get('split') != 'calibration':
            raise Blocked('only explicit calibration rows may reach source selection')
        key = (row['candidate_id'], int(row['probe']))
        if key in grouped:
            raise Blocked('duplicate calibration candidate/probe; aggregate timed rounds first')
        grouped[key] = row
    result = {}
    for cid in {c for c, _ in grouped}:
        chosen = None
        status = 'RECALL_NOT_REACHED'
        for probe in grid:
            row = grouped.get((cid, probe))
            if row is None:
                status = 'NOT_MEASURED_CALIBRATION_GRID'
                break
            rec = row.get('grid_recall', row.get('minimum_pass_recall'))
            if not _finite(rec):
                status = row.get('status', 'INVALID_NUMERIC')
                break
            if float(rec) < target:
                continue
            if not _finite(row.get('minimum_pass_recall')) or float(row['minimum_pass_recall']) < target:
                status = 'TIMED_CALIBRATION_RECALL_NOT_REACHED'
                break
            if not _finite(row.get(field)) or float(row[field]) <= 0:
                status = 'MISSING_CALIBRATION_TIMING'
                break
            chosen = dict(candidate_id=cid, probe=probe, latency=float(row[field]),
                          minimum_pass_recall=float(row['minimum_pass_recall']), status='SELECTED')
            break
        result[cid] = chosen or {'candidate_id': cid, 'probe': None, 'status': status}
    return result


def choose_actions(calibration_rows, precal, config):
    """No evaluation labels accepted. Returns frozen deployment action records."""
    ids = _order(config)
    calibrated = _calibration_choices(calibration_rows, config)
    b = precal['baseline']

    def single(cid):
        if cid is None:
            return {'candidate_id': None, 'probe': None, 'status': 'NEEDS_EVIDENCE'}
        return dict(calibrated.get(cid, {'candidate_id': cid, 'probe': None, 'status': 'NOT_MEASURED'}))

    def competition(candidates):
        items = [single(cid) for cid in candidates]
        missing = [a for a in items if a['status'] not in ('SELECTED', 'RECALL_NOT_REACHED', 'TIMED_CALIBRATION_RECALL_NOT_REACHED')]
        if missing:
            return {'candidate_id': None, 'probe': None, 'status': 'INCOMPLETE_CALIBRATION', 'missing': missing}
        ok = [a for a in items if a['status'] == 'SELECTED']
        if not ok:
            return {'candidate_id': None, 'probe': None, 'status': 'RECALL_NOT_REACHED'}
        return dict(min(ok, key=lambda a: (a['latency'], a['candidate_id'] != b, ids.index(a['candidate_id']))))

    p = precal['profile_action']
    actions = {'B': single(b),
               'P': dict(p, status='PROFILE_RECOMMENDATION_NOT_RECALL_GUARANTEE') if p else single(None),
               'P+Cal': single(p['candidate_id'] if p else None),
               'Cal3' if int(config.get('index_trials', 3)) == 3 and not config.get('policy_variant') else 'CalBudget': competition(precal['index_trials']),
               'metadata_only_same_budget': competition(precal['metadata_trials'])}
    for name, decision in precal.get('strong_baselines', {}).items():
        actions[name] = single(decision['candidate_id'])
    return {'actions': actions, 'calibration_outcomes': calibrated,
            'precalibration_sha256': precal['precalibration_sha256'],
            'latency_field': config.get('latency_field', 'latency'),
            'policy_variant': config.get('policy_variant'),
            'target': float(config['target']), 'evaluation_inputs_read': False}


def frozen_metadata_predictions(metadata, registry, model_path, screen_path, expected_hashes):
    """Use the recovered original screen encoder and final model byte-for-byte."""
    for name, path in [('model', model_path), ('screen', screen_path)]:
        if not Path(path).is_file() or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected_hashes[name]:
            raise Blocked(name + ' missing or frozen SHA256 mismatch')
    spec = importlib.util.spec_from_file_location('frozen_original_screen', screen_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = json.loads(Path(model_path).read_text(encoding='utf-8'))
    model = raw['final_model'] if 'final_model' in raw else raw
    module.validate_registry(registry)
    return module.predict(model, metadata, registry)


def calibration_prefix_summary(request_rows, query_prefix, rounds, target,
                               latency_rule='median_per_query_medians'):
    """Recompute replay from recorded requests, preserving missing round/query.

    Request rows: split,candidate_id,probe,round,query_id,hits,k,latency.
    Warmups and grid rounds must be tagged role; only timed rows enter latency.
    This is labelled replay and does not pretend to measure reduced-run wall time.
    """
    if latency_rule not in ('median_per_query_medians', 'median_round_means'):
        raise Blocked('unknown frozen calibration latency rule')
    groups = {}
    required = {(int(r), str(q)) for r in rounds for q in query_prefix}
    for row in request_rows:
        if row.get('split') != 'calibration':
            raise Blocked('evaluation queries cannot enlarge calibration')
        if row.get('role') != 'timed' or (int(row['round']), str(row['query_id'])) not in required:
            continue
        key = (row['candidate_id'], int(row['probe']))
        rid = (int(row['round']), str(row['query_id']))
        if rid in groups.setdefault(key, {}):
            raise Blocked('duplicate calibration request')
        groups[key][rid] = row
    result = []
    for (cid, probe), records in groups.items():
        if set(records) != required:
            result.append(dict(candidate_id=cid, probe=probe, split='calibration', status='NOT_MEASURED_COMPLETE_PREFIX', minimum_pass_recall=None, latency=None))
            continue
        recalls = [sum(int(records[(int(r), str(q))]['hits']) for q in query_prefix) /
                   sum(int(records[(int(r), str(q))]['k']) for q in query_prefix) for r in rounds]
        per_query = [statistics.median(float(records[(int(r), str(q))]['latency']) for r in rounds) for q in query_prefix]
        latency = statistics.median(per_query) if latency_rule == 'median_per_query_medians' else statistics.median(
            statistics.mean(float(records[(int(r), str(q))]['latency']) for q in query_prefix) for r in rounds)
        result.append(dict(candidate_id=cid, probe=probe, split='calibration',
                           minimum_pass_recall=min(recalls), latency=latency, latency_rule=latency_rule,
                           status='RECALL_QUALIFIED' if min(recalls) >= target else 'RECALL_NOT_REACHED',
                           replay=True, query_count=len(query_prefix), timing_rounds=len(rounds)))
    return result


def _read(path):
    p = Path(path)
    if p.suffix.lower() == '.csv':
        with p.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if 'compatible' in row:
                if row['compatible'].lower() not in ('true', 'false'):
                    raise Blocked('CSV compatible must be true or false')
                row['compatible'] = row['compatible'].lower() == 'true'
        return rows
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cfg = _read(args.config)
    if args.output.exists():
        raise FileExistsError('preserve previous decision outputs')
    root = args.config.resolve().parent
    rows = lambda key: _read(root / cfg[key])
    try:
        pre = select_sources(rows('metadata_rows'), rows('profile_rows'), cfg)
        result = {'precalibration': pre}
        if cfg.get('calibration_rows'):
            result['deployment'] = choose_actions(rows('calibration_rows'), pre, cfg)
    except Blocked as exc:
        result = {'status': 'BLOCKED', 'reason': str(exc), 'new_measurements': 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    if result.get('status') == 'BLOCKED':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
