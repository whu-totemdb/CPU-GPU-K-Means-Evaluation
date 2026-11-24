"""DEVELOPMENT FIXTURES ONLY: no scientific measurements or clustering seeds."""
import copy
import unittest

import numpy as np

from cal3.profile import geometry_profile, nested_profiles
from cal3.selection import (
    Blocked, select_sources, choose_actions, calibration_prefix_summary,
    screen_candidates,
)


def fixture():
    cfg = {'registry_order': ['a', 'b', 'c', 'd'], 'baseline_order': ['b', 'a', 'c', 'd'],
           'M': 3, 'index_trials': 3, 'target': .95, 'probe_grid': [1, 2],
           'candidate_scope': 'DEVELOPMENT_FIXTURE_FOUR', 'latency_field': 'latency'}
    meta = [{'candidate_id': c, 'compatible': True, 'metadata_score': s} for c, s in zip('abcd', [4, 1, 3, 2])]
    prof = [{'candidate_id': c, 'eligibility': 'ELIGIBLE', 'W': w, 'probe': 1,
             'route_recall': .96, 'training_seconds': t, 'objective': o, 'estimated_scan': w,
             'training_time_hardware_domain': 'DEVELOPMENT_SAME_HOST', 'training_time_scope': 'fixture'}
            for c, w, t, o in [('a', 20, 4, 3), ('b', 30, 2, 4), ('c', 10, 3, 2), ('d', 5, 1, 1)]]
    cal = []
    for c in 'abcd':
        for probe in [1, 2]:
            cal.append({'candidate_id': c, 'probe': probe, 'split': 'calibration',
                        'minimum_pass_recall': .90 if probe == 1 else .96,
                        'latency': {'a': 2, 'b': 1, 'c': 1, 'd': .5}[c]})
    return cfg, meta, prof, cal


class SelectionTests(unittest.TestCase):
    def test_fixed_baseline_is_not_replaced(self):
        cfg, meta, _, _ = fixture()
        cfg["fixed_baseline"] = "a"
        self.assertEqual(screen_candidates(meta, cfg)["baseline"], "a")
        meta[0]["compatible"] = False
        with self.assertRaises(Blocked):
            screen_candidates(meta, cfg)

    def test_scan_only_ranking_is_explicit(self):
        cfg, meta, prof, _ = fixture()
        cfg["index_trials"] = 2
        prof[0]["estimated_scan"] = 3
        self.assertEqual(select_sources(meta, prof, cfg)["index_trials"], ["b", "c"])
        cfg["ranking_field"] = "estimated_scan"
        self.assertEqual(select_sources(meta, prof, cfg)["index_trials"], ["b", "a"])
        cfg["ranking_field"] = "qps"
        with self.assertRaises(Blocked):
            select_sources(meta, prof, cfg)

    def test_fastest_training_rejects_mixed_and_unknown_hardware(self):
        cfg, meta, prof, cal = fixture()
        pre = select_sources(meta, prof, cfg)
        self.assertEqual(pre['strong_baselines']['fastest_qualified_training']['candidate_id'], 'd')
        prof[-1]['training_time_hardware_domain'] = 'OTHER_MACHINE'
        pre = select_sources(meta, prof, cfg)
        self.assertIsNone(pre['strong_baselines']['fastest_qualified_training']['candidate_id'])
        self.assertEqual(pre['strong_baselines']['minimum_objective']['candidate_id'], 'd')
        prof[-1]['training_time_hardware_domain'] = None
        pre = select_sources(meta, prof, cfg)
        self.assertFalse(pre['strong_baselines']['fastest_qualified_training']['matched_training_hardware'])

    def test_budget_variants_do_not_reuse_original_cal3_name(self):
        cfg, meta, prof, cal = fixture()
        cfg['policy_variant'] = 'M_3'
        actions = choose_actions(cal, select_sources(meta, prof, cfg), cfg)['actions']
        self.assertIn('CalBudget', actions)
        self.assertNotIn('Cal3', actions)

    def test_seven_compatible_sources_make_m10_and_m15_identity(self):
        ids = [str(i) for i in range(15)]
        meta = [{'candidate_id': cid, 'compatible': i < 7, 'metadata_score': -i} for i, cid in enumerate(ids)]
        cfg = {'registry_order': ids, 'baseline_order': ids, 'M': 10}
        a = screen_candidates(meta, cfg)
        b = screen_candidates(meta, dict(cfg, M=15))
        self.assertEqual(a['actual_M'], 7)
        self.assertEqual(a['screened'], b['screened'])
        self.assertEqual(a['identity_group'], 'ALL_7_COMPATIBLE_CANDIDATES')
        self.assertTrue(a['requested_M_exceeds_compatible_pool'])


    def test_original_policy_semantics_and_baseline_tie(self):
        cfg, meta, prof, cal = fixture()
        pre = select_sources(meta, prof, cfg)
        self.assertEqual(pre['screened'], ['b', 'a', 'c'])
        self.assertEqual(pre['index_trials'], ['b', 'c', 'a'])
        actions = choose_actions(cal, pre, cfg)['actions']
        self.assertEqual(actions['Cal3']['candidate_id'], 'b')
        self.assertEqual(actions['P']['candidate_id'], 'c')
        self.assertEqual(actions['P']['probe'], 1)
        self.assertEqual(actions['P+Cal']['probe'], 2)
        self.assertEqual(actions['fastest_qualified_training']['candidate_id'], 'd')

    def test_test_qps_never_changes_source_decisions(self):
        cfg, meta, prof, cal = fixture()
        pre = select_sources(meta, prof, cfg)
        first = choose_actions(cal, pre, cfg)
        polluted = copy.deepcopy(cal)
        for i, row in enumerate(polluted):
            row['evaluation_qps'] = 1000000 * i
        self.assertEqual(first, choose_actions(polluted, pre, cfg))
        polluted[0]['split'] = 'evaluation'
        with self.assertRaises(Blocked):
            choose_actions(polluted, pre, cfg)

    def test_no_posthoc_baseline_substitution_or_missing_timing(self):
        cfg, meta, prof, cal = fixture()
        bad = copy.deepcopy(prof)
        bad[1]['eligibility'] = 'OOM'
        with self.assertRaises(Blocked):
            select_sources(meta, bad, cfg)
        pre = select_sources(meta, prof, cfg)
        rows = [r for r in cal if r['candidate_id'] != 'a']
        actions = choose_actions(rows, pre, cfg)['actions']
        self.assertEqual(actions['Cal3']['status'], 'INCOMPLETE_CALIBRATION')

    def test_first_grid_qualifier_failure_does_not_move_probe(self):
        cfg, meta, prof, cal = fixture()
        pre = select_sources(meta, prof, cfg)
        for row in cal:
            if row['candidate_id'] == 'c' and row['probe'] == 1:
                row['grid_recall'] = .96
                row['minimum_pass_recall'] = .94
        actions = choose_actions(cal, pre, cfg)
        self.assertEqual(actions['actions']['P+Cal']['status'], 'TIMED_CALIBRATION_RECALL_NOT_REACHED')
        self.assertIsNone(actions['actions']['P+Cal']['probe'])


    def test_calibration_prefix_missing_round_stays_missing(self):
        requests = [{'candidate_id': 'a', 'probe': 1, 'split': 'calibration', 'role': 'timed',
                     'query_id': q, 'round': r, 'hits': 10, 'k': 10, 'latency': 1 + r}
                    for q in ['x', 'y'] for r in [1, 2, 3]]
        rows = calibration_prefix_summary(requests, ['x', 'y'], [1, 2, 3], .95)
        self.assertEqual(rows[0]['latency'], 3)
        self.assertEqual(rows[0]['minimum_pass_recall'], 1)
        missing = calibration_prefix_summary(requests[:-1], ['x', 'y'], [1, 2, 3], .95)
        self.assertIsNone(missing[0]['latency'])
        with self.assertRaises(Blocked):
            calibration_prefix_summary(requests + [requests[0]], ['x', 'y'], [1, 2, 3], .95)


class ProfileTests(unittest.TestCase):
    def test_original_work_equation_and_stable_tie(self):
        base = np.array([[0.], [0.], [2.], [2.]])
        queries = np.array([[1.]])
        gt = np.zeros((1, 10, 1))
        result = geometry_profile(base, queries, gt, np.array([[0.], [2.]]), 100, 'l2', [1, 2])
        self.assertEqual(result['probe'], 1)  # midpoint tie selects centroid 0
        self.assertEqual(result['estimated_scan'], 50)
        self.assertEqual(result['W'], 53)
        self.assertIsNone(result['actual_database_scan'])

    def test_normalized_ip_rejects_zero_and_nonunit_assets(self):
        base = np.eye(2)
        gt = np.tile(base[:1], (1, 10, 1))
        with self.assertRaises(ValueError):
            geometry_profile(base, base[:1], gt, base * 2, 2, 'ip', [1, 2], normalized_ip=True)
        good = geometry_profile(base, base[:1], gt, base, 2, 'ip', [1, 2], normalized_ip=True)
        self.assertEqual(good['route_recall'], 1)

    def test_nested_prefix_and_query_split_guards(self):
        base = np.arange(20, dtype=float).reshape(10, 2)
        q = base[:2]
        gt = np.tile(np.arange(10), (2, 1))
        with self.assertRaises(ValueError):
            nested_profiles(base, q, gt, base[:2], np.arange(10), np.arange(2), [4], [1], 'l2', [1, 2],
                            evaluation_query_ids=['z'], calibration_query_ids=['x', 'z'])
        rows = nested_profiles(base, q, gt, base[:2], np.arange(10), np.arange(2), [4, 8], [1, 2], 'l2', [1, 2])
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[0]['base_prefix_sha256'], rows[1]['base_prefix_sha256'])
        self.assertEqual(rows[0]['query_prefix_sha256'], rows[2]['query_prefix_sha256'])


if __name__ == '__main__':
    unittest.main()
