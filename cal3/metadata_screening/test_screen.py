"""Pure decision/encoding checks: deliberately no fit and no inference model."""
import math,unittest
from cal3.metadata_screening.screen import public_metadata,encode_metadata,protected_top_m,choose_M,numeric_label,targets_and_weights,baseline_ranking,apply_supervision_admission,oracle_inclusion_bounds,bind_workload_metadata,task_groups

def registry():
    return [{'candidate_id':f'c{i}', 'raw_algorithm':f'alg{i}','device':'cpu' if i%2 else 'cuda','compute_dtype':'fp32','approximate':False,'options_match':{'threads':4} if i%2 else {}} for i in range(15)]
def row(task='a',family='f',candidate='c0',q='4',status='OK'):
    return dict(workload_id=task,family_id=family,candidate_id=candidate,qps_at_r95=q,terminal_status=status,utility_supervised='true',n='1000',d='16',nlist='10',metric='l2',normalization='false')

class ScreenTests(unittest.TestCase):
    def test_outcome_keys_rejected_at_prediction_boundary(self):
        r=registry();meta=public_metadata(row())
        for key in ['qps_at_r95','runtime','memory','centroids','profiles','family_id','workload_id','minimum_nprobe_at_r95']:
            with self.assertRaises(ValueError):encode_metadata(dict(meta,**{key:42}),r[0],r)
    def test_training_extract_ignores_outcomes(self):
        a=row();b=dict(a,qps_at_r95='999999',minimum_nprobe_at_r95='64',runtime='4',memory='8',profiles={'bad':1})
        self.assertEqual(public_metadata(a),public_metadata(b))
    def test_static_metadata_can_distinguish_candidates(self):
        r=registry();m=public_metadata(row());self.assertNotEqual(encode_metadata(m,r[0],r),encode_metadata(m,r[1],r))
    def test_metadata_has_no_inferred_missing_success(self):
        r=registry();m=public_metadata(row())
        with self.assertRaises(ValueError):encode_metadata(dict(m,n='nan'),r[0],r)
        with self.assertRaises(ValueError):encode_metadata(dict(m,normalization='UNKNOWN'),r[0],r)
    def test_baseline_protection_tie_and_finite(self):
        ids=[c['candidate_id'] for c in registry()];scores={c:1.0 for c in ids};scores['c0']=float('nan')
        out=protected_top_m(scores,ids,['c14']+ids,3,ids)
        self.assertEqual(out['candidates'],['c14','c1','c2'])
        self.assertEqual(protected_top_m(scores,[],ids,3,ids)['status'],'NO_ELIGIBLE_BASELINE')
    def test_negative_not_zero_target(self):
        self.assertIsNone(numeric_label(row(q='0')));self.assertIsNone(numeric_label(row(q='5',status='R95_NOT_REACHED')))
        examples=targets_and_weights([row(),row(candidate='c1',q='0')]);self.assertEqual(len(examples),1);self.assertEqual(examples[0][1],0)
    def test_weight_family_task_candidate_balance(self):
        rows=[row('a','f','c0'),row('a','f','c1'),row('b','f','c0'),row('c','g','c0')]
        ex=targets_and_weights(rows);self.assertAlmostEqual(sum(w for _,_,w in ex),1)
        self.assertAlmostEqual(sum(w for r,_,w in ex if r['family_id']=='f'),.5)
    def test_baseline_coverage_before_utility(self):
        rows=[row('a','f','c0','5'),row('a','f','c1','10'),row('b','g','c0','5'),row('b','g','c1','0')]
        self.assertEqual(baseline_ranking(rows,registry())[0]['candidate_id'],'c0')
    def test_backend_exclusion_does_not_mutate_original(self):
        original=row(candidate='c1',q='99')
        copied=apply_supervision_admission([original],{'supervision_excluded_candidates':{'c1':'BACKEND_UNKNOWN'}})[0]
        self.assertEqual(numeric_label(original),99);self.assertIsNone(numeric_label(copied))
        self.assertEqual(copied['qps_at_r95'],'99');self.assertNotIn('_supervision_eligible',original)
    def test_partial_oracle_bounds(self):
        config={'supervision_excluded_candidates':{'c2':'BACKEND_UNKNOWN'},'known_nonutility_terminal_statuses':['R95_NOT_REACHED_ON_FROZEN_GRID']}
        rows=apply_supervision_admission([row(candidate='c0',q='10'),row(candidate='c1',q='5'),row(candidate='c2',q='99')],config)
        a=oracle_inclusion_bounds(rows,['c0'],['c0','c1','c2'],config)
        self.assertEqual((a['lower'],a['upper']),(0,1));self.assertEqual(a['unknown_candidate_ids'],['c2'])
        b=oracle_inclusion_bounds(rows,['c0','c2'],['c0','c1','c2'],config)
        self.assertEqual((b['lower'],b['upper']),(1,1))
        c=oracle_inclusion_bounds(rows,['c1'],['c0','c1','c2'],config)
        self.assertEqual((c['lower'],c['upper']),(0,0))
        d=oracle_inclusion_bounds(rows,['c2'],['c0','c1','c2'],config)
        self.assertEqual((d['lower'],d['upper']),(0,1))
    def test_known_negative_is_not_unknown_or_zero(self):
        config={'known_nonutility_terminal_statuses':['R95_NOT_REACHED_ON_FROZEN_GRID']}
        rows=[row(candidate='c0',q='10'),row(candidate='c1',q='',status='R95_NOT_REACHED_ON_FROZEN_GRID')]
        a=oracle_inclusion_bounds(rows,['c0'],['c0','c1'],config)
        self.assertEqual(a['unknown_candidate_ids'],[]);self.assertEqual((a['lower'],a['upper']),(1,1))
    def test_blank_terminal_metadata_binds_without_label_imputation(self):
        original=row(q='',status='INVALID_ALGORITHM_OUTPUT_NONFINITE_CENTROIDS')
        for field in ['family_id','n','d','nlist','metric','normalization']:original[field]=''
        matrix=[dict(workload_id='a',family_id='f',n='1000',d='16',nlist='10',metric='l2',normalized='false')]
        bound,repairs=bind_workload_metadata([original],matrix)
        self.assertEqual(len(repairs),6);self.assertEqual(bound[0]['family_id'],'f')
        self.assertEqual(bound[0]['qps_at_r95'],'');self.assertEqual(bound[0]['terminal_status'],original['terminal_status'])
        self.assertEqual(original['family_id'],'');self.assertIsNone(numeric_label(bound[0]))
    def test_nonblank_binding_conflict_rejected(self):
        matrix=[dict(workload_id='a',family_id='different_family',n='1000',d='16',nlist='10',metric='l2',normalized='false')]
        with self.assertRaisesRegex(ValueError,'metadata conflict'):bind_workload_metadata([row()],matrix)
    def test_m_rule_and_fallback(self):
        self.assertEqual(choose_M({'3':.8,'5':.9,'7':.9,'10':1}),5)
        self.assertEqual(choose_M({'3':.0,'5':.0,'7':.0,'10':.0}),10)

if __name__=='__main__': unittest.main()
