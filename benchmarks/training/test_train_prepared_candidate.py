"""Toy-only tests: no native adapter, KMeans, GPU, SQL or source dataset calls."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('prepared_worker', HERE / 'train_prepared_candidate.py')
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)
NORMALIZER = HERE.parent / 'assets/materialize_exact_gt.py'

class Dataset:
    def __init__(self, x): self.x = x; self.shape=x.shape; self.dtype=x.dtype
    def __getitem__(self, key): return self.x[key].copy()

class FakeH5:
    def __init__(self, x, applied=False): self.d=Dataset(x); self.attrs={'normalization_applied':applied,'normalization_required':True}
    def __enter__(self): return self
    def __exit__(self,*_): pass
    def __contains__(self,k): return k=='train'
    def __getitem__(self,k): return self.d

class PreparedWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup); self.root=Path(self.temp.name)
        # Real frozen normalizer source body; only synthetic arrays are passed.
        self.normalizer=w.load_normalization_function(NORMALIZER)
        self.x=np.asarray([[3,4],[0,2],[8,6]],dtype=np.float32)
        effective=self.x.copy();self.normalizer(effective,True,'toy')
        base=self.root/'base.hdf5';base.write_bytes(b'TOY FILE IDENTITY; h5 reading mocked')
        r={'asset_id':'toy','status':'MATERIALIZED_EXACT_TOP10','n_base':3,'d':2,'metric':'ip','normalized':True,'raw_base_array_sha256':w.array_sha(self.x),'base_array_sha256':w.array_sha(effective),'base_source_ids_array_sha256':'row-order','files':{'base.hdf5':w.sha(base)},'base_selection':{'mode':'already_selected_ascending_source_ids'}}
        rp=self.root/'asset_receipt.json';rp.write_text(json.dumps(r))
        self.u={'asset_id':'toy','n':3,'d':2,'k':2,'metric':'ip','normalize':True,'base_path':str(base),'base_sha256':w.sha(base),'base_raw_array_sha256':r['raw_base_array_sha256'],'base_effective_array_sha256':r['base_array_sha256'],'base_source_ids_array_sha256':'row-order','asset_receipt_path':str(rp),'asset_receipt_sha256':w.sha(rp)}
    def h5(self,x=None,applied=False):
        return patch.dict(sys.modules,{'h5py':types.SimpleNamespace(File=lambda *a,**k:FakeH5(self.x if x is None else x,applied))})
    def test_prepared_all_rows_no_resampling_exact_normalization(self):
        with self.h5(): x,r=w.load_prepared(self.u,self.normalizer)
        self.assertEqual(x.shape,(3,2));self.assertEqual(w.array_sha(x),self.u['base_effective_array_sha256'])
        self.assertEqual(r['raw_array_sha256'],w.array_sha(self.x));self.assertIn('no resampling',r['selection'])
    def test_bad_shape_rejected(self):
        with self.h5(self.x[:2]),self.assertRaisesRegex(ValueError,'SHAPE_DTYPE'):w.load_prepared(self.u,self.normalizer)
    def test_reordered_rows_rejected(self):
        with self.h5(self.x[::-1]),self.assertRaisesRegex(ValueError,'RAW_ARRAY_SHA'):w.load_prepared(self.u,self.normalizer)
    def test_double_normalization_rejected(self):
        with self.h5(applied=True),self.assertRaisesRegex(ValueError,'BASE_NOT_RAW'):w.load_prepared(self.u,self.normalizer)
    def test_zero_normalized_vector_rejected(self):
        with self.assertRaisesRegex(ValueError,'zero-norm'): self.normalizer(np.zeros((1,2),dtype=np.float32),True,'toy')
    def test_no_overwrite(self):
        p=self.root/'result.json';w.write_new(p,{'a':1})
        with self.assertRaises(FileExistsError):w.write_new(p,{'a':2})
        self.assertEqual(w.read(p),{'a':1})
    def test_unfrozen_rejected_before_file_reads(self):
        with self.assertRaisesRegex(ValueError,'FROZEN_MANIFEST'):w.validate({'namespace':'FINAL_VLDB_V35','frozen':False}, {})
    def test_lane_binding_requires_exact_assignment(self):
        m={'resources':{'cpu1':{'cpus':[20,21,22,23],'numa':0}}}
        u={'device':'cpu','resource_lane':'cpu1','assigned_cpu_list':[20,21,22,23],'assigned_numa':0}
        self.assertEqual(w.resource_binding(m,u)['numa'],0)
        u['assigned_cpu_list']=[16,17,18,19]
        with self.assertRaisesRegex(ValueError,'JOB_RESOURCE'):w.resource_binding(m,u)
    def test_binding_failure_not_algorithm_negative(self):
        self.assertEqual(w.classify_failure(ValueError('hash'),'input_preparation'),'MEASUREMENT_FAILED')
        self.assertEqual(w.classify_failure(MemoryError(),'adapter'),'OOM')
    def test_native_prepared_adapter_arguments_and_sampling(self):
        binary=self.root/'native';binary.write_bytes(b'NOT EXECUTABLE; mock subprocess only')
        u=dict(self.u,algorithm_options={'sampling_fraction':1.0,'threads':4,'keep_prepared_input':False},execution_binding={'super_runner':str(binary)},cpu_threads=4,max_iter=20,seed=20260805)
        x=self.x.copy();self.normalizer(x,True,'toy')
        r=types.SimpleNamespace(sampling_cardinality=lambda **k:{'unsupported':k['n']*k['sampling_fraction']<k['k']},_last_json=lambda s:json.loads(s))
        def fake_run(command,**kwargs):
            self.assertIn('--angular',command);self.assertEqual(command[command.index('--input')+1],str(self.root/'prepared_effective.f32'))
            self.assertEqual(command[command.index('--sampling-fraction')+1],'1.0')
            self.assertEqual(w.sha(command[command.index('--input')+1]),u['base_effective_array_sha256'])
            np.asarray([[.6,.8],[0,1]],dtype=np.float32).tofile(command[command.index('--centroids-out')+1])
            return types.SimpleNamespace(returncode=0,stdout=json.dumps({'status':'OK','t_train_wall':.1,'n_iter':2}),stderr='')
        m={'source_sha256':{str(binary):w.sha(binary)}}
        with patch.object(w.subprocess,'run',fake_run): d=w.native_super(r,m,u,x,self.root)
        self.assertEqual(d['centroids'].shape,(2,2));self.assertTrue((self.root/'native_centroids.f32').exists());self.assertFalse((self.root/'prepared_effective.f32').exists())
        u['algorithm_options']['sampling_fraction']=.3
        with self.assertRaisesRegex(ValueError,'SAMPLING_UNSUPPORTED'):w.native_super(r,m,u,x,self.root)

if __name__=='__main__':unittest.main()
