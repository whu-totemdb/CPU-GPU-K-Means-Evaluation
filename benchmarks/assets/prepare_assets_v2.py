"""V35 asset preparation V2 (registered row selection repaired): bounded CPU, no K-Means/SeekDB/models.
Default is metadata dry-run. Execution requires externally frozen manifest+SHA.
Reuse locked V2 IndexFlat GT and normalization functions, never embedded GT.
"""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,resource,sys,time
SELECTION_SEED=20260914
BASE_SELECTION_SEED=20260805
RULES={'test_prefix1000','holdout_train1000_without_replacement'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def dump_new(path,value):
 with Path(path).open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
def validate(m,j,execute=False):
 if m.get('namespace')!='FINAL_VLDB_V35' or m.get('cpu_threads')!=4:raise ValueError('namespace/CPU contract')
 if j['query_rule'] not in RULES or j['source_train_rows']<1010 or j['metric'] not in ('l2','ip'):raise ValueError('invalid asset contract')
 if j['n_base']!=j['n_selected_before_holdout']-(1000 if j['query_rule'].startswith('holdout') else 0):raise ValueError('base exclusion count drift')
 selection=j['base_selection']
 if selection['source_train_rows']!=j['source_train_rows'] or selection['requested_rows']!=j['n_selected_before_holdout'] or selection['seed']!=BASE_SELECTION_SEED:raise ValueError('registered base selection drift')
 if not 0<selection['requested_rows']<=selection['source_train_rows']:raise ValueError('base selection cardinality')
 expected='all_rows' if selection['requested_rows']==selection['source_train_rows'] else 'sorted_choice_without_replacement'
 if selection['method']!=expected:raise ValueError('base selection must reproduce registered all_rows or sorted choice')
 if j['query_rule'].startswith('holdout') and selection['method']!='all_rows':raise ValueError('holdout must use authorized whole-train pool')
 if j['calibration_query_ids']!=[0,499] or j['evaluation_query_ids']!=[500,999] or j['top_k']!=10:raise ValueError('split/GT contract drift')
 if j['selection_seed']!=SELECTION_SEED:raise ValueError('selection seed drift')
 if execute:
  if not m.get('frozen') or not m.get('gt_source_sha256') or not m.get('faiss_version') or not j.get('source_sha256'):raise ValueError('unfrozen or unbound source/GT/library')
  if 'FINAL_VLDB_V35' not in Path(m['output_root']).resolve().parts:raise ValueError('output namespace')
def choose_base_ids(selection,np):
 total=selection['source_train_rows'];n=selection['requested_rows'];seed=selection['seed']
 if not 0<n<=total or seed!=BASE_SELECTION_SEED:raise ValueError('registered base count/seed')
 if selection['method']=='all_rows' and n==total:return np.arange(total,dtype=np.int64)
 if selection['method']=='sorted_choice_without_replacement' and n<total:
  return np.sort(np.random.default_rng(seed).choice(total,n,replace=False)).astype(np.int64)
 raise ValueError('unregistered base selection')
def choose_ids(n,rule,np):
 if rule=='test_prefix1000':return np.arange(1000,dtype=np.int64)
 if rule=='holdout_train1000_without_replacement':return np.random.default_rng(SELECTION_SEED).choice(n,size=1000,replace=False).astype(np.int64)
 raise ValueError('query rule')
def remaining_ids(n,query_ids,np):
 mask=np.ones(n,dtype=bool);mask[query_ids]=False
 return np.flatnonzero(mask).astype(np.int64)
def read_ordered(dataset,ids,np):
 order=np.argsort(ids,kind='stable');return np.asarray(dataset[ids[order]],dtype=np.float32,order='C')[np.argsort(order,kind='stable')]
def stats(a,np):
 finite=np.isfinite(a);norm=np.linalg.norm(a,axis=1);validnorm=norm[np.isfinite(norm)]
 return {'rows':len(a),'nonfinite_values':int((~finite).sum()),'nonfinite_rows':int((~finite).any(1).sum()),'zero_norm_rows':int((norm==0).sum()),'norm_min':float(validnorm.min()) if len(validnorm) else None,'norm_max':float(validnorm.max()) if len(validnorm) else None}
def merge_stats(target,value):
 for k in ('rows','nonfinite_values','nonfinite_rows','zero_norm_rows'):target[k]=target.get(k,0)+value[k]
 for k,fn in [('norm_min',min),('norm_max',max)]:
  if value[k] is not None:target[k]=value[k] if target.get(k) is None else fn(target[k],value[k])
def array_sha(a):return hashlib.sha256(a.tobytes(order='C')).hexdigest()
def prepare(m,j):
 validate(m,j,True)
 os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
 if m.get('cpu_list'):os.sched_setaffinity(0,set(m['cpu_list']))
 sys.path.insert(0,m['repository'])
 if m.get('numa') is not None:
  from cvpr_workspace.entrypoints.downstream_v1.run_with_linux_numa_binding_v1 import bind_memory,system_nodes
  bind_memory(m['numa'],system_nodes())
 import numpy as np,h5py,faiss,threadpoolctl
 if str(faiss.__version__)!=m['faiss_version']:raise ValueError('FAISS version drift')
 source=Path(j['source_path']);gtcode=Path(m['gt_source_path'])
 if sha(gtcode)!=m['gt_source_sha256']:raise ValueError('GT code SHA drift')
 spec=importlib.util.spec_from_file_location('v35_locked_exact_gt',gtcode);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 root=Path(m['output_root']);root.mkdir(parents=True,exist_ok=True)
 with (root/'asset_cpu.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  out=root/j['asset_id']
  if out.exists():raise FileExistsError('asset exists; no implicit overwrite/retry')
  out.mkdir();start=time.time();cost={};receipt={'asset_id':j['asset_id'],'tasks':j['task_ids'],'status':'STARTED','started':start,'source':j['source_path'],'query_rule':j['query_rule'],'selection_seed':SELECTION_SEED,'base_selection':j['base_selection'],'source_train_rows':j['source_train_rows'],'n_selected_before_holdout':j['n_selected_before_holdout'],'selection_evidence':j['selection_evidence'],'metric':j['metric'],'normalized':j['normalized'],'top_k':10,'new_db_units':0,'new_kmeans_units':0}
  try:
   begun=time.perf_counter();source_sha=sha(source);cost['source_hash_seconds']=time.perf_counter()-begun
   if source_sha!=j['source_sha256']:raise ValueError('source SHA drift')
   receipt.update(source_sha256=source_sha,source_size_bytes=source.stat().st_size,source_mtime_ns=source.stat().st_mtime_ns)
   with h5py.File(source,'r') as f,threadpoolctl.threadpool_limits(limits=4):
    if f['train'].shape!=(j['source_train_rows'],j['d']):raise ValueError('source train shape drift')
    selection_start=time.perf_counter();selected_ids=choose_base_ids(j['base_selection'],np)
    ids=choose_ids(len(selected_ids),j['query_rule'],np)
    if j['query_rule'].startswith('holdout'):
     baseids=selected_ids[remaining_ids(len(selected_ids),ids,np)];ids=selected_ids[ids]
    else:baseids=selected_ids
    cost['row_selection_seconds']=time.perf_counter()-selection_start
    if len(baseids)!=j['n_base']:raise ValueError('base cardinality drift')
    if j['query_rule'].startswith('holdout') and np.intersect1d(ids,baseids).size:raise ValueError('train/query overlap')
    qsource=f['train'] if j['query_rule'].startswith('holdout') else f['test']
    if j['query_rule']=='test_prefix1000' and qsource.shape[0]<1000:raise ValueError('fewer than1000 independent queries')
    begun=time.perf_counter();query=read_ordered(qsource,ids,np);raw_query_stats=stats(query,np)
    if raw_query_stats['nonfinite_values']:raise ValueError('nonfinite query source')
    raw_query=query.copy();qprep=module.apply_frozen_normalization(query,j['normalized'],'query');cost['query_materialization_seconds']=time.perf_counter()-begun
    # Preserve source row order; new SQL/base IDs are zero-based position in baseids.
    with (out/'base_source_ids.npy').open('xb') as ff:np.save(ff,baseids,allow_pickle=False)
    for role,sl in [('calibration',slice(0,500)),('evaluation',slice(500,1000))]:
     dest=out/role;dest.mkdir()
     for name,array in [('queries.npy',raw_query[sl]),('effective_queries.npy',query[sl]),('source_ids.npy',ids[sl]),('query_ids.npy',np.arange(1000,dtype=np.int64)[sl])]:
      with (dest/name).open('xb') as ff:np.save(ff,array,allow_pickle=False)
    begun=time.perf_counter();raw_base_stats={};effective_base_stats={};base_hash=hashlib.sha256();raw_base_hash=hashlib.sha256();index=module.exact_index(j['metric'],j['d']);faiss.omp_set_num_threads(4)
    if faiss.omp_get_max_threads()!=4:raise ValueError('FAISS thread binding failed')
    add_seconds=0.0;write_seconds=0.0
    with h5py.File(out/'base.hdf5','x') as output:
     dataset=output.create_dataset('train',shape=(j['n_base'],j['d']),dtype='float32',chunks=(min(m['block_rows'],j['n_base']),j['d']))
     for startrow in range(0,len(baseids),m['block_rows']):
      blockids=baseids[startrow:startrow+m['block_rows']];x=np.asarray(f['train'][blockids],dtype=np.float32,order='C');ss=stats(x,np);merge_stats(raw_base_stats,ss)
      if ss['nonfinite_values']:raise ValueError('nonfinite base source')
      raw_base_hash.update(x.tobytes(order='C'));tt=time.perf_counter();dataset[startrow:startrow+len(x)]=x;write_seconds+=time.perf_counter()-tt
      module.apply_frozen_normalization(x,j['normalized'],'base');merge_stats(effective_base_stats,stats(x,np))
      base_hash.update(x.tobytes(order='C'))
      tt=time.perf_counter();index.add(x);add_seconds+=time.perf_counter()-tt
     output.attrs['metric']=j['metric'];output.attrs['normalization_applied']=False;output.attrs['normalization_required']=j['normalized'];output.attrs['source_sha256']=source_sha;output.attrs['base_source_ids_sha256']=array_sha(baseids)
    cost['base_materialization_plus_gt_index_seconds']=time.perf_counter()-begun;cost['gt_index_add_seconds_subset']=add_seconds;cost['base_write_seconds_subset']=write_seconds
    if index.ntotal!=j['n_base']:raise ValueError('exact index base cardinality')
    begun=time.perf_counter();distances,neighbor_ids=index.search(query,10);cost['gt_exact_search_seconds']=time.perf_counter()-begun
    if neighbor_ids.shape!=(1000,10) or (neighbor_ids<0).any() or (neighbor_ids>=j['n_base']).any() or not np.isfinite(distances).all():raise ValueError('invalid exact GT output')
    if any(len(set(map(int,row)))!=10 for row in neighbor_ids):raise ValueError('duplicate GT IDs')
    for role,sl in [('calibration',slice(0,500)),('evaluation',slice(500,1000))]:
     with (out/role/'exact_top10.npz').open('xb') as ff:np.savez_compressed(ff,neighbor_ids=neighbor_ids[sl].astype(np.int64),distances=distances[sl].astype(np.float32),query_indices=np.arange(1000,dtype=np.int64)[sl])
    receipt.update(status='MATERIALIZED_EXACT_TOP10',n_base=j['n_base'],d=j['d'],query_count=1000,calibration_count=500,evaluation_count=500,raw_base_stats=raw_base_stats,effective_base_stats=effective_base_stats,raw_query_stats=raw_query_stats,effective_query_stats=stats(query,np),stored_base_representation='source FP32 before contract normalization; downstream applies frozen normalization once',stored_queries_representation='queries.npy raw FP32; effective_queries.npy already contract-normalized for GT/profile reuse',raw_base_array_sha256=raw_base_hash.hexdigest(),raw_query_array_sha256=array_sha(raw_query),query_preprocessing=qprep,base_array_sha256=base_hash.hexdigest(),base_source_ids_array_sha256=array_sha(baseids),query_source_ids_array_sha256=array_sha(ids),query_array_sha256=array_sha(query),calibration_query_array_sha256=array_sha(query[:500]),evaluation_query_array_sha256=array_sha(query[500:]),gt_ids_array_sha256=array_sha(neighbor_ids.astype(np.int64)),tie_policy='unchanged locked FAISS IndexFlat search ordering; no custom tie reorder',gt_source_sha256=m['gt_source_sha256'],faiss_version=faiss.__version__,numpy_version=np.__version__,thread_pools=threadpoolctl.threadpool_info(),query_batch_size_for_gt=1000,embedded_neighbors_used=False,base_row_order='ascending original source IDs after registered row selection and authorized holdout; SQL ID=base-position+1')
   receipt['files']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
  except Exception as e:receipt.update(status='ASSET_PREPARATION_FAILED',error=repr(e));raise
  finally:
   receipt.update(ended=time.time(),wall_seconds=time.time()-start,cost_components=cost,cost_note='base materialization includes GT index add; subset times must not be added twice',process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
   dump_new(out/'receipt.json',receipt)
  return receipt

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--manifest-sha256');p.add_argument('--asset-id',required=True);p.add_argument('--execute',action='store_true');a=p.parse_args();m=json.loads(a.manifest.read_text());jobs=[x for x in m['assets'] if x['asset_id']==a.asset_id]
 if len(jobs)!=1:raise ValueError('asset identity not unique')
 j=jobs[0];validate(m,j,a.execute)
 if not a.execute:print(json.dumps({'status':'DRY_RUN','asset':j,'physical_assets_computed':False}));return
 if sha(a.manifest)!=a.manifest_sha256:raise ValueError('manifest freeze SHA mismatch')
 print(json.dumps(prepare(m,j),indent=2))
if __name__=='__main__':main()
