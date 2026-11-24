"""FAISS IVF_FLAT with externally frozen centroids; never calls train().

Calibration scans the predeclared grid. Deployment probes for three recall targets
are fixed before opening final evaluation queries. Two independent index builds,
two warmup passes, 3 calibration and 10 evaluation timing passes. Full calibration
curves and deployment evaluation points are distinct outputs.
"""
from pathlib import Path
import os,sys,time,json,hashlib,csv,argparse,traceback,resource,subprocess,signal
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
def run(m, task, candidate, build, root, cpu_ids):
 os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',CUDA_VISIBLE_DEVICES='')
 assert os.sched_getaffinity(0) == set(cpu_ids)
 import numpy as np,faiss,threadpoolctl
 faiss.omp_set_num_threads(4)
 t=next(x for x in m['tasks'] if x['task']==task)
 asset=root/'results/assets'/task;training=root/'results/training'/task/candidate
 tr=json.loads((training/'result.json').read_text());ar=json.loads((asset/'result.json').read_text())
 assert tr['status']=='OK' and ar['status']=='ASSETS_READY'
 out=root/'results/faiss'/task/candidate/('build'+str(build));out.mkdir(parents=True,exist_ok=False)
 rec={'task':task,'candidate_id':tr['candidate_id'],'algorithm':candidate,'build':build,'started':time.time(),'status':'STARTED','n':t['n'],'d':t['d'],'nlist':m['nlist'],'metric':t['metric'],'training_result_sha256':sha(training/'result.json'),'source_centroid_sha256':tr['centroid_sha256'],'effective_centroid_sha256':tr['effective_centroid_sha256'],'faiss_version':faiss.__version__,'training_called':False,'cpu_ids':sorted(os.sched_getaffinity(0)),'numa_node':1,'concurrency':1,'environment_before':{'loadavg':list(os.getloadavg()),'meminfo':Path('/proc/meminfo').read_text(),'cpu_stat':Path('/proc/stat').read_text()},'manifest_sha256':m['_sha']}
 try:
  for p,h in [(asset/'effective_base.npy',ar['files']['effective_base.npy']),(training/'effective_centroids.npy',tr['effective_centroid_sha256'])]:assert sha(p)==h
  x=np.load(asset/'effective_base.npy');centers=np.load(training/'effective_centroids.npy');assert x.shape==(t['n'],t['d'])
  if t['metric']=='ip':
   assert np.max(np.abs(np.linalg.norm(x,axis=1)-1))<1e-5
   assert np.max(np.abs(np.linalg.norm(centers,axis=1)-1))<1e-5
  with threadpoolctl.threadpool_limits(4):
   metric=faiss.METRIC_INNER_PRODUCT if t['metric']=='ip' else faiss.METRIC_L2
   s=time.perf_counter();quantizer=faiss.IndexFlatIP(t['d']) if t['metric']=='ip' else faiss.IndexFlatL2(t['d']);quantizer.add(centers)
   index=faiss.IndexIVFFlat(quantizer,t['d'],m['nlist'],metric);index.is_trained=True
   assert np.array_equal(quantizer.reconstruct_n(0,m['nlist']),centers),'CENTROID_READBACK_MISMATCH'
   index.add(x);rec['index_build_seconds']=time.perf_counter()-s
   assert index.ntotal==len(x)
   sizes=np.array([index.invlists.list_size(i) for i in range(m['nlist'])],dtype=np.int64)
   assert np.array_equal(sizes,np.load(training/'list_sizes.npy')),'ASSIGNMENT_MISMATCH'
   # Exact list-membership agreement, not only histogram agreement.
   expected_assign=np.load(training/'assignments.npy')
   for li in range(m['nlist']):
    n=int(sizes[li])
    if n:
     ptr=index.invlists.get_ids(li)
     try:ids=faiss.rev_swig_ptr(ptr,n).copy();assert np.all(expected_assign[ids]==li)
     finally:index.invlists.release_ids(li,ptr)
   rec['centroid_and_assignment_checks']='EXACT_MATCH'
   faiss.write_index(index,str(out/'index.faiss'));rec['index_sha256']=sha(out/'index.faiss')
   raw=(out/'requests.csv').open('x',newline='');writer=csv.writer(raw);writer.writerow(['split','probe','phase','pass','query_id','search_nanoseconds','top10_hits','actual_distance_computations'])
   passes=[]
   def measure(split,probe,q,gt,offset,timed):
    index.nprobe=probe;rows=[];latencies=[]
    for iteration in range(2+timed):
     phase='warmup' if iteration<2 else 'timed'
     ids_all=np.empty((len(q),10),np.int64);dists_all=np.empty((len(q),10),np.float32);times=np.empty(len(q),np.int64);scans=np.empty(len(q),np.int64)
     faiss.cvar.indexIVF_stats.reset();s=time.perf_counter_ns()
     for qi in range(len(q)):
      before=faiss.cvar.indexIVF_stats.ndis
      begin=time.perf_counter_ns();D,I=index.search(q[qi:qi+1],10);times[qi]=time.perf_counter_ns()-begin
      ids_all[qi]=I[0];dists_all[qi]=D[0];scans[qi]=faiss.cvar.indexIVF_stats.ndis-before
     wall=(time.perf_counter_ns()-s)/1e9
     assert ((ids_all>=-1)&(ids_all<len(x))).all(),'INVALID_SEARCH_IDS'
     assert np.isfinite(dists_all[ids_all>=0]).all(),'INVALID_NUMERIC_SEARCH_RESULT'
     hits=[len(set(row.tolist())&set(gt[i].tolist())) for i,row in enumerate(ids_all)]
     for qi in range(len(q)):writer.writerow([split,probe,phase,iteration-2 if iteration>=2 else iteration,qi+offset,int(times[qi]),hits[qi],int(scans[qi])])
     row={'split':split,'probe':probe,'phase':phase,'pass':iteration-2 if iteration>=2 else iteration,'recall_at10':sum(hits)/(10*len(q)),'search_qps':float(len(q)/(times.sum()/1e9)),'python_client_qps':len(q)/wall,'client_seconds':wall,'mean_actual_scanned_distances':float(scans.mean())}
     passes.append(row)
     if phase=='timed':rows.append(row);latencies.extend(times.tolist())
    raw.flush();os.fsync(raw.fileno())
    # Each request contributes directly. These are not quantiles of query medians.
    a=np.asarray(latencies)/1e6
    return {'split':split,'probe':probe,'recall_at10':float(np.median([r['recall_at10'] for r in rows])),'minimum_pass_recall':min(r['recall_at10'] for r in rows),'search_qps_median':float(np.median([r['search_qps'] for r in rows])),'python_client_qps_median':float(np.median([r['python_client_qps'] for r in rows])),'request_p50_ms':float(np.quantile(a,.50)),'request_p95_ms':float(np.quantile(a,.95)),'request_p99_ms':float(np.quantile(a,.99)),'request_count':len(a),'mean_actual_scanned_distances':float(np.mean([r['mean_actual_scanned_distances'] for r in rows]))}
   for name in ['calibration_queries.npy','calibration_gt.npy']:assert sha(asset/name)==ar['files'][name]
   cal=np.load(asset/'calibration_queries.npy');calgt=np.load(asset/'calibration_gt.npy');curves=[]
   for probe in m['planned_downstream']['probe_grid']:curves.append(measure('calibration',probe,cal,calgt,0,3))
   selected=[]
   for target in m['planned_downstream']['recall_targets']:
    feasible=[r for r in curves if r['minimum_pass_recall']>=target]
    selected.append({'target_recall':target,'probe':feasible[0]['probe'] if feasible else None,'status':'CALIBRATION_QUALIFIED' if feasible else 'RECALL_NOT_REACHED'})
   write(out/'calibration.json',curves);write(out/'deployment_probes.frozen.json',selected)
   # Final evaluation files are opened only after the above choices are saved.
   for name in ['evaluation_queries.npy','evaluation_gt.npy']:assert sha(asset/name)==ar['files'][name]
   eq=np.load(asset/'evaluation_queries.npy');egt=np.load(asset/'evaluation_gt.npy');evals=[]
   for probe in sorted({v['probe'] for v in selected if v['probe'] is not None}):evals.append(measure('evaluation',probe,eq,egt,500,10))
   for choice in selected:
    row=next((v for v in evals if v['probe']==choice['probe']),None)
    choice['evaluation_status']='NOT_MEASURED_NO_CALIBRATION_PROBE' if row is None else ('RECALL_QUALIFIED' if row['minimum_pass_recall']>=choice['target_recall'] else 'RECALL_NOT_REACHED')
    choice['evaluation']=row
   raw.close();write(out/'passes.json',passes);write(out/'evaluation.json',evals);write(out/'deployments.json',selected)
   rec.update(status='MEASURED',calibration_curve_count=len(curves),evaluation_probe_count=len(evals),deployments=selected,latency_boundary='FAISS Python index.search entry through returned arrays, one real request per row; no SQL',client_boundary='serial Python request loop including stats reads and result copies; excludes recall scoring and persistence; not a SQL client QPS',curve_scope='full predeclared calibration grid; final evaluation only at calibration-selected deployment probes')
 except Exception as e:rec.update(status='EVALUATION_FAILED',error=repr(e),traceback=traceback.format_exc())
 finally:
  rec.update(ended=time.time(),wall_seconds=time.time()-rec['started'],peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,environment_after={'loadavg':list(os.getloadavg()),'cpu_stat':Path('/proc/stat').read_text()})
  write(out/'result.json',rec)
 return rec['status']=='MEASURED'
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--cpu-ids',nargs=4,type=int,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--sha256',required=True);p.add_argument('--task',required=True);p.add_argument('--candidate',required=True);p.add_argument('--build',type=int,choices=[1,2],required=True);a=p.parse_args()
 assert sha(a.manifest)==a.sha256;m=json.loads(a.manifest.read_text());m['_sha']=a.sha256
 assert sha(__file__)==m['evaluation_worker_sha256'],'WORKER_CODE_DRIFT'
 sys.exit(0 if run(m,a.task,a.candidate,a.build,a.root,a.cpu_ids) else 2)
