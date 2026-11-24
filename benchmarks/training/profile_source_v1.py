"""Bounded geometry proxy, not asserted equivalent to installed SeekDB routing."""
import numpy as np,hashlib,time

def routes(x,c,metric,block=256):
 if metric not in ['ip','l2']:raise ValueError('unsupported metric')
 if x.ndim!=2 or c.ndim!=2 or x.shape[1]!=c.shape[1]:raise ValueError('shape')
 if not np.isfinite(x).all() or not np.isfinite(c).all():raise ValueError('nonfinite')
 result=[]
 for start in range(0,len(x),block):
  a=x[start:start+block].astype(np.float64);b=c.astype(np.float64)
  d=-(a@b.T) if metric=='ip' else (a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T
  result.append(np.argsort(d,axis=1,kind='stable'))
 return np.concatenate(result)

def profile(data,queries,gt_vectors,centroids,n,metric,grid,block=256):
 start=time.perf_counter();assert len(data)<=8192 and len(queries)<=128
 assert gt_vectors.shape[:2]==(len(queries),10)
 assignment=routes(data,centroids,metric,block)[:,0]
 counts=np.bincount(assignment,minlength=len(centroids)).astype(float);sizes=counts/len(data)*n
 query_routes=routes(queries,centroids,metric,block)
 gt_lists=routes(gt_vectors.reshape(-1,centroids.shape[1]),centroids,metric,block)[:,0].reshape(len(queries),10)
 curves=[]
 for p in grid:
  if p>len(centroids):continue
  chosen=query_routes[:,:p];covered=(chosen[:,:,None]==gt_lists[:,None,:]).any(1)
  work=sizes[chosen].sum(1)
  curves.append({'probe':p,'route_recall':float(covered.mean()),'scan_mean':float(work.mean()),'scan_p90':float(np.quantile(work,.9)),'scan_p99':float(np.quantile(work,.99))})
 selected=next((v for v in curves if v['route_recall']>=.95),None)
 return {'kind':'SAMPLED_GEOMETRY_PROXY_UNVERIFIED_ROUTING_EQUIVALENCE','centroid_payload_sha256':hashlib.sha256(centroids.tobytes()).hexdigest(),'n':n,'d':centroids.shape[1],'nlist':len(centroids),'metric':metric,'data_sample_count':len(data),'query_count':len(queries),'tie_policy':'ascending_centroid_id_proxy','list_cv':float(sizes.std()/sizes.mean()),'list_max_over_mean':float(sizes.max()/sizes.mean()),'sample_empty_fraction':float((counts==0).mean()),'curves':curves,'proxy_point':selected,'compute_seconds':time.perf_counter()-start}

def choose_m1(candidates,candidate_order=None):
 valid=[c for c in candidates if c['eligibility']['state']=='ELIGIBLE' and c.get('profile',{}).get('proxy_point')]
 if not valid:return {'status':'NEEDS_EVIDENCE','action':None}
 def cost(c):
  p=c['profile'];q=p['proxy_point'];return p['nlist']*p['d']+q['probe']+p['d']*q['scan_mean']
 if candidate_order is None:
  raise ValueError('provide the frozen action order explicitly')
 winner=min(valid,key=lambda c:(cost(c),candidate_order.index(c['candidate_id'])))
 return {'status':'PROXY_RECOMMENDATION_NOT_RECALL_GUARANTEE','action':winner['candidate_id'],'estimated_work':cost(winner)}
