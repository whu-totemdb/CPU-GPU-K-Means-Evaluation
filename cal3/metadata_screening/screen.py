"""Metadata-only screening, independent of V21/SW1. No database or training runner.

The optional fit entry point fits development labels only after an explicit config
freeze. It never accepts prospective tasks. Scoring uses a strict metadata DTO;
family/workload IDs are only split keys and cannot enter the design matrix.
"""
from pathlib import Path
import argparse,csv,hashlib,json,math

WORKLOAD_KEYS=frozenset(('n','d','nlist','metric','normalization'))

def flag(value):
    if value is True or str(value).lower()=='true': return True
    if value is False or str(value).lower()=='false': return False
    raise ValueError('missing or invalid boolean: '+str(value))

def public_metadata(row):
    """Explicit binding boundary: outcomes never flow into encode_metadata."""
    return {key:row[key] for key in WORKLOAD_KEYS}

def numeric_label(row):
    if row.get('_supervision_eligible') is False: return None
    try: q=float(row['qps_at_r95'])
    except (ValueError,TypeError,KeyError): return None
    return q if math.isfinite(q) and q>0 and row.get('terminal_status')=='OK' and flag(row.get('utility_supervised','false')) else None

def apply_supervision_admission(rows,config):
    """Copy-only label view: keep original QPS and terminal evidence immutable."""
    excluded=config.get('supervision_excluded_candidates',{})
    return [dict(row,_supervision_eligible=row['candidate_id'] not in excluded,
                 _supervision_exclusion=excluded.get(row['candidate_id'])) for row in rows]

def oracle_inclusion_bounds(rows,shortlist,registry_ids,config):
    """Partial identification of genuine full-registry oracle, no imputed QPS."""
    positive=[row for row in rows if numeric_label(row) is not None]
    if not positive: raise ValueError('no observed admissible utility anchor')
    oracle=max(numeric_label(row) for row in positive)
    winners={row['candidate_id'] for row in positive if numeric_label(row)==oracle}
    known_negative=set(config.get('known_nonutility_terminal_statuses',[]))
    indexed={row['candidate_id']:row for row in rows};unknown=set()
    for cid in registry_ids:
        row=indexed.get(cid)
        if row is None: unknown.add(cid)
        elif row.get('terminal_status') in known_negative: continue
        elif numeric_label(row) is None: unknown.add(cid)
    included=set(shortlist);observed_included=bool(winners & included)
    return dict(lower=float(observed_included and unknown <= included),
                upper=float(observed_included or bool(unknown & included)),
                observed_subset_inclusion=float(observed_included),
                observed_winner_ids=sorted(winners),unknown_candidate_ids=sorted(unknown))

def validate_registry(registry):
    ids=[c['candidate_id'] for c in registry]
    if len(ids)!=15 or len(set(ids))!=15: raise ValueError('expected15 unique registry candidates')
    return ids

def descriptor(candidate,registry):
    """Only static registry fields; no measured candidate features."""
    result={}
    for field in ('candidate_id','raw_algorithm','device','compute_dtype','approximate'):
        vocabulary=sorted({str(c.get(field,'UNKNOWN')) for c in registry})
        for value in vocabulary: result[field+'='+value]=float(str(candidate.get(field,'UNKNOWN'))==value)
    for field in ('sampling_fraction','threads'):
        value=candidate.get('options_match',{}).get(field)
        missing=value is None
        if not missing and not math.isfinite(float(value)): raise ValueError('nonfinite static metadata')
        result[field]=0.0 if missing else float(value)
        result[field+'_missing']=float(missing)
    return result

def encode_metadata(metadata,candidate,registry):
    """Reject additional keys rather than allowing target-outcome injection."""
    if set(metadata)!=WORKLOAD_KEYS: raise ValueError('metadata keys must match strict whitelist')
    base={}
    for key in ('n','d','nlist'):
        value=float(metadata[key])
        if not math.isfinite(value) or value<=0: raise ValueError('invalid '+key)
        base['log_'+key]=math.log(value)
    metric=str(metadata['metric']).lower()
    if metric not in ('l2','ip'): raise ValueError('unsupported metric')
    base.update(metric_l2=float(metric=='l2'),metric_ip=float(metric=='ip'),normalization=float(flag(metadata['normalization'])))
    static=descriptor(candidate,registry);features=dict(base)
    features.update({'candidate:'+k:v for k,v in static.items()})
    for wk,wv in base.items():
        for ck,cv in static.items(): features[wk+'*'+ck]=wv*cv
    return features

def bind_workload_metadata(rows,matrix_rows):
    """Fill absent static fields from frozen workload identity; reject conflicts.

    This is a copy-only metadata join, not label imputation: QPS, terminal status,
    candidate ID and all scientific outcomes are left byte-for-byte unchanged.
    """
    mapping={row['workload_id']:row for row in matrix_rows}
    if len(mapping)!=len(matrix_rows): raise ValueError('duplicate workload matrix key')
    fields={'family_id':'family_id','n':'n','d':'d','nlist':'nlist','metric':'metric','normalization':'normalized'}
    bound=[];repairs=[]
    for row in rows:
        task=row['workload_id']
        if task not in mapping: raise ValueError('workload absent from frozen matrix: '+task)
        result=dict(row)
        for field,source_field in fields.items():
            value=mapping[task].get(source_field)
            if value is None or str(value).strip()=='': raise ValueError('missing matrix '+source_field+' for '+task)
            old=row.get(field)
            if old is None or str(old).strip()=='':
                result[field]=value
                repairs.append(dict(workload_id=task,candidate_id=row['candidate_id'],field=field,original_value=old,bound_value=value,source_key=task,source_field=source_field))
            elif old!=value:
                raise ValueError('metadata conflict '+task+' '+field+': '+str(old)+' != '+str(value))
        bound.append(result)
    return bound,repairs

def task_groups(rows):
    result={}
    for row in rows: result.setdefault(row['workload_id'],[]).append(row)
    for task,group in result.items():
        if len({row['family_id'] for row in group})!=1: raise ValueError('conflicting family: '+task)
        if len({json.dumps(public_metadata(row),sort_keys=True) for row in group})!=1: raise ValueError('conflicting metadata: '+task)
        if len({r['candidate_id'] for r in group})!=len(group): raise ValueError('duplicate task candidate')
    return result

def targets_and_weights(rows):
    """Every oracle/reference is within its own training task, never another fold."""
    groups=task_groups(rows); numeric={t:[r for r in rs if numeric_label(r) is not None] for t,rs in groups.items()}
    numeric={t:rs for t,rs in numeric.items() if rs}
    families={}
    for task,rs in numeric.items(): families.setdefault(rs[0]['family_id'],[]).append(task)
    output=[]
    for family,tasks in sorted(families.items()):
        for task in sorted(tasks):
            rs=numeric[task];oracle=max(numeric_label(r) for r in rs)
            for row in rs:
                output.append((row,math.log(numeric_label(row)/oracle),1/(len(families)*len(tasks)*len(rs))))
    return output

def baseline_ranking(rows,registry):
    """Development-only complete ranking; absent labels remain absent, never zero."""
    groups={t:rs for t,rs in task_groups(rows).items() if any(numeric_label(r) is not None for r in rs)}
    result=[]
    for order,candidate in enumerate(registry):
        by_family={};coverage=0
        for task,rs in groups.items():
            selected=next((r for r in rs if r['candidate_id']==candidate['candidate_id']),None)
            value=numeric_label(selected) if selected else None
            if value is None: continue
            coverage+=1;oracle=max(numeric_label(r) for r in rs if numeric_label(r) is not None)
            by_family.setdefault(rs[0]['family_id'],[]).append(value/oracle)
        utility=sum(sum(v)/len(v) for v in by_family.values())/len(by_family) if by_family else None
        result.append(dict(candidate_id=candidate['candidate_id'],covered_tasks=coverage,total_tasks=len(groups),family_macro_retained_utility=utility,registry_order=order))
    return sorted(result,key=lambda r:(-r['covered_tasks'],-(r['family_macro_retained_utility'] if r['family_macro_retained_utility'] is not None else -1),r['registry_order']))

def protected_top_m(scores,eligible,baseline_order,M,registry_order):
    """M includes the first eligible strong baseline. No QPS is read here."""
    if M<1: raise ValueError('M must be positive')
    allowed=set(eligible)
    baseline=next((c for c in baseline_order if c in allowed),None)
    if baseline is None: return dict(status='NO_ELIGIBLE_BASELINE',baseline=None,candidates=[])
    finite=[c for c in allowed if c!=baseline and c in scores and math.isfinite(scores[c])]
    finite.sort(key=lambda c:(-scores[c],registry_order.index(c)))
    return dict(status='SCREENED',baseline=baseline,candidates=[baseline]+finite[:M-1])

def choose_M(inclusion,threshold=.90):
    for M in (3,5,7,10):
        if inclusion[str(M)]>=threshold: return M
    return 10

def fit_ridge(rows,registry,alpha):
    import numpy as np
    examples=targets_and_weights(rows)
    if not examples: raise ValueError('empty training part')
    candidates={c['candidate_id']:c for c in registry}
    encoded=[encode_metadata(public_metadata(r),candidates[r['candidate_id']],registry) for r,_,_ in examples]
    names=sorted(encoded[0]);X=np.asarray([[d[k] for k in names] for d in encoded]);y=np.asarray([v for _,v,_ in examples]);w=np.asarray([v for _,_,v in examples])
    mean=(X*w[:,None]).sum(0);std=np.sqrt(((X-mean)**2*w[:,None]).sum(0));std[std==0]=1
    Z=(X-mean)/std;intercept=float((y*w).sum());rhs=Z.T@(w*(y-intercept));coef=np.linalg.solve(Z.T@(w[:,None]*Z)+alpha*np.eye(Z.shape[1]),rhs)
    return dict(alpha=alpha,feature_names=names,mean=mean.tolist(),std=std.tolist(),coef=coef.tolist(),intercept=intercept,training_tasks=sorted({r['workload_id'] for r,_,_ in examples}),training_families=sorted({r['family_id'] for r,_,_ in examples}),positive_training_rows=len(examples))

def predict(model,metadata,registry):
    result={}
    for c in registry:
        f=encode_metadata(metadata,c,registry)
        if set(f)!=set(model['feature_names']): raise ValueError('schema mismatch')
        result[c['candidate_id']]=model['intercept']+sum(co*(f[k]-mu)/sd for k,mu,sd,co in zip(model['feature_names'],model['mean'],model['std'],model['coef']))
    return result

def validation_mse(model,rows,registry):
    examples=targets_and_weights(rows);cache={}
    loss=0
    for row,y,w in examples:
        task=row['workload_id']
        if task not in cache: cache[task]=predict(model,public_metadata(row),registry)
        loss+=w*(cache[task][row['candidate_id']]-y)**2
    return loss

def select_alpha(rows,registry,grid):
    families=sorted({r['family_id'] for r,_,_ in targets_and_weights(rows)});scores=[]
    if len(families)<2: raise ValueError('inner CV needs at least2 families')
    for alpha in grid:
        folds=[]
        for heldout in families:
            train=[r for r in rows if r['family_id']!=heldout];test=[r for r in rows if r['family_id']==heldout]
            model=fit_ridge(train,registry,alpha)
            folds.append(dict(heldout_family=heldout,mse=validation_mse(model,test,registry),training_tasks=model['training_tasks'],training_families=model['training_families']))
        scores.append(dict(alpha=alpha,family_macro_mse=sum(x['mse'] for x in folds)/len(folds),folds=folds))
    winner=min(scores,key=lambda r:(r['family_macro_mse'],-r['alpha']))
    return winner['alpha'],scores

def develop(rows,registry,config):
    """Development-only nested LOFO. Does not create prospective actions."""
    validate_registry(registry);rows=apply_supervision_admission(rows,config);all_groups=task_groups(rows)
    groups={t:rs for t,rs in all_groups.items() if any(numeric_label(r) is not None for r in rs)}
    numeric_rows=[r for rs in groups.values() for r in rs];families=sorted({rs[0]['family_id'] for rs in groups.values()})
    if len(groups)!=config['expected_numeric_workloads'] or len(families)!=config['expected_numeric_families']: raise ValueError('development denominator differs from frozen expectation')
    ids=[c['candidate_id'] for c in registry];folds=[];predictions=[]
    for heldout in families:
        train=[r for r in numeric_rows if r['family_id']!=heldout];test=[r for r in numeric_rows if r['family_id']==heldout]
        alpha,inner=select_alpha(train,registry,config['alpha_grid']);model=fit_ridge(train,registry,alpha);ranking=baseline_ranking(train,registry)
        folds.append(dict(heldout_family=heldout,model=model,inner_validation=inner,strong_fixed_ranking=ranking))
        for task,rs in task_groups(test).items():
            scores=predict(model,public_metadata(rs[0]),registry)
            # Freeze all top-M predictions before inspecting heldout label values.
            screens={str(M):protected_top_m(scores,ids,[r['candidate_id'] for r in ranking],M,ids) for M in config['M_grid']}
            bounds={str(M):oracle_inclusion_bounds(rs,screens[str(M)]['candidates'],ids,config) for M in config['M_grid']}
            predictions.append(dict(workload_id=task,family_id=heldout,metadata=public_metadata(rs[0]),scores=scores,screened_candidates=screens,oracle_inclusion_bounds_development_only=bounds))
    summaries={kind:{str(M):sum(sum(r['oracle_inclusion_bounds_development_only'][str(M)][kind] for r in predictions if r['family_id']==f)/sum(r['family_id']==f for r in predictions) for f in families)/len(families) for M in config['M_grid']} for kind in ('lower','upper','observed_subset_inclusion')}
    inclusion=summaries['lower'];M=choose_M(inclusion)
    M_reason='PROVABLE_FULL_REGISTRY_LOWER_BOUND_GE_0.90' if inclusion[str(M)]>=.90 else 'PREDECLARED_FALLBACK10_FULL_ORACLE_NOT_IDENTIFIED'
    alpha,final_cv=select_alpha(numeric_rows,registry,config['alpha_grid']);final=fit_ridge(numeric_rows,registry,alpha)
    return dict(version=config['version'],role='DEVELOPMENT_SCREEN_NOT_INDEPENDENT_FINAL_TEST',numeric_workloads=len(groups),numeric_families=len(families),positive_labels=sum(numeric_label(r) is not None for r in numeric_rows),nonpositive_or_missing_label_records=[{'workload_id':r['workload_id'],'candidate_id':r['candidate_id'],'terminal_status':r['terminal_status'],'supervision_exclusion':r.get('_supervision_exclusion')} for r in rows if numeric_label(r) is None],folds=folds,heldout_predictions=predictions,full_registry_oracle_inclusion_family_macro_lower=summaries['lower'],full_registry_oracle_inclusion_family_macro_upper=summaries['upper'],observed_subset_inclusion_diagnostic_family_macro=summaries['observed_subset_inclusion'],frozen_M=M,M_selection_reason=M_reason,final_model=final,final_alpha_selection=final_cv,strong_fixed_ranking=baseline_ranking(numeric_rows,registry),prospective_actions=[])

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fit-development',action='store_true');parser.add_argument('--freeze',type=Path);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    here=Path(__file__).resolve().parent;root=here;cfg_path=here/'config.json';config=json.loads(cfg_path.read_text())
    if not args.fit_development or not args.freeze: parser.error('fitting requires explicit --fit-development and --freeze receipt')
    freeze=json.loads(args.freeze.read_text())
    if freeze.get('config_sha256')!=hashlib.sha256(cfg_path.read_bytes()).hexdigest() or freeze.get('screen_source_sha256')!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(): raise ValueError('config/source freeze mismatch')
    if freeze.get('development_fit_authorized') is not True: raise ValueError('fit not authorized by frozen plan receipt')
    labels=root/config['source_labels'];registry_path=root/config['source_registry'];matrix_path=root/config['source_workload_matrix']
    for name,path in [('labels_sha256',labels),('registry_sha256',registry_path),('workload_matrix_sha256',matrix_path)]:
        if freeze.get(name)!=hashlib.sha256(path.read_bytes()).hexdigest(): raise ValueError(name+' freeze mismatch')
    if args.output.exists(): raise FileExistsError('preserve prior outputs')
    rows=list(csv.DictReader(labels.open()));matrix_rows=list(csv.DictReader(matrix_path.open()));rows,repairs=bind_workload_metadata(rows,matrix_rows);registry=json.loads(registry_path.read_text())['candidates'];result=develop(rows,registry,config);result['metadata_binding_repairs']=repairs;result['metadata_binding_source']=config['source_workload_matrix']
    args.output.mkdir(parents=True);(args.output/'development_screen_results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(args.output/'input_freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')

if __name__=='__main__': main()
