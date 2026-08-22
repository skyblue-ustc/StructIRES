#!/usr/bin/env python3
"""Score completed design selections with an S3-held-out direct-RNA proxy.

The proxy is fit without any record whose provenance contains S3.  It is an
independent computational evaluator, never an optimization objective.
"""
from __future__ import annotations

import argparse, csv, hashlib, json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def paired_bootstrap(rows: list[dict[str, object]], left: str, right: str, seed: int = 20260822) -> dict[str, object]:
    """Compare methods at the parent-by-run level, not candidate level."""
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (str(row['run_seed']), str(row['parent_id']), str(row['method']))
        grouped.setdefault(key, []).append(float(row['direct_rna_s3_heldout_proxy']))
    units=[]
    for run_seed, parent_id in sorted({k[:2] for k in grouped}):
        a=grouped.get((run_seed,parent_id,left)); b=grouped.get((run_seed,parent_id,right))
        if a is None or b is None:
            continue
        units.append(float(np.mean(a)-np.mean(b)))
    values=np.asarray(units, dtype=float)
    if len(values) == 0:
        raise ValueError(f'No paired units for {left} vs {right}')
    rng=np.random.default_rng(seed)
    boot=np.mean(rng.choice(values, size=(10000,len(values)), replace=True),axis=1)
    return {'left_method':left,'right_method':right,'n_parent_run_units':int(len(values)),
            'mean_left_minus_right':float(values.mean()),
            'std_left_minus_right':float(values.std(ddof=1)),
            'bootstrap_ci95_low':float(np.quantile(boot,.025)),
            'bootstrap_ci95_high':float(np.quantile(boot,.975))}


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument('--data',type=Path,required=True); p.add_argument('--selected',type=Path,nargs='+',required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--seeds',default='42,43,44'); a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    data=[r for r in csv.DictReader(a.data.open()) if r['consensus_status'] in {'active','inactive'}]
    test=[r for r in data if 'S3' in r['source_tables'].split('|')]
    train=[r for r in data if 'S3' not in r['source_tables'].split('|')]
    y=np.asarray([int(r['consensus_label']) for r in train]); yt=np.asarray([int(r['consensus_label']) for r in test])
    if set(y)!={0,1} or set(yt)!={0,1}: raise ValueError('S3 split must retain both classes')
    vec=HashingVectorizer(analyzer='char',ngram_range=(3,6),n_features=65536,alternate_sign=False,lowercase=False,norm='l2',dtype=np.float32)
    x=vec.transform([r['sequence'].replace('U','T') for r in train]); xt=vec.transform([r['sequence'].replace('U','T') for r in test])
    models=[]; probs=[]
    for seed in [int(s) for s in a.seeds.split(',')]:
        m=SGDClassifier(loss='log_loss',penalty='elasticnet',alpha=1e-5,l1_ratio=.05,class_weight='balanced',max_iter=2000,tol=1e-4,random_state=seed)
        m.fit(x,y); models.append(m); probs.append(m.predict_proba(xt)[:,1])
    prob=np.mean(probs,axis=0)
    metrics={'n_train':len(train),'n_test_s3':len(test),'n_test_positive':int(yt.sum()),'auc':float(roc_auc_score(yt,prob)),'aupr':float(average_precision_score(yt,prob)),'n_models':len(models),'assay':'direct-RNA MPRA','split':'exclude every S3-provenance record from training'}
    selected=[]
    for path in a.selected:
        selected.extend(csv.DictReader(path.open()))
    # selected CSV intentionally stores hashes only; recover sequences from sibling pool JSONL.
    by_id={}
    for path in a.selected:
        pool=path.parent/'candidates.jsonl'
        with pool.open() as h:
            by_id.update({r['candidate_id']:r['sequence'] for r in map(json.loads,h)})
    seqs=[by_id[r['candidate_id']].replace('U','T') for r in selected]
    xp=vec.transform(seqs); sp=np.mean([m.predict_proba(xp)[:,1] for m in models],axis=0)
    rows=[]
    for r,s in zip(selected,sp): rows.append({'method':r['method'],'run_seed':r['run_seed'],'parent_id':r['parent_id'],'candidate_id':r['candidate_id'],'direct_rna_s3_heldout_proxy':float(s)})
    with (a.output_dir/'selected_proxy_scores.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=[]
    for method in sorted({r['method'] for r in rows}):
        values=[r['direct_rna_s3_heldout_proxy'] for r in rows if r['method']==method]
        summary.append({'method':method,'n':len(values),'mean_direct_rna_s3_heldout_proxy':float(np.mean(values)),'std_direct_rna_s3_heldout_proxy':float(np.std(values,ddof=1))})
    with (a.output_dir/'summary.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    available={str(row['method']) for row in rows}
    requested=[('robust_full','score_only'),('robust_full','random_mutation'),('structure_only','score_only'),
               ('robust_full','utrlm_score_only'),('utrlm_score_only','score_only'),
               ('robust_full','rnafm_fold0_score_only'),('rnafm_fold0_score_only','score_only'),
               ('rnafm_fold0_score_only','random_mutation'),
               ('structires','rnafm_score_only'),('structires','rnafm_plus_energy'),
               ('structires','rnafm_plus_ensemble')]
    comparisons=[paired_bootstrap(rows,left,right) for left,right in requested if {left,right} <= available]
    if comparisons:
        with (a.output_dir/'paired_method_comparisons.csv').open('w',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(comparisons[0]));w.writeheader();w.writerows(comparisons)
    (a.output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2,sort_keys=True)+'\n')
    (a.output_dir/'run_manifest.json').write_text(json.dumps({'schema_version':1,'data':str(a.data),'data_sha256':sha(a.data),'selected':[{'path':str(p),'sha256':sha(p)} for p in a.selected],'scope':'independent S3-held-out direct-RNA computational proxy; not experimental activity','metrics':metrics},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'metrics':metrics,'summary':summary,'comparisons':comparisons},indent=2))

if __name__=='__main__': main()
