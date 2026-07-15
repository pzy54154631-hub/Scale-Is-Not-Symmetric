#!/usr/bin/env python
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import FeatureUnion

NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
MODE_PREFIXES = {
    'base': ['base'],
    'large_pos': ['large_seed', 'large_pos'],
    'small_pos': ['small_seed', 'small_pos'],
    'balanced': ['balanced_seed', 'balanced'],
    'all_pairs': ['all_seed', 'all_pairs'],
    'real_only': ['real_seed', 'real_only'],
    'answer_tuned': ['answer_tuned'],
}

def parse_args():
    ap=argparse.ArgumentParser()
    ap.add_argument('--eval-csv', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--out-dir', default='results/scaleqa_natural/router_multiseed')
    ap.add_argument('--seeds', type=int, default=50)
    ap.add_argument('--test-size', type=float, default=0.2)
    ap.add_argument('--dev-size', type=float, default=0.2)
    ap.add_argument('--modes', default='base,large_pos,small_pos,balanced,all_pairs,real_only')
    return ap.parse_args()

def mode_of(adapter):
    a=str(adapter)
    for mode,prefixes in MODE_PREFIXES.items():
        for p in prefixes:
            if a == p or a.startswith(p):
                return mode
    return None

def parse_num(text):
    m=NUM_RE.search(str(text).replace(',', '').replace('−','-'))
    if not m: return None, ''
    raw=m.group(0)
    try: val=float(raw)
    except Exception: return None, raw
    return val, raw

def num_feats(text):
    val,raw=parse_num(text)
    if val is None:
        return {'has_num':0,'is_decimal':0,'lt_one':0,'dec_places':0,'log_bucket':0,'raw_len':0}
    absval=abs(val)
    dec=0
    if '.' in raw:
        dec=len(raw.split('.',1)[1].split('e',1)[0].split('E',1)[0])
    return {'has_num':1,'is_decimal':int('.' in raw),'lt_one':int(absval < 1.0),'dec_places':dec,'log_bucket':int(np.clip(np.floor(math.log10(absval+1e-12)),-8,8)),'raw_len':len(raw)}

def text_for(row):
    q='' if pd.isna(row.question) else str(row.question)
    bp='' if pd.isna(row.base_prediction) else str(row.base_prediction)
    f=num_feats(bp)
    return ' '.join([
        'question='+q,
        'base_prediction='+bp,
        'has_percent='+str(int('%' in q or 'percent' in q.lower() or 'percentage' in q.lower())),
        'base_decimal='+str(f['is_decimal']),
        'base_lt_one='+str(f['lt_one']),
        'base_dec_places='+str(f['dec_places']),
        'base_log_bucket='+str(f['log_bucket']),
        'base_raw_len='+str(f['raw_len']),
    ])

def aggregate(df, wanted):
    df=df.copy()
    df['mode']=df['adapter'].map(mode_of)
    df=df[df['mode'].isin(wanted)].copy()
    if df.empty: raise SystemExit('no matching modes')
    # metadata from base row when present, else first row
    meta_cols=['candidate_id','idx','question','answers','source_split','source_row_idx','scale_error_subtype','scale_factors','scale_directions']
    base=df[df['mode'].eq('base')].drop_duplicates('candidate_id')
    if len(base)==0:
        base=df.drop_duplicates('candidate_id')
    meta=base[meta_cols+['prediction']].rename(columns={'prediction':'base_prediction'})
    score=df.groupby(['candidate_id','mode'],as_index=False)['score'].mean()
    wide=score.pivot(index='candidate_id', columns='mode', values='score')
    for m in wanted:
        if m not in wide.columns:
            raise SystemExit(f'missing mode {m}')
    inst=meta.merge(wide.reset_index(), on='candidate_id', how='inner')
    return inst

def choose_scores(inst, ids, choices):
    rows=inst[inst.candidate_id.isin(ids)]
    vals=[]
    for r in rows.itertuples(index=False):
        vals.append(float(getattr(r, choices[r.candidate_id])))
    return float(np.mean(vals)) if vals else float('nan')

def fixed_scores(inst, ids, modes):
    rows=inst[inst.candidate_id.isin(ids)]
    return {m: float(rows[m].mean()) for m in modes}

def best_by_subtype(inst, train_ids, modes):
    tr=inst[inst.candidate_id.isin(train_ids)]
    out={}
    for st,g in tr.groupby('scale_error_subtype'):
        means={m: float(g[m].mean()) for m in modes}
        out[st]=max(means, key=means.get)
    global_means={m: float(tr[m].mean()) for m in modes}
    return out, max(global_means, key=global_means.get)

def route_by_subtype(inst, ids, subtype_best, global_best, pred_col='scale_error_subtype'):
    rows=inst[inst.candidate_id.isin(ids)]
    choices={r.candidate_id: subtype_best.get(getattr(r,pred_col), global_best) for r in rows.itertuples(index=False)}
    return choose_scores(inst, ids, choices), choices

def fit_subtype_router(inst, train_ids, dev_ids, test_ids, modes):
    train=inst[inst.candidate_id.isin(train_ids)].copy(); dev=inst[inst.candidate_id.isin(dev_ids)].copy(); test=inst[inst.candidate_id.isin(test_ids)].copy()
    for x in [train,dev,test]: x['router_text']=[text_for(r) for r in x.itertuples(index=False)]
    subtype_best, global_best=best_by_subtype(inst, train_ids, modes)
    best=None; best_dev=-1; best_c=None
    for c in [0.03,0.1,0.3,1,3,10,30]:
        vec=FeatureUnion([
            ('word', TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=1, max_features=8000)),
            ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=1, max_features=8000)),
        ])
        X=vec.fit_transform(train['router_text'])
        clf=LogisticRegression(C=c, max_iter=3000, class_weight='balanced', solver='liblinear', multi_class='ovr')
        clf.fit(X, train['scale_error_subtype'])
        pred=clf.predict(vec.transform(dev['router_text']))
        tmp=dev.copy(); tmp['pred_subtype']=pred
        choices={r.candidate_id: subtype_best.get(r.pred_subtype, global_best) for r in tmp.itertuples(index=False)}
        dev_score=choose_scores(inst, dev_ids, choices)
        if dev_score > best_dev:
            best_dev=dev_score; best=(vec,clf); best_c=c
    vec,clf=best
    pred=clf.predict(vec.transform(test['router_text']))
    tmp=test.copy(); tmp['pred_subtype']=pred
    choices={r.candidate_id: subtype_best.get(r.pred_subtype, global_best) for r in tmp.itertuples(index=False)}
    route_score=choose_scores(inst, test_ids, choices)
    subtype_acc=float(accuracy_score(test['scale_error_subtype'], pred))
    choice_df=tmp[['candidate_id','scale_error_subtype']].copy()
    choice_df['pred_subtype']=pred
    choice_df['chosen_mode']=[choices[cid] for cid in choice_df['candidate_id']]
    return route_score, subtype_acc, best_dev, best_c, choice_df, subtype_best, global_best

def main():
    args=parse_args(); out=Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    modes=[x.strip() for x in args.modes.split(',') if x.strip()]
    df=pd.read_csv(args.eval_csv)
    inst=aggregate(df, modes)
    strata=inst['scale_error_subtype'].fillna('unknown').astype(str).values
    ids=inst['candidate_id'].values
    rows=[]; choices=[]; maps=[]
    for seed in range(args.seeds):
        s1=StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=seed)
        train_dev_idx,test_idx=next(s1.split(ids,strata))
        train_dev_ids=ids[train_dev_idx]; test_ids=set(ids[test_idx]); train_dev_strata=strata[train_dev_idx]
        s2=StratifiedShuffleSplit(n_splits=1, test_size=args.dev_size/(1-args.test_size), random_state=seed+1000)
        tr_rel,dev_rel=next(s2.split(train_dev_ids, train_dev_strata))
        train_ids=set(train_dev_ids[tr_rel]); dev_ids=set(train_dev_ids[dev_rel])
        fixed=fixed_scores(inst, test_ids, modes)
        train_fixed=fixed_scores(inst, train_ids, modes)
        global_best=max(train_fixed, key=train_fixed.get)
        global_score=choose_scores(inst, test_ids, {cid:global_best for cid in test_ids})
        subtype_best, subtype_global=best_by_subtype(inst, train_ids, modes)
        subtype_score, _=route_by_subtype(inst, test_ids, subtype_best, subtype_global)
        pred_score, subtype_pred_acc, dev_score, c, choice_df, subtype_map, subtype_global=fit_subtype_router(inst, train_ids, dev_ids, test_ids, modes)
        oracle=float(inst[inst.candidate_id.isin(test_ids)][modes].max(axis=1).mean())
        row={'seed':seed,'n_train':len(train_ids),'n_dev':len(dev_ids),'n_test':len(test_ids),'global_best_mode':global_best,'global_best':global_score,'oracle_subtype_router':subtype_score,'predicted_subtype_router':pred_score,'predicted_subtype_acc':subtype_pred_acc,'predicted_subtype_dev':dev_score,'predicted_C':c,'oracle_instance':oracle}
        row.update({f'fixed_{k}':v for k,v in fixed.items()})
        rows.append(row)
        choice_df['seed']=seed; choices.append(choice_df)
        maps.append({'seed':seed,'global':subtype_global,**subtype_map})
    res=pd.DataFrame(rows)
    res.to_csv(out/f'{args.name}_multiseed_router_results.csv', index=False)
    pd.concat(choices,ignore_index=True).to_csv(out/f'{args.name}_multiseed_router_choices.csv', index=False)
    pd.DataFrame(maps).to_csv(out/f'{args.name}_multiseed_router_maps.csv', index=False)
    metric_cols=[c for c in res.columns if c.startswith('fixed_') or c in ['global_best','oracle_subtype_router','predicted_subtype_router','predicted_subtype_acc','oracle_instance']]
    summary={'name':args.name,'n_instances':int(inst.candidate_id.nunique()),'modes':modes,'subtype_counts':inst.scale_error_subtype.value_counts().to_dict(),'source_split_counts':inst.source_split.value_counts().to_dict(),'metrics':{}}
    for c in metric_cols:
        summary['metrics'][c]={'mean':float(res[c].mean()),'std':float(res[c].std(ddof=1)),'min':float(res[c].min()),'max':float(res[c].max())}
    (out/f'{args.name}_multiseed_router_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=[f'# Multi-seed regime router: {args.name}','',f'N={summary["n_instances"]}','','| Metric | Mean | Std | Min | Max |','|---|---:|---:|---:|---:|']
    for c,m in summary['metrics'].items(): lines.append(f'| {c} | {m["mean"]:.4f} | {m["std"]:.4f} | {m["min"]:.4f} | {m["max"]:.4f} |')
    (out/f'{args.name}_multiseed_router_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary, indent=2))
if __name__=='__main__': main()
