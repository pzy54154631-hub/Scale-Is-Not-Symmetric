#!/usr/bin/env python
"""Interpretable rule-table policy router for ScaleQA-Natural."""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit

NUM_RE = re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", re.I)

def norm(x):
    return "" if x is None or (isinstance(x,float) and pd.isna(x)) else str(x).lower()

def nums(x):
    out=[]
    for m in NUM_RE.findall(str(x).replace(',','').replace('−','-')):
        try: out.append(float(m))
        except Exception: pass
    return out

def qtype(q):
    s=norm(q)
    if any(w in s for w in ['average','mean']): return 'average'
    if any(w in s for w in ['difference','more than','less than']): return 'difference'
    if any(w in s for w in ['ratio','times','a:b']): return 'ratio'
    if any(w in s for w in ['percent','percentage','%']): return 'percent'
    if any(w in s for w in ['highest','largest','maximum','greatest']): return 'max'
    if any(w in s for w in ['lowest','smallest','minimum','least']): return 'min'
    if any(w in s for w in ['sum','total']): return 'sum'
    return 'lookup'

def unit(ctx,q):
    s=norm(ctx)+' '+norm(q)
    if '%' in s or 'percent' in s or 'percentage' in s: return 'percent'
    if any(w in s for w in ['million','billion','thousand']): return 'large_unit'
    if any(w in s for w in ['ratio','rate','per ']): return 'ratio_rate'
    if any(w in s for w in ['$', 'usd','dollar','euro']): return 'currency'
    return 'none'

def bucket(x):
    xs=nums(x)
    if not xs: return 'no_num'
    v=abs(xs[0])
    if v<0.01: return '<0.01'
    if v<0.1: return '0.01-0.1'
    if v<1: return '0.1-1'
    if v<10: return '1-10'
    if v<100: return '10-100'
    if v<1000: return '100-1000'
    return '>=1000'

def read(spec):
    if '::' in spec: prefix,path=spec.split('::',1)
    else: prefix,path='',spec
    df=pd.read_csv(path)
    df=df[~df['adapter'].astype(str).str.contains('oracle',case=False,na=False)].copy()
    df['policy']=(prefix+':'+df['adapter'].astype(str)) if prefix else df['adapter'].astype(str)
    for c in ['candidate_id','question','answers','prediction','score','scale_error_subtype','source_split','source_row_idx']:
        if c not in df.columns: df[c]=''
    return df[['candidate_id','policy','question','answers','prediction','score','scale_error_subtype','source_split','source_row_idx']]

def build(files,ctx_csv,manifest):
    df=pd.concat([read(x) for x in files],ignore_index=True).drop_duplicates(['candidate_id','policy'],keep='last')
    ctx=pd.read_csv(ctx_csv)
    df=df.merge(ctx[['candidate_id','chart_context']],on='candidate_id',how='left')
    if manifest:
        man=pd.read_csv(manifest)
        keep=[c for c in ['candidate_id','source_split','source_row_idx','scale_error_subtype'] if c in man.columns]
        df=df.merge(man[keep],on='candidate_id',how='left',suffixes=('','_m'))
        for c in ['source_split','source_row_idx','scale_error_subtype']:
            cm=c+'_m'
            if cm in df.columns: df[c]=df[c].where(df[c].astype(str).str.len()>0,df[cm])
    df['chart_group']=df['source_split'].fillna('').astype(str)+'::'+df['source_row_idx'].fillna('').astype(str)
    df.loc[df['chart_group'].isin(['::','nan::nan']),'chart_group']=df['candidate_id']
    inst=df.sort_values('policy').drop_duplicates('candidate_id').copy()
    pred_map=df.pivot_table(index='candidate_id',columns='policy',values='prediction',aggfunc='first')
    inst=inst.merge(pred_map.reset_index(),on='candidate_id',how='left',suffixes=('','_p'))
    ans_col='main:answer_tuned' if 'main:answer_tuned' in inst.columns else ('tta:direct' if 'tta:direct' in inst.columns else 'prediction')
    inst['qtype']=[qtype(x) for x in inst['question']]
    inst['unit']=[unit(c,q) for c,q in zip(inst['chart_context'],inst['question'])]
    inst['pred_bucket']=[bucket(x) for x in inst[ans_col]]
    return df,inst

def split(inst,seed):
    ids=inst['candidate_id'].to_numpy(); strata=inst['scale_error_subtype'].fillna('unknown').to_numpy(); groups=inst['chart_group'].to_numpy()
    try:
        g=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed); td,te=next(g.split(ids,strata,groups))
    except Exception:
        s=StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=seed); td,te=next(s.split(ids,strata))
    td_df=inst.iloc[td].reset_index(drop=True)
    try:
        g=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+999); tr,dv=next(g.split(td_df['candidate_id'],td_df['scale_error_subtype'],td_df['chart_group']))
    except Exception:
        s=StratifiedShuffleSplit(n_splits=1,test_size=.25,random_state=seed+999); tr,dv=next(s.split(td_df['candidate_id'],td_df['scale_error_subtype'].fillna('unknown')))
    return set(td_df.iloc[tr]['candidate_id']),set(td_df.iloc[dv]['candidate_id']),set(inst.iloc[te]['candidate_id'])

def choose(df,choices):
    vals=[]
    for cid,pol in choices.items():
        sub=df[(df.candidate_id==cid)&(df.policy==pol)]
        if len(sub): vals.append(float(sub.score.iloc[0]))
    return float(np.mean(vals)) if vals else float('nan')

def policy_means(df,ids):
    return df[df.candidate_id.isin(ids)].groupby('policy')['score'].mean().to_dict()

def learn_table(df,inst,train_ids,key_cols,min_n):
    tr=df[df.candidate_id.isin(train_ids)].merge(inst[['candidate_id']+key_cols],on='candidate_id',how='left')
    global_best=max(policy_means(df,train_ids),key=policy_means(df,train_ids).get)
    table={}
    for key,g in tr.groupby(key_cols,dropna=False):
        if not isinstance(key,tuple): key=(key,)
        if g.candidate_id.nunique() < min_n: continue
        table[key]=g.groupby('policy')['score'].mean().sort_values(ascending=False).index[0]
    return table,global_best

def apply_table(inst,ids,key_cols,table,fallback):
    out={}
    for r in inst[inst.candidate_id.isin(ids)].itertuples(index=False):
        key=tuple(getattr(r,c) for c in key_cols)
        out[r.candidate_id]=table.get(key,fallback)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--policy-file',action='append',required=True)
    ap.add_argument('--chart-context-csv',required=True)
    ap.add_argument('--manifest-csv',default='')
    ap.add_argument('--out-dir',default='results/scaleqa_natural/rule_table_router')
    ap.add_argument('--name',default='scaleqa520_rule_table_router')
    ap.add_argument('--seeds',type=int,default=50)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    df,inst=build(args.policy_file,args.chart_context_csv,args.manifest_csv)
    configs=[['qtype'],['unit'],['pred_bucket'],['qtype','unit'],['qtype','pred_bucket'],['unit','pred_bucket'],['qtype','unit','pred_bucket']]
    mins=[1,3,5,10]
    rows=[]
    for seed in range(args.seeds):
        tr,dv,te=split(inst,seed)
        best_dev=-1; best_cfg=None; best_table=None; best_fb=None
        for cfg in configs:
            for mn in mins:
                table,fb=learn_table(df,inst,tr,cfg,mn)
                acc=choose(df,apply_table(inst,dv,cfg,table,fb))
                if acc>best_dev:
                    best_dev=acc; best_cfg=(cfg,mn); best_table=table; best_fb=fb
        te_choices=apply_table(inst,te,best_cfg[0],best_table,best_fb)
        fixed=policy_means(df,te)
        train_fixed=policy_means(df,tr)
        train_best=max(train_fixed,key=train_fixed.get)
        row={'seed':seed,'router':choose(df,te_choices),'dev':best_dev,'cfg':'+'.join(best_cfg[0]),'min_n':best_cfg[1],'fallback':best_fb,'train_selected_fixed':fixed.get(train_best,float('nan')),'oracle':df[df.candidate_id.isin(te)].groupby('candidate_id')['score'].max().mean()}
        row.update({f'fixed_{k}':v for k,v in fixed.items()})
        rows.append(row)
    res=pd.DataFrame(rows)
    res_path=out/f'{args.name}_results.csv'; res.to_csv(res_path,index=False)
    metrics=[c for c in res.columns if c.startswith('fixed_')]+['router','train_selected_fixed','oracle']
    summ={'name':args.name,'n':int(inst.candidate_id.nunique()),'metrics':{c:{'mean':float(res[c].mean()),'std':float(res[c].std(ddof=1)),'min':float(res[c].min()),'max':float(res[c].max())} for c in metrics if c in res},'config_counts':res[['cfg','min_n']].astype(str).agg(':'.join,axis=1).value_counts().to_dict(),'files':{'results':str(res_path)}}
    (out/f'{args.name}_summary.json').write_text(json.dumps(summ,indent=2),encoding='utf-8')
    print(json.dumps(summ,indent=2))

if __name__=='__main__': main()
