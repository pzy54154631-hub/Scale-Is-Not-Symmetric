#!/usr/bin/env python
from __future__ import annotations
import argparse, ast, io, json, math, re, sys
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
sys.path.append('scripts')
from eval_scaleqa_candidate_verifier import normalize_image, score, to_list

LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
NUM_RE=re.compile(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?", re.I)
PRIOR_ORDER=[
 'tta:self_check','tta:axis','tta:direct','tta:unit_guard','tta:math','tta:tta_majority','main:answer_tuned',
 'main:large_seed4','main:large_seed5','main:large_seed2','main:large_seed3','main:large_seed1','tta:scale','main:base',
 'verifier:candidate_majority','verifier:letter_axis','verifier:letter_scale_guard'
]

def norm_text(x):
    return re.sub(r'\s+',' ',str(x).strip().lower().replace(',',''))

def parse_num(x):
    m=NUM_RE.search(str(x).replace(',','').replace('−','-'))
    if not m: return None
    try: return float(m.group(0))
    except Exception: return None

def fmt_num(v):
    if v is None or not math.isfinite(v): return None
    return str(int(round(v))) if abs(v-round(v))<1e-8 else f"{v:.8g}"

def build_pool(candidate_csvs, max_candidates=14, add_scale_variants=True):
    frames=[]
    for p in candidate_csvs:
        df=pd.read_csv(p)
        if 'candidate_id' not in df.columns or 'prediction' not in df.columns: continue
        if 'adapter' not in df.columns: df['adapter']=Path(p).stem
        frames.append(df[['candidate_id','adapter','prediction']].dropna())
    all_df=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=['candidate_id','adapter','prediction'])
    rank={a:i for i,a in enumerate(PRIOR_ORDER)}
    out={}
    for cid,g in all_df.groupby('candidate_id'):
        rows=[]; seen=set()
        g=g.copy(); g['rank']=g['adapter'].map(lambda x: rank.get(str(x),999))
        for _,r in g.sort_values('rank').iterrows():
            pred=str(r.prediction).strip()
            if not pred: continue
            key=norm_text(pred)
            if key not in seen:
                seen.add(key); rows.append((str(r.adapter),pred))
            if add_scale_variants:
                v=parse_num(pred)
                if v is not None:
                    for tag,mul in [('scale_x100',100),('scale_div100',0.01),('scale_x10',10),('scale_div10',0.1)]:
                        fp=fmt_num(v*mul)
                        if fp and norm_text(fp) not in seen:
                            seen.add(norm_text(fp)); rows.append((tag,fp))
            if len(rows)>=max_candidates: break
        out[str(cid)]=rows[:max_candidates]
    return out

def load_context(path):
    if not path or not Path(path).exists(): return {}
    df=pd.read_csv(path)
    cid_col='candidate_id' if 'candidate_id' in df.columns else None
    if not cid_col: return {}
    # chart_context may exist, otherwise synthesize from columns.
    ctx={}
    for _,r in df.iterrows():
        if 'chart_context' in df.columns and pd.notna(r.get('chart_context')):
            ctx[str(r[cid_col])]=str(r['chart_context'])
        else:
            parts=[]
            for c in ['title','x_axis','y_axis','units','ticks','legend','notes']:
                if c in df.columns and pd.notna(r.get(c)): parts.append(f'{c}={r[c]}')
            ctx[str(r[cid_col])]='; '.join(parts)
    return ctx

def candidate_block(cands):
    return '\n'.join(f'{LETTERS[i]}. {pred}' for i,(_,pred) in enumerate(cands)) or 'A. unknown'

def parse_letter(raw,n):
    s=str(raw).upper()
    for pat in [r'LETTER\s*[:=]\s*([A-Z])', r'ANSWER\s*[:=]\s*([A-Z])', r'\b([A-Z])\b']:
        m=re.search(pat,s)
        if m and m.group(1) in LETTERS[:n]: return m.group(1)
    for ch in s.strip():
        if ch in LETTERS[:n]: return ch
    return 'A'

PROMPTS={
 'structured_context_letter': (
   'You are a careful chart-question verifier. Candidate answers may differ only by numeric scale, such as 0.61 vs 61.\n'
   'Use BOTH the image and the structured chart context. Pay special attention to axis tick labels, percent signs, legends, and units.\n'
   'Choose the candidate whose numeric scale matches the chart and question. Output only LETTER: X.\n\n'
   'Structured chart context:\n{context}\n\nQuestion: {question}\nCandidates:\n{candidates}\n\nLETTER:'),
 'structured_scale_audit': (
   'Audit this chart QA answer as a scale problem. First identify the displayed chart unit or scale from the image/context. Then choose the candidate with the required output scale.\n'
   'Do not prefer larger or smaller numbers by default; prefer the number in the unit asked by the question.\n'
   'Return only a single line: LETTER: <one candidate letter>.\n\n'
   'Chart context:\n{context}\n\nQuestion: {question}\nCandidate answers:\n{candidates}\n\nLETTER:'),
}

def gen(model,processor,image,prompt,max_new_tokens):
    image=normalize_image(image)
    msg=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
    text=processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs=process_vision_info(msg)
    inputs=processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
    device=next(model.parameters()).device
    inputs={k:v.to(device) if hasattr(v,'to') else v for k,v in inputs.items()}
    with torch.no_grad():
        out=model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed=[o[len(i):] for i,o in zip(inputs['input_ids'],out)]
    return processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--parquet', required=True)
    ap.add_argument('--candidate-csv', action='append', default=[])
    ap.add_argument('--context-csv', default='')
    ap.add_argument('--output', required=True)
    ap.add_argument('--model-dir', default='/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--adapter', default='outputs/adapters/chartqa_answer_only_3000_r16')
    ap.add_argument('--max-candidates', type=int, default=14)
    ap.add_argument('--max-new-tokens', type=int, default=24)
    args=ap.parse_args()
    df=pd.read_parquet(args.parquet)
    pools=build_pool(args.candidate_csv,args.max_candidates,True)
    contexts=load_context(args.context_csv)
    print(f'[info] examples={len(df)} pools={len(pools)} contexts={len(contexts)}', flush=True)
    processor=AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=torch.float16, device_map='auto', local_files_only=True)
    if args.adapter and args.adapter.lower()!='none': model=PeftModel.from_pretrained(model,args.adapter,is_trainable=False)
    model.eval()
    rows=[]
    for n,(idx,ex) in enumerate(df.iterrows(), start=1):
        cid=str(ex.get('candidate_id',idx)); q=str(ex.get('query',ex.get('question',''))); answers=ex.get('label',ex.get('answers',''))
        cands=pools.get(cid,[])
        if not cands:
            base=str(ex.get('prediction','')) or '0'; cands=[('fallback',base)]
        block=candidate_block(cands)
        oracle=max(score(pred,answers) for _,pred in cands) if cands else 0
        rows.append({'adapter':'structured_candidate_oracle','idx':idx,'candidate_id':cid,'question':q,'answers':json.dumps(to_list(answers),ensure_ascii=False),'prediction':'','raw':json.dumps(cands,ensure_ascii=False),'score':oracle,'scale_error_subtype':ex.get('scale_error_subtype',''),'scale_directions':ex.get('scale_directions',''),'n_candidates':len(cands)})
        ctx=contexts.get(cid,'')
        for pname,tmpl in PROMPTS.items():
            raw=gen(model,processor,ex['image'],tmpl.format(context=ctx or '(not available)',question=q,candidates=block),args.max_new_tokens)
            letter=parse_letter(raw,len(cands)); pred=cands[LETTERS.index(letter)][1]
            rows.append({'adapter':pname,'idx':idx,'candidate_id':cid,'question':q,'answers':json.dumps(to_list(answers),ensure_ascii=False),'prediction':pred,'raw':raw,'score':score(pred,answers),'scale_error_subtype':ex.get('scale_error_subtype',''),'scale_directions':ex.get('scale_directions',''),'n_candidates':len(cands)})
        if n%25==0:
            tmp=pd.DataFrame(rows)
            print(f"[progress] {n}/{len(df)} " + ' '.join(f"{a}={tmp[tmp.adapter==a]['score'].mean():.4f}" for a in PROMPTS), flush=True)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    res=pd.DataFrame(rows); res.to_csv(out,index=False)
    summary={'overall':res.groupby('adapter')['score'].mean().to_dict(), 'by_direction':res.groupby(['adapter','scale_directions'])['score'].mean().reset_index().to_dict('records'), 'by_subtype':res.groupby(['adapter','scale_error_subtype'])['score'].mean().reset_index().to_dict('records'), 'n_examples':len(df)}
    out.with_suffix(out.suffix+'.summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False), flush=True)
if __name__=='__main__': main()

