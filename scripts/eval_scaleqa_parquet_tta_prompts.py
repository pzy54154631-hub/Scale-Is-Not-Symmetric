#!/usr/bin/env python
from __future__ import annotations
import argparse, ast, json, os, re, io
from PIL import Image
from collections import defaultdict
from pathlib import Path
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

PROMPTS={
'direct': 'Read the image carefully and answer the question. Return only the final short answer.\nQuestion: {question}',
'scale': 'Read the chart carefully and answer the question. Preserve the original unit and scale shown in the chart. If the value is 59, answer 59, not 0.59. If the value is 0.59, answer 0.59, not 59. Return only the final short answer.\nQuestion: {question}',
'axis': 'Read the axis labels, tick marks, legend, and data values carefully. Use the chart\'s displayed numbers exactly. Return only one short final answer.\nQuestion: {question}',
'math': 'Solve the chart question carefully. If arithmetic is required, identify the relevant values from the chart and compute silently. Do not explain. Return only the final answer.\nQuestion: {question}',
'self_check': 'Read the chart and answer the question. Before giving the answer, silently check whether the answer should be a decimal, percentage, or whole number according to the chart labels. Return only the final answer.\nQuestion: {question}',
'unit_guard': 'Answer the chart question using the same numeric scale as the chart. Do not rescale percentages, fractions, or axis values unless the question explicitly asks for conversion. Return only the final short answer.\nQuestion: {question}',
}
NUM_RE=re.compile(r'[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?', re.I)

def to_list(x):
    if isinstance(x,list): return x
    if x is None: return []
    s=str(x).strip()
    for fn in (json.loads, ast.literal_eval):
        try:
            y=fn(s); return y if isinstance(y,list) else [y]
        except Exception: pass
    return [s]

def norm(x):
    s=str(x).strip().lower().replace('−','-').replace(',','').replace('%',' percent ')
    s=re.sub(r'\s+',' ',s)
    return s.strip(' .,:;"\'[](){}')

def nums(x):
    out=[]
    for m in NUM_RE.findall(str(x).lower().replace(',','').replace('−','-')):
        try: out.append(float(m))
        except Exception: pass
    return out

def close(p,g):
    return abs(p-g) <= (1e-4 if abs(g)<1e-12 else max(0.05*abs(g),1e-3))

def score(pred, answers):
    pn=norm(pred); al=[a for a in to_list(answers) if str(a).strip()]
    ansn=[norm(a) for a in al]
    if any(pn==a for a in ansn): return 1
    ps=nums(pred)
    for a in al:
        for p in ps:
            for g in nums(a):
                if close(p,g): return 1
    for a in ansn:
        if len(a)>=3 and len(pn)>=3 and (a in pn or pn in a): return 1
    return 0

def clean(x):
    s=str(x).strip()
    for pat in [r'FINAL\s*:\s*(.+)', r'answer\s*[:：]\s*(.+)']:
        m=re.search(pat,s,flags=re.I)
        if m: s=m.group(1)
    s=s.splitlines()[0].strip().strip(' .,:;"\'[](){}')
    if len(s.split())>10:
        ns=nums(s)
        if ns:
            v=ns[-1]
            return str(int(round(v))) if abs(v-round(v))<1e-6 else f'{v:.6g}'
    return s

def majority(preds):
    d=defaultdict(list)
    for name,p in preds:
        k=norm(p)
        if k: d[k].append((name,p))
    if not d:
        return preds[0][1]
    counts=sorted([(len(v),k,v) for k,v in d.items()], reverse=True, key=lambda x:x[0])
    if len(counts)==1 or counts[0][0]>counts[1][0]: return counts[0][2][0][1]
    for name,p in preds:
        if name=='direct': return p
    return counts[0][2][0][1]

def normalize_image(image):
    if isinstance(image, Image.Image):
        return image.convert('RGB')
    if isinstance(image, dict):
        if image.get('bytes') is not None:
            return Image.open(io.BytesIO(image['bytes'])).convert('RGB')
        if image.get('path'):
            return Image.open(image['path']).convert('RGB')
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert('RGB')
    return image

def gen(model,processor,image,prompt,max_new_tokens):
    image=normalize_image(image)
    msg=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':prompt}]}]
    text=processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs=process_vision_info(msg)
    inputs=processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors='pt')
    device=next(model.parameters()).device
    inputs={k:v.to(device) if hasattr(v,'to') else v for k,v in inputs.items()}
    with torch.no_grad():
        out=model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False)
    trimmed=[o[len(i):] for i,o in zip(inputs['input_ids'],out)]
    raw=processor.batch_decode(trimmed,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
    return clean(raw), raw

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--parquet',required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--model-dir',default='/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--adapter',default='outputs/adapters/chartqa_answer_only_3000_r16')
    ap.add_argument('--max-samples',type=int,default=None)
    ap.add_argument('--max-new-tokens',type=int,default=32)
    args=ap.parse_args()
    df=pd.read_parquet(args.parquet)
    if args.max_samples: df=df.iloc[:args.max_samples].copy()
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    model=Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_dir,torch_dtype=torch.float16,device_map='auto',local_files_only=True)
    if args.adapter and args.adapter.lower()!='none':
        model=PeftModel.from_pretrained(model,args.adapter,is_trainable=False)
    model.eval()
    rows=[]
    for idx,ex in df.iterrows():
        image=ex['image']; q=str(ex['query']); answers=ex['label']; cid=ex.get('candidate_id',idx)
        preds=[]; scores=[]
        for pname,tmpl in PROMPTS.items():
            pred,raw=gen(model,processor,image,tmpl.format(question=q),args.max_new_tokens)
            sc=score(pred,answers)
            preds.append((pname,pred)); scores.append(sc)
            rows.append({'adapter':pname,'idx':idx,'candidate_id':cid,'question':q,'answers':json.dumps(to_list(answers),ensure_ascii=False),'prediction':pred,'raw':raw,'score':sc,'scale_error_subtype':ex.get('scale_error_subtype','')})
        maj=majority(preds); maj_sc=score(maj,answers)
        rows.append({'adapter':'tta_majority','idx':idx,'candidate_id':cid,'question':q,'answers':json.dumps(to_list(answers),ensure_ascii=False),'prediction':maj,'raw':json.dumps(dict(preds),ensure_ascii=False),'score':maj_sc,'scale_error_subtype':ex.get('scale_error_subtype','')})
        rows.append({'adapter':'tta_oracle','idx':idx,'candidate_id':cid,'question':q,'answers':json.dumps(to_list(answers),ensure_ascii=False),'prediction':'','raw':json.dumps(dict(preds),ensure_ascii=False),'score':max(scores),'scale_error_subtype':ex.get('scale_error_subtype','')})
        if len({r['idx'] for r in rows}) % 25 == 0:
            done=len(set(r['idx'] for r in rows)); print('done',done,'direct_acc',pd.DataFrame(rows).query("adapter=='direct'")['score'].mean(), flush=True)
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    res=pd.DataFrame(rows); res.to_csv(out,index=False)
    summary={'overall':res.groupby('adapter')['score'].mean().to_dict(),'by_subset':res.groupby(['adapter','scale_error_subtype'])['score'].mean().reset_index().to_dict('records')}
    out.with_suffix(out.suffix+'.summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary['overall'],indent=2))
if __name__=='__main__': main()
