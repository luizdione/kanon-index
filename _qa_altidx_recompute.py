#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KANON-QA 2026-05-31: recompute alt-index hyperauthorship correlations under the
FIELD-RELATIVE canonical KANON, joining altindices_cache.json to the rebuilt
_per_researcher_full.csv. Compares with v8 (json) and v9 (tex)."""
import json, numpy as np, pandas as pd

def avg_ranks(s):
    s=np.asarray(s,float); order=np.argsort(s,kind='mergesort')
    r=np.empty(len(s)); r[order]=np.arange(1,len(s)+1)
    ss=s[order]; i=0
    while i<len(s):
        j=i
        while j+1<len(s) and ss[j+1]==ss[i]: j+=1
        if j>i: r[order[i:j+1]]=(i+1+j+1)/2.0
        i=j+1
    return r
def spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float)
    m=~(np.isnan(a)|np.isnan(b));
    if m.sum()<3: return float('nan')
    return float(np.corrcoef(avg_ranks(a[m]),avg_ranks(b[m]))[0,1])

cache=json.load(open('dados_reais/2026-05-30/altindices_cache.json'))
rows=[]
for orcid,v in cache.items():
    if not isinstance(v,dict) or 'error' in v: continue
    if v.get('data_quality')!='ok': continue
    if v.get('n_works',0)<=0: continue
    if abs(v.get('h_recomp',0)-v.get('h_csv',0))>3: continue
    rows.append(dict(orcid=orcid, field=v['field'], group=v['group'],
                     g=v.get('g'), h_I=v.get('h_I'), h_m=v.get('h_m'), h=v.get('h_recomp')))
val=pd.DataFrame(rows)
print("validated cohort n=",len(val)," nobel=",(val.group.str.contains('aureate',case=False)|val.group.str.contains('obel',case=False)).sum())
print("group values:", val.group.unique())

pr=pd.read_csv('_per_researcher_full.csv')[['orcid','field','KANON','A','avg_authors_per_paper']]
m=val.merge(pr,on='orcid',how='inner',suffixes=('','_pr'))
print("merged n=",len(m))
print("per-field n:", m.groupby('field').size().to_dict())

FIELDS=['Medicine','Physics','Chemistry','Economics']
print("\n=== Hyperauthorship sensitivity: Spearman(index, avg_authors_per_paper) ===")
print(f"{'field':10s} {'KANON':>8} {'A':>8} {'h_I':>8} {'h_m':>8} {'g':>8} {'h':>8}")
for f in FIELDS:
    d=m[m.field==f]
    aa=d['avg_authors_per_paper'].values
    print(f"{f:10s} {spearman(d.KANON,aa):8.3f} {spearman(d.A,aa):8.3f} {spearman(d.h_I,aa):8.3f} {spearman(d.h_m,aa):8.3f} {spearman(d.g,aa):8.3f} {spearman(d.h,aa):8.3f}")
aa=m['avg_authors_per_paper'].values
print(f"{'POOLED':10s} {spearman(m.KANON,aa):8.3f} {spearman(m.A,aa):8.3f} {spearman(m.h_I,aa):8.3f} {spearman(m.h_m,aa):8.3f} {spearman(m.g,aa):8.3f} {spearman(m.h,aa):8.3f}")

print("\n=== Cross-index Spearman (pooled validated cohort) ===")
print("KANON-h_I =%.3f  KANON-h_m =%.3f  KANON-g =%.3f  KANON-h =%.3f  h_I-h_m =%.3f"%(
    spearman(m.KANON,m.h_I), spearman(m.KANON,m.h_m), spearman(m.KANON,m.g),
    spearman(m.KANON,m.h), spearman(m.h_I,m.h_m)))

print("\n--- v8 (json-backed) KANON: -0.89/-0.81/-0.71/-0.81 ; A: -0.95/-0.93/-0.97/-1.00 ; KANON-h_I 0.50 KANON-h_m 0.39")
print("--- v9 (tex)        KANON: -0.83/-0.58/-0.87/-0.50 ; A: -0.93/-0.95/-0.95/-0.99 ; KANON-h_I 0.62 KANON-h_m 0.52")
