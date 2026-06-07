#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KANON-QA 2026-05-31: regenerate dados_reais/2026-05-30/altindices_results.json
under the FIELD-RELATIVE canonical KANON.

The prior (interrupted) field-relative migration recomputed the alt-index
correlations into kanon_resultados_v9.tex but did NOT overwrite the on-disk
artefact, which still held the pre-correction (saturating-KANON) values. This
script reads the unchanged per-author g/h_I/h_m cache and the rebuilt
field-relative _per_researcher_full.csv, recomputes the within-author
relationships, and rewrites the artefact so every number in v9 is traceable.

g, h_I, h_m, h are UNCHANGED (they do not depend on KANON's C/J definition);
only KANON and A change because the canonical components are now field-relative.
"""
import json, numpy as np, pandas as pd

def avg_ranks(s):
    s=np.asarray(s,float); o=np.argsort(s,kind='mergesort'); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    ss=s[o]; i=0
    while i<len(s):
        j=i
        while j+1<len(s) and ss[j+1]==ss[i]: j+=1
        if j>i: r[o[i:j+1]]=(i+1+j+1)/2.0
        i=j+1
    return r
def spear(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); m=~(np.isnan(a)|np.isnan(b))
    if m.sum()<3: return float('nan')
    return round(float(np.corrcoef(avg_ranks(a[m]),avg_ranks(b[m]))[0,1]),3)

CACHE='dados_reais/2026-05-30/altindices_cache.json'
OUT='dados_reais/2026-05-30/altindices_results.json'
cache=json.load(open(CACHE))
rows=[]
for orcid,v in cache.items():
    if not isinstance(v,dict) or 'error' in v: continue
    if v.get('data_quality')!='ok' or v.get('n_works',0)<=0: continue
    if abs(v.get('h_recomp',0)-v.get('h_csv',0))>3: continue
    rows.append(dict(orcid=orcid, field=v['field'], group=v['group'],
                     g=v.get('g'), h_I=v.get('h_I'), h_m=v.get('h_m'), h=v.get('h_recomp')))
val=pd.DataFrame(rows)
pr=pd.read_csv('_per_researcher_full.csv')[['orcid','KANON','A','avg_authors_per_paper']]
m=val.merge(pr,on='orcid',how='inner')
FIELDS=['Medicine','Physics','Chemistry','Economics']
INDICES=['KANON','h','A','h_I','g','h_m']
colmap={'KANON':'KANON','h':'h','A':'A','h_I':'h_I','g':'g','h_m':'h_m'}

cross={}
for a in INDICES:
    cross[a]={}
    for b in INDICES:
        cross[a][b]=1.0 if a==b else spear(m[colmap[a]],m[colmap[b]])
aa=m['avg_authors_per_paper'].values
pooled={ix: spear(m[colmap[ix]],aa) for ix in INDICES}
perfield={}
for f in FIELDS:
    d=m[m.field==f]; av=d['avg_authors_per_paper'].values
    perfield[f]={'n':int(len(d)), **{ix: spear(d[colmap[ix]],av) for ix in INDICES}}

n_nob=int((m.group.str.contains('obel',case=False)|m.group.str.contains('aureate',case=False)).sum())
out=dict(
    n_validated=int(len(m)), n_nobel=n_nob, n_elite=int(len(m)-n_nob),
    cross_spearman=cross, hyperauth_sensitivity_pooled=pooled,
    hyperauth_sensitivity_perfield=perfield,
    note=("ORCID-validated subset only: cache profiles with data_quality==ok, n_works>0 "
          "and |h_recomp-h_csv|<=3. KANON and A use the FIELD-RELATIVE canonical components "
          "(regenerated 2026-05-31 to match the field-relative migration; g/h_I/h_m/h are "
          "definition-independent and unchanged). 150/164 laureates have ORCIDs unlinked in "
          "OpenAlex, so laureate-vs-elite AUC for g/h_I/h_m is NOT computable here and is "
          "deferred to a host run with canonical OpenAlex author IDs. Cross-index and "
          "hyperauthorship-sensitivity correlations are within-author relationships."))
json.dump(out,open(OUT,'w'),indent=1)
print('WROTE',OUT)
print('KANON per-field:', {f:perfield[f]['KANON'] for f in FIELDS})
print('A per-field:', {f:perfield[f]['A'] for f in FIELDS})
print('KANON-h_I=%s  KANON-h_m=%s  h_I-h_m=%s'%(cross['KANON']['h_I'],cross['KANON']['h_m'],cross['h_I']['h_m']))
