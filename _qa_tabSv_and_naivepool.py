# -*- coding: utf-8 -*-
"""KANON-QA 2026-05-31 (follow-up): (c.2) regenerate tab:Sv under field-relative
canonical pipeline; (c.3) compute naive-pool h-index AUC framings.

Methodology (documented):
- Components C, A, J are the FIELD-RELATIVE canonical columns already in
  _per_researcher_full.csv. Per-field weights from kanon_optimized_weights_v4.json.
- For each S-variant, KANON' = 100*(wC*C + wA*A + wS*S' + wJ*J) with the canonical
  per-field weights; for v10 (drop S) the weights wC,wA,wJ are renormalised to sum 1.
- Mean AUC = mean over the 4 fields of in-sample AUC(is_nobel, KANON') on CLEAN
  profiles (data_quality=='ok'), matching the original tab:Sv ("canonical per-field
  weights", in-sample, repaired benchmark).
- rank_shift' = rank_h_clean - rank_KANON'_clean (per field, descending min-rank).
- Mean rho rank-shift = mean over fields of Spearman(avg_authors_per_paper, rank_shift').
"""
import json, numpy as np, pandas as pd
import kanon_audit_lib as k

FIELDS=['Medicine','Physics','Chemistry','Economics']
W=json.load(open('dados_reais/kanon_optimized_weights_v4.json'))
d=pd.read_csv('_per_researcher_full.csv')
cl=d[d.data_quality=='ok'].copy().reset_index(drop=True)
cl['is_nobel']=cl['group'].str.contains('Nobel',case=False).astype(int)

def desc_rank(x):
    return pd.Series(x).rank(ascending=False, method='min').values
def spearman(a,b):
    a=np.asarray(a,float); b=np.asarray(b,float); m=~(np.isnan(a)|np.isnan(b))
    return float(np.corrcoef(k._avg_ranks(a[m]),k._avg_ranks(b[m]))[0,1])
def pctl_within(series, fields):
    out=np.zeros(len(series))
    for f in np.unique(fields):
        msk=fields==f
        out[msk]=k._avg_ranks(series[msk])/msk.sum()
    return out

fld=cl['field'].values
S_text=np.clip(cl['S_text_raw'].values,0,1)
S_mesh=np.clip(cl['mesh_coverage'].values,0,1)

def Svar(name):
    if name=='v1':  return cl['S'].values
    if name=='v2':  return S_text
    if name=='v3':  return np.maximum(S_text,S_mesh)
    if name=='v6':  return pctl_within(S_text,fld)
    if name=='v9':  return 1.0-pctl_within(S_text,fld)
    if name=='v11': return S_mesh
    if name=='v12': return pctl_within(S_mesh,fld)
    return None

def kanon_with_S(Sp, drop_S=False):
    out=np.zeros(len(cl))
    for f in FIELDS:
        m=fld==f; th=W[f]; wC,wA,wS,wJ=th['wC'],th['wA'],th['wS'],th['wJ']
        if drop_S:
            s=wC+wA+wJ; wC,wA,wJ=wC/s,wA/s,wJ/s; wS=0.0
        out[m]=100*(wC*cl['C'].values[m]+wA*cl['A'].values[m]+wS*(0 if drop_S else Sp[m])+wJ*cl['J'].values[m])
    return out

def mean_auc(kan):
    a=[]
    for f in FIELDS:
        m=fld==f
        a.append(k.roc_auc(cl['is_nobel'].values[m], kan[m]))
    return np.mean(a), a

def mean_rankshift_rho(kan):
    rhos=[]
    for f in FIELDS:
        m=fld==f
        rk=desc_rank(kan[m]); rh=desc_rank(cl['h_index'].values[m])
        rs=rh-rk
        rhos.append(spearman(cl['avg_authors_per_paper'].values[m], rs))
    return np.mean(rhos), rhos

print("=== (c.2) tab:Sv regenerated under FIELD-RELATIVE canonical ===")
print(f"{'variant':22s} {'meanAUC':>8} {'dVS v1':>8} {'mean rho rankshift':>20}")
rows={}
kan_v1=kanon_with_S(Svar('v1'))
auc_v1,_=mean_auc(kan_v1)
rho_v1,rho_v1f=mean_rankshift_rho(kan_v1)
for name,label in [('v1','v1 (canonical)'),('v2','v2 text-only'),('v3','v3 max(text,MeSH)'),
                   ('v6','v6 pctl(S_text)'),('v9','v9 invert pctl'),('v10','v10 drop S (3-comp)'),
                   ('v11','v11 MeSH-only'),('v12','v12 pctl(MeSH)')]:
    if name=='v10':
        kan=kanon_with_S(None,drop_S=True)
    else:
        kan=kanon_with_S(Svar(name))
    auc,af=mean_auc(kan); rows[name]=auc
    drho = ''
    if name in ('v1','v10'):
        rho,rf=mean_rankshift_rho(kan); drho=f"{rho:+.3f}"
    print(f"{label:22s} {auc:8.3f} {auc-auc_v1:+8.3f} {drho:>20}")
print(f"\n  v1 per-field AUC: {[round(x,3) for x in mean_auc(kan_v1)[1]]}")
print(f"  v1 per-field rankshift rho: {[round(x,3) for x in rho_v1f]}")

print("\n=== (c.3) naive-pool h-index AUC framings ===")
# field-matched (baseline, should be ~0.51-0.64)
fm=[]
for f in FIELDS:
    m=fld==f
    fm.append(k.roc_auc(cl['is_nobel'].values[m], cl['h_index'].values[m]))
print("field-matched h-AUC per field:", [round(x,3) for x in fm], "mean", round(np.mean(fm),3))
# naive A: each field's laureates vs ALL elites pooled (cross-field), h-index
elite_all=cl[cl.is_nobel==0]
naiveA=[]
for f in FIELDS:
    lf=cl[(cl.field==f)&(cl.is_nobel==1)]
    y=np.r_[np.ones(len(lf)), np.zeros(len(elite_all))]
    s=np.r_[lf['h_index'].values, elite_all['h_index'].values]
    naiveA.append(k.roc_auc(y,s))
print("naive-A (field laureates vs ALL-field elite pool) h-AUC:", [round(x,3) for x in naiveA], "max", round(max(naiveA),3))
# naive B: all laureates vs all elites, ignoring field
y=cl['is_nobel'].values
print("naive-B (all laureates vs all elites pooled, h):", round(k.roc_auc(y, cl['h_index'].values),3))
# naive C: laureates vs elites WITHIN field but pooling laureates across (i.e., global)
print("global pooled h-AUC (all, by h):", round(k.roc_auc(cl['is_nobel'].values, cl['h_index'].values),3))
