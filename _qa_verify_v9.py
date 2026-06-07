#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KANON-QA 2026-05-31: verify v9 headline numbers against real data via
field-relative (canonical) components and the canonical v6 weights."""
import json, numpy as np, pandas as pd
import kanon_audit_lib as k

FIELDS = k.FIELDS
W = json.load(open('dados_reais/kanon_optimized_weights_v4.json'))

def dq(d):
    return ((d['total_publications'] >= 30) & (d['total_citations'] >= 1000) &
            (d['complexity_n_works_analyzed'] >= 10))

comb = k.load_combined().reset_index(drop=True)
print("TOTAL combined rows:", len(comb), "| nobel:", int(comb.is_nobel.sum()))

# clean per field counts
print("\n=== clean counts per field (data_quality ok) ===")
tot=0
for f in FIELDS:
    d = comb[comb.field==f]; dd = d[dq(d)]
    nN=int(dd.is_nobel.sum()); nE=int((dd.is_nobel==0).sum())
    print(f"  {f}: clean total {len(dd)} (nobel {nN}, elite {nE})"); tot+=len(dd)
print("  TOTAL clean:", tot)

print("\n=== Standalone component AUCs (field-relative, corrected) ===")
print(f"{'field':10s} {'C':>6} {'A':>6} {'S':>6} {'J':>6} {'h':>6} {'KANON':>6}")
medA_nob=[]; medA_eli=[]
rawk_med={}
for f in FIELDS:
    d = comb[comb.field==f].copy(); d=d[dq(d)].reset_index(drop=True)
    comp = k.comp_corrected(d)
    th = W[f]; theta=[th['wC'],th['wA'],th['wS'],th['wJ'],th['beta'],th['alpha']]
    C,A,S,J = k.components_ASCJ(comp, theta)
    y=d['is_nobel'].values; h=d['h_index_use'].values
    kan = k.kanon_score(comp, theta)
    aucC=k.roc_auc(y,C); aucA=k.roc_auc(y,A); aucS=k.roc_auc(y,S)
    aucJ=k.roc_auc(y,J); auch=k.roc_auc(y,h); auck=k.roc_auc(y,kan)
    print(f"{f:10s} {aucC:6.3f} {aucA:6.3f} {aucS:6.3f} {aucJ:6.3f} {auch:6.3f} {auck:6.3f}")
    medA_nob.append((f, np.median(A[y==1]))); medA_eli.append((f,np.median(A[y==0])))
    rawk_med[f]=np.median(kan)

print("\n=== median A laureate vs elite (pooled over clean) ===")
# pooled needs per-field A then concat
allA=[]; ally=[]
for f in FIELDS:
    d=comb[comb.field==f].copy(); d=d[dq(d)].reset_index(drop=True)
    comp=k.comp_corrected(d); th=W[f]; theta=[th['wC'],th['wA'],th['wS'],th['wJ'],th['beta'],th['alpha']]
    _,A,_,_=k.components_ASCJ(comp,theta)
    allA.append(A); ally.append(d['is_nobel'].values)
allA=np.concatenate(allA); ally=np.concatenate(ally)
print(f"  median A nobel={np.median(allA[ally==1]):.3f}  elite={np.median(allA[ally==0]):.3f}")

print("\n=== raw KANON median by field ===")
for f in FIELDS: print(f"  {f}: {rawk_med[f]:.1f}")

print("\n=== KANON_pctl medians (within-field percentile of raw KANON) ===")
# build per-field percentile then pooled medians by group
allp=[]; ally2=[]
for f in FIELDS:
    d=comb[comb.field==f].copy(); d=d[dq(d)].reset_index(drop=True)
    comp=k.comp_corrected(d); th=W[f]; theta=[th['wC'],th['wA'],th['wS'],th['wJ'],th['beta'],th['alpha']]
    kan=k.kanon_score(comp,theta)
    pr=k._avg_ranks(kan)/len(kan)*100.0
    allp.append(pr); ally2.append(d['is_nobel'].values)
allp=np.concatenate(allp); ally2=np.concatenate(ally2)
print(f"  median KANON_pctl nobel={np.median(allp[ally2==1]):.1f}  elite={np.median(allp[ally2==0]):.1f}")

print("\n=== exemplars: Mayor, Parisi, Hargittai (A under canonical Physics/Chem alpha) ===")
for nm in ['Mayor','Parisi','Hargittai']:
    rows=comb[comb['name'].str.contains(nm, case=False, na=False)] if 'name' in comb.columns else pd.DataFrame()
    for _,r in rows.iterrows():
        f=r['field']; th=W.get(f)
        if th is None: continue
        alpha=th['alpha']
        apos=np.clip(1.0-r['avg_author_position']/20.0,0,1)
        avg=max(r['avg_authors_per_paper'],1.0)
        A=np.clip(apos/avg**alpha,0,1)
        print(f"  {r['name'][:28]:28s} field={f:9s} h={r.get('h_index','?')} "
              f"authors/paper={r['avg_authors_per_paper']:.2f} alpha={alpha} A={A:.3f}")

print("\n=== KANON vs KANON_global Spearman per field ===")
# global weights = mean of per-field? Need a single vector. Use mean of the 4 field weight vectors (renorm).
wmat=np.array([[W[f]['wC'],W[f]['wA'],W[f]['wS'],W[f]['wJ']] for f in FIELDS])
gw=wmat.mean(axis=0); gw=gw/gw.sum()
gbeta=np.mean([W[f]['beta'] for f in FIELDS]); galpha=np.mean([W[f]['alpha'] for f in FIELDS])
def spear(a,b):
    ra=k._avg_ranks(a); rb=k._avg_ranks(b)
    return np.corrcoef(ra,rb)[0,1]
for f in FIELDS:
    d=comb[comb.field==f].copy(); d=d[dq(d)].reset_index(drop=True)
    comp=k.comp_corrected(d); th=W[f]
    kan=k.kanon_score(comp,[th['wC'],th['wA'],th['wS'],th['wJ'],th['beta'],th['alpha']])
    kang=k.kanon_score(comp,[gw[0],gw[1],gw[2],gw[3],gbeta,galpha])
    print(f"  {f}: spearman(KANON,KANON_global)={spear(kan,kang):.3f}")
