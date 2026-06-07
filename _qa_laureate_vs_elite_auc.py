# -*- coding: utf-8 -*-
"""Flag 5 analysis: per-field laureate-vs-elite AUC for h, g, h_I, h_m (and KANON, A).
Positives = validated laureates (resolved by canonical OpenAlex author id, |h_recomp-h_csv|<=3).
Negatives = validated elites from altindices_cache.json (ORCID-linked, |h_recomp-h_csv|<=3).
KANON/A from _per_researcher_full.csv (field-relative canonical)."""
import json, numpy as np, pandas as pd
import kanon_audit_lib as k

FIELDS=['Medicine','Physics','Chemistry','Economics']
lau=json.load(open('dados_reais/2026-05-31/laureate_altindices_cache.json'))
eli=json.load(open('dados_reais/2026-05-30/altindices_cache.json'))
pr=pd.read_csv('_per_researcher_full.csv')
pr_by_name=pr.set_index('name'); pr_by_orcid=pr.set_index('orcid')

rows=[]
for nm,v in lau.items():
    if not v.get('resolved') or not v.get('h_match'): continue
    kanon=A=np.nan
    if nm in pr_by_name.index:
        r=pr_by_name.loc[nm]
        if hasattr(r,'iloc') and getattr(r,'ndim',1)>1: r=r.iloc[0]
        kanon=float(r['KANON']); A=float(r['A'])
    rows.append(dict(field=v['field'],label=1,h=v['h_recomp'],g=v['g'],h_I=v['h_I'],h_m=v['h_m'],KANON=kanon,A=A))
for orc,v in eli.items():
    if not isinstance(v,dict) or 'error' in v: continue
    if v.get('data_quality')!='ok' or v.get('n_works',0)<=0: continue
    if abs(v.get('h_recomp',0)-v.get('h_csv',0))>3: continue
    kanon=A=np.nan
    if orc in pr_by_orcid.index:
        r=pr_by_orcid.loc[orc]
        if getattr(r,'ndim',1)>1: r=r.iloc[0]
        kanon=float(r['KANON']); A=float(r['A'])
    rows.append(dict(field=v['field'],label=0,h=v['h_recomp'],g=v['g'],h_I=v['h_I'],h_m=v['h_m'],KANON=kanon,A=A))
df=pd.DataFrame(rows)
print("cohort: laureates(validated)=%d  elites(validated)=%d"%((df.label==1).sum(),(df.label==0).sum()))
print("per-field n (laureate/elite):")
for f in FIELDS:
    d=df[df.field==f]; print("  %s: %d / %d"%(f,(d.label==1).sum(),(d.label==0).sum()))

INDICES=['h','g','h_m','h_I','KANON','A']
print("\n=== Laureate-vs-elite AUC per field (validated cohort) ===")
print(f"{'index':8}"+''.join(f'{f[:4]:>8}' for f in FIELDS)+f"{'POOLED':>9}")
for ix in INDICES:
    line=f"{ix:8}"
    for f in FIELDS:
        d=df[df.field==f].dropna(subset=[ix]) if ix in('KANON','A') else df[df.field==f]
        line+=f"{k.roc_auc(d.label.values,d[ix].values):8.3f}"
    dp=df.dropna(subset=[ix]) if ix in('KANON','A') else df
    # pooled AUC computed within-field then averaged (fields not comparable in raw index)
    aucs=[k.roc_auc(df[df.field==f].dropna(subset=[ix])[ 'label'].values if ix in('KANON','A') else df[df.field==f]['label'].values,
                    df[df.field==f].dropna(subset=[ix])[ix].values if ix in('KANON','A') else df[df.field==f][ix].values) for f in FIELDS]
    line+=f"{np.mean(aucs):9.3f}"
    print(line)
print("\n(POOLED = mean of per-field AUCs; raw indices are not cross-field comparable.)")
df.to_csv('dados_reais/2026-05-31/laureate_vs_elite_altindices.csv',index=False)
print("saved dados_reais/2026-05-31/laureate_vs_elite_altindices.csv")
