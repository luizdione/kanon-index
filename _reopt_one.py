#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-field CV re-optimization on FIELD-RELATIVE components (scipy-free).
Usage: python _reopt_one.py <Field>   ->  writes _reopt_parts/<Field>.json"""
import os, sys, json, numpy as np
import kanon_audit_lib as k

N_WEIGHTS = 2500
K_FOLDS = 5
MAX_W = 0.50
PARTS = '_reopt_parts'

def main(field):
    os.makedirs(PARTS, exist_ok=True)
    comb = k.load_combined().reset_index(drop=True)
    d = comb[comb.field == field].copy().reset_index(drop=True)
    ok = ((d['total_publications'] >= 30) & (d['total_citations'] >= 1000) &
          (d['complexity_n_works_analyzed'] >= 10)).values
    d = d[ok].reset_index(drop=True)
    comp = k.comp_corrected(d)
    y = d['is_nobel'].values; h = d['h_index_use'].values
    n_nob = int((y == 1).sum()); n_eli = int((y == 0).sum())

    def per_fold(c, yy, fi):
        return k.optimize_field(c, yy, n_weights=N_WEIGHTS, max_w=MAX_W, seed=42 + fi)
    cvdf, med = k.cv_evaluate(comp, y, h, per_fold, k=K_FOLDS, seed=42)
    wC, wA, wS, wJ, beta, alpha = [float(x) for x in med]
    rec = dict(wC=round(wC, 4), wA=round(wA, 4), wS=round(wS, 4), wJ=round(wJ, 4),
               beta=round(beta, 4), alpha=round(alpha, 4),
               auc_kanon_mean=round(float(cvdf.auc_kanon.mean()), 4),
               auc_kanon_std=round(float(cvdf.auc_kanon.std()), 4),
               auc_hindex_mean=round(float(cvdf.auc_h.mean()), 4),
               auc_hindex_std=round(float(cvdf.auc_h.std()), 4),
               delong_p_median=round(float(cvdf.delong_p.median()), 4),
               kanon_wins_folds=int((cvdf.auc_kanon > cvdf.auc_h).sum()),
               total_folds=K_FOLDS,
               kanon_superior=bool(int((cvdf.auc_kanon > cvdf.auc_h).sum()) >= 3),
               n_nobel=n_nob, n_elite=n_eli)
    json.dump(rec, open(os.path.join(PARTS, field + '.json'), 'w'), indent=2)
    print(f"[{field}] AUC_K={rec['auc_kanon_mean']:.3f} AUC_h={rec['auc_hindex_mean']:.3f} "
          f"wins {rec['kanon_wins_folds']}/5 w=({rec['wC']:.2f},{rec['wA']:.2f},"
          f"{rec['wS']:.2f},{rec['wJ']:.2f}) beta={rec['beta']:.2f} alpha={rec['alpha']:.2f} "
          f"DeLong p={rec['delong_p_median']:.3f}  (n={n_nob}+{n_eli})")

if __name__ == '__main__':
    main(sys.argv[1])
