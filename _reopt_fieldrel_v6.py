#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KANON re-optimization v6 — CANONICAL field-relative C/J (scipy-free).
=====================================================================
Author decision (2026-05-30): adopt the field-relative percentile C and J
(comp_corrected in kanon_audit_lib) as the CANONICAL components, replacing the
saturating min(1,c/1e4)/min(1,journals/50) used through v5's headline pipeline.

This script re-optimizes the per-field weights on the CORRECTED components using
the reproducible, scipy-free multi-start simplex optimizer in kanon_audit_lib
(Dirichlet sampling of the weight simplex, n_weights samples, x a (beta,alpha)
grid; objective = train AUC-ROC Nobel vs matched elite; 5-fold stratified CV;
DeLong test vs h-index). Canonical weights = cross-validated medians (renorm).

Output: dados_reais/<date>/otimizacao_v6_fieldrel/kanon_optimized_weights_v6.json
and overwrites dados_reais/kanon_optimized_weights_v4.json (the file the rest of
the pipeline reads) with the v6 field-relative weights.
"""
import os, json, numpy as np, pandas as pd
from datetime import datetime
import kanon_audit_lib as k

FIELDS = k.FIELDS
N_WEIGHTS = 4000
K_FOLDS = 5
MAX_W = 0.50

def data_quality(d):
    return ((d['total_publications'] >= 30) & (d['total_citations'] >= 1000) &
            (d['complexity_n_works_analyzed'] >= 10))

def main():
    comb = k.load_combined()
    comb = comb.reset_index(drop=True)
    out = {}
    table_rows = []
    for f in FIELDS:
        d = comb[comb.field == f].copy().reset_index(drop=True)
        ok = data_quality(d).values
        d = d[ok].reset_index(drop=True)            # optimize on clean profiles
        comp = k.comp_corrected(d)                  # FIELD-RELATIVE C/J
        y = d['is_nobel'].values
        h = d['h_index_use'].values
        n_nob = int((y == 1).sum()); n_eli = int((y == 0).sum())

        def per_fold(comp_tr, y_tr, fi):
            return k.optimize_field(comp_tr, y_tr, n_weights=N_WEIGHTS,
                                    max_w=MAX_W, seed=42 + fi)
        cvdf, med = k.cv_evaluate(comp, y, h, per_fold, k=K_FOLDS, seed=42)
        wC, wA, wS, wJ, beta, alpha = med
        auc_k = float(cvdf.auc_kanon.mean()); auc_k_sd = float(cvdf.auc_kanon.std())
        auc_h = float(cvdf.auc_h.mean()); auc_h_sd = float(cvdf.auc_h.std())
        wins = int((cvdf.auc_kanon > cvdf.auc_h).sum())
        dp = float(cvdf.delong_p.median())
        out[f] = dict(wC=round(float(wC), 4), wA=round(float(wA), 4),
                      wS=round(float(wS), 4), wJ=round(float(wJ), 4),
                      beta=round(float(beta), 4), alpha=round(float(alpha), 4),
                      auc_kanon_mean=round(auc_k, 4), auc_kanon_std=round(auc_k_sd, 4),
                      auc_hindex_mean=round(auc_h, 4), auc_hindex_std=round(auc_h_sd, 4),
                      delong_p_median=round(dp, 4),
                      kanon_wins_folds=wins, total_folds=K_FOLDS,
                      kanon_superior=bool(wins >= 3), n_nobel=n_nob, n_elite=n_eli)
        table_rows.append((f, auc_k, auc_h, wins, wC, wA, wS, wJ, beta, alpha, dp))
        print(f"[{f}] AUC_K={auc_k:.3f}+-{auc_k_sd:.3f} AUC_h={auc_h:.3f} wins {wins}/5 "
              f"| wC={wC:.3f} wA={wA:.3f} wS={wS:.3f} wJ={wJ:.3f} beta={beta:.3f} alpha={alpha:.3f} "
              f"DeLong p_med={dp:.3f}", flush=True)

    out['_meta'] = dict(
        version='v6', architecture='4 components (C,A,S,J); C,J = within-field percentile of log (FIELD-RELATIVE, canonical)',
        optimizer=f'scipy-free multi-start simplex (Dirichlet n_weights={N_WEIGHTS}) x (beta,alpha) grid, Stratified {K_FOLDS}-Fold CV',
        constraint='sum(weights)=1, weight in [~0.01,0.5], beta in [0,1], alpha in [0.3,1.0]',
        criterion='AUC-ROC (Nobel vs field-matched elite)', validation='DeLong test vs h-index',
        components='comp_corrected (field-relative C and J)',
        note='Replaces saturating C=min(1,c/1e4), J=min(1,journals/50). Weights are CV medians.',
        generated=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        k_folds=K_FOLDS, n_weights=N_WEIGHTS, max_weight=MAX_W)

    date_tag = datetime.now().strftime('%Y-%m-%d')
    odir = os.path.join('dados_reais', date_tag, 'otimizacao_v6_fieldrel')
    os.makedirs(odir, exist_ok=True)
    json.dump(out, open(os.path.join(odir, 'kanon_optimized_weights_v6.json'), 'w'), indent=2)
    # overwrite the canonical file the pipeline reads
    json.dump(out, open('dados_reais/kanon_optimized_weights_v4.json', 'w'), indent=2)
    print('\nSAVED', os.path.join(odir, 'kanon_optimized_weights_v6.json'))
    print('OVERWROTE dados_reais/kanon_optimized_weights_v4.json (canonical, now v6 field-relative)')
    print('\n=== Table 1 (field-relative, canonical) ===')
    for (f, ak, ah, w, wC, wA, wS, wJ, be, al, dp) in table_rows:
        print(f"{f:10s} AUC_K={ak:.3f} AUC_h={ah:.3f} wins {w}/5 "
              f"w=({wC:.2f},{wA:.2f},{wS:.2f},{wJ:.2f}) beta={be:.2f} alpha={al:.2f}")

if __name__ == '__main__':
    main()
