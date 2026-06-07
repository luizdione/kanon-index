#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KANON-Index Weight Optimizer v5 — CORRECTED, field-relative C and J
====================================================================
Motivation (audit 2026-05-27): the v4 components C = min(1, citations/1e4) and
J = min(1, journals/50) SATURATE at 1.0 for 91-99% of elite profiles, so they
carry almost no signal and do NOT match the paper's field-relative equations.
v5 replaces both with the WITHIN-FIELD PERCENTILE of the log value (non-degenerate,
field-relative), keeping A and S identical to v4. Everything else (SLSQP multi-start,
5-fold stratified CV, DeLong, sensitivity, dataset MD5) is inherited unchanged from v4.

This is the script to run on the machine that has SciPy (the Cowork sandbox does not).

USAGE (same flags as v4):
    python kanon_weight_optimizer_v5.py
    python kanon_weight_optimizer_v5.py --k-folds 5 --n-restarts 20 --max-weight 0.50

OUTPUT: dados_reais/YYYY-MM-DD/otimizacao_v5/  (weights, cv_results, report, sensitivity)
Note: run AFTER the data repair (kanon_real_*.csv with the 6 re-collected profiles).
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kanon_weight_optimizer_v4 as v4


def _field_pctl_log(values):
    """Within-(this dataframe) empirical percentile of log(1+value), in (0,1]."""
    from scipy.stats import rankdata
    v = np.asarray(values, float)
    return rankdata(np.log1p(np.clip(v, 0, None))) / len(v)


def extract_components_corrected(df):
    """v5 components: C and J field-relative (percentile of log); A and S as in v4.
    run() calls this PER FIELD, so 'within df' == 'within field' (correct)."""
    C = _field_pctl_log(df['total_citations'].values)
    J = _field_pctl_log(df['journals'].values)
    A_pos = np.clip(1.0 - df['avg_author_position'].values / 20.0, 0, 1)
    avg_auth = np.maximum(df['avg_authors_per_paper'].values, 1.0)
    S_text = np.clip(df['complexity_keyword_score'].values, 0, 1)
    S_concepts = np.clip(df['complexity_concept_score'].values, 0, 1)
    return dict(C=C, A_pos=A_pos, avg_auth=avg_auth,
                S_text=S_text, S_concepts=S_concepts, J=J)


def main():
    import argparse
    from datetime import datetime
    ap = argparse.ArgumentParser(description='KANON v5 (corrected field-relative C/J, SLSQP)')
    ap.add_argument('--k-folds', type=int, default=5)
    ap.add_argument('--n-restarts', type=int, default=20)
    ap.add_argument('--max-weight', type=float, default=0.50)
    ap.add_argument('--data-dir', type=str, default=None)
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    # Monkey-patch v4 to use the corrected components, then reuse the whole pipeline.
    v4.extract_components = extract_components_corrected
    data_dir = args.data_dir or os.path.join(v4.SCRIPT_DIR, 'dados_reais')
    date_tag = datetime.now().strftime('%Y-%m-%d')
    output_dir = args.output or os.path.join(data_dir, date_tag, 'otimizacao_v5')
    print("[v5] CORRECTED components: C,J = within-field percentile of log value (non-degenerate)")
    v4.run(data_dir, args.k_folds, args.n_restarts, args.max_weight, output_dir)


if __name__ == '__main__':
    main()
