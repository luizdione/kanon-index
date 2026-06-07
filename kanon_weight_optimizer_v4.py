#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KANON-Index Weight Optimizer v4 — Arquitetura de 4 componentes (Fase 0)
=======================================================================
MUDANCA ARQUITETURAL CENTRAL (briefing v2.0, Problema 7):
  O KANON passa de 5 para 4 componentes. Os antigos M (complexidade
  metodologica) e R (custo experimental / h-index) sao FUNDIDOS em um
  unico componente S (Sofisticacao Experimental):

      F = (wC*C + wA*A + wS*S + wJ*J) * 100
      S = beta * S_text + (1 - beta) * S_concepts

  - S_text     = complexity_keyword_score  (marcadores curados no abstract)
  - S_concepts = complexity_concept_score  (classificacao OpenAlex; migrar p/ Topics)
  - beta       = peso relativo entre as duas fontes, APRENDIDO na otimizacao
  - alpha      = expoente de penalidade de hiperautoria, dentro de A (como na v3)

CORRECOES vs v3 (itens 3 e 4 da revisao 2026-05-20):
  - Otimizador trocado de L-BFGS-B para SLSQP: o L-BFGS-B ignora silenciosamente
    o argumento `constraints` no SciPy, entao a restricao soma(pesos)=1 NUNCA era
    aplicada. SLSQP suporta bounds + restricao de igualdade simultaneamente.
  - Teto de peso ajustado de 0.40 para 0.50.
  - Multi-start (n=20) com pontos de partida factiveis no simplex.
  - Reporta DP dos pesos entre folds E entre os top-5 restarts (estabilidade).
  - Analise de sensibilidade dos pesos (+-10% por componente -> dAUC).
  - Hash MD5 do dataset embutido na saida (reprodutibilidade, Problema 4).

Sem dependencia de scikit-learn: AUC-ROC e StratifiedKFold implementados
internamente (scipy apenas para minimize/rankdata/norm).

USO:
    python kanon_weight_optimizer_v4.py
    python kanon_weight_optimizer_v4.py --k-folds 5 --n-restarts 20 --max-weight 0.50
    python kanon_weight_optimizer_v4.py --data-dir dados_reais

SAIDA (dados_reais/YYYY-MM-DD/otimizacao_v4/):
    kanon_optimized_weights_v4.json   — pesos + beta + alpha por area, com IC/DP
    kanon_cv_results_v4.csv           — resultados por fold
    kanon_validation_report_v4.md     — relatorio de validacao
    kanon_sensitivity_v4.json         — analise de sensibilidade
"""

import argparse
import hashlib
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata, mannwhitneyu, norm

warnings.filterwarnings('ignore')

FIELDS = ['Medicine', 'Physics', 'Chemistry', 'Economics']
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ordem dos parametros do vetor theta otimizado
PARAM_NAMES = ['wC', 'wA', 'wS', 'wJ', 'beta', 'alpha']
WEIGHT_NAMES = ['wC', 'wA', 'wS', 'wJ']


# ============================================================================
# METRICAS INTERNAS (sem sklearn)
# ============================================================================
def roc_auc(y_true, scores):
    """AUC-ROC via estatistica de Mann-Whitney (trata empates por rank medio)."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    n1 = int(np.sum(y_true == 1))
    n0 = int(np.sum(y_true == 0))
    if n1 == 0 or n0 == 0:
        return float('nan')
    ranks = rankdata(scores)
    return (np.sum(ranks[y_true == 1]) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def stratified_kfold_indices(y, k, seed=42):
    """Folds estratificados: round-robin dentro de cada classe apos shuffle."""
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, sample in enumerate(idx):
            folds[i % k].append(int(sample))
    all_idx = set(range(len(y)))
    splits = []
    for i in range(k):
        test_idx = np.array(sorted(folds[i]), dtype=int)
        train_idx = np.array(sorted(all_idx - set(folds[i])), dtype=int)
        splits.append((train_idx, test_idx))
    return splits


def delong_test(y_true, scores_a, scores_b):
    """DeLong (Sun & Xu 2014) para comparar AUC_A vs AUC_B. Retorna (z, p)."""
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    n1 = int(np.sum(y_true == 1))
    n0 = int(np.sum(y_true == 0))
    if n1 < 2 or n0 < 2:
        return 0.0, 1.0

    pos_a, neg_a = scores_a[y_true == 1], scores_a[y_true == 0]
    pos_b, neg_b = scores_b[y_true == 1], scores_b[y_true == 0]

    def placement(pos_scores, neg_scores):
        v10 = np.array([np.mean(p > neg_scores) + 0.5 * np.mean(p == neg_scores)
                        for p in pos_scores])
        v01 = np.array([np.mean(pos_scores > n) + 0.5 * np.mean(pos_scores == n)
                        for n in neg_scores])
        return v10, v01

    v10_a, v01_a = placement(pos_a, neg_a)
    v10_b, v01_b = placement(pos_b, neg_b)
    s10 = np.cov(v10_a, v10_b)
    s01 = np.cov(v01_a, v01_b)
    var_diff = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n1 + \
               (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n0
    if var_diff <= 0:
        return 0.0, 1.0
    auc_a = roc_auc(y_true, scores_a)
    auc_b = roc_auc(y_true, scores_b)
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)


# ============================================================================
# DADOS E COMPONENTES
# ============================================================================
def load_real_data(data_dir):
    res_path = os.path.join(data_dir, 'kanon_real_researchers.csv')
    nob_path = os.path.join(data_dir, 'kanon_real_nobel.csv')
    for p in (res_path, nob_path):
        if not os.path.exists(p):
            print(f"ERRO: {p} nao encontrado. Rode o collector primeiro.")
            sys.exit(1)
    res = pd.read_csv(res_path)
    nob = pd.read_csv(nob_path)

    # Correcao h_index=0 (mantida para o baseline comparativo)
    for df in (res, nob):
        df['h_index_use'] = df['h_index'].copy()
        mask = df['h_index_use'] <= 0
        df.loc[mask, 'h_index_use'] = np.minimum(
            np.sqrt(df.loc[mask, 'total_citations'].clip(lower=0)).astype(int),
            df.loc[mask, 'total_publications'])

    res['is_nobel'] = 0
    nob['is_nobel'] = 1
    combined = pd.concat([res, nob], ignore_index=True)
    print(f"[OK] Dados: {len(res)} pesquisadores + {len(nob)} Nobel = {len(combined)}")
    return combined


def dataset_md5(combined):
    """Hash MD5 estavel do dataset (reprodutibilidade — Problema 4)."""
    cols = ['name', 'field', 'total_citations', 'avg_author_position',
            'avg_authors_per_paper', 'complexity_keyword_score',
            'complexity_concept_score', 'journals', 'is_nobel']
    cols = [c for c in cols if c in combined.columns]
    payload = combined[cols].sort_values('name').to_csv(index=False).encode('utf-8')
    return hashlib.md5(payload).hexdigest()[:12]


def extract_components(df):
    """Retorna as partes fixas dos componentes (independem dos pesos)."""
    C = np.minimum(1.0, df['total_citations'].values / 10000.0)
    A_pos = np.clip(1.0 - (df['avg_author_position'].values / 20.0), 0, 1)
    avg_auth = np.maximum(df['avg_authors_per_paper'].values, 1.0)
    S_text = np.clip(df['complexity_keyword_score'].values, 0, 1)
    S_concepts = np.clip(df['complexity_concept_score'].values, 0, 1)
    J = np.minimum(1.0, df['journals'].values / 50.0)
    return dict(C=C, A_pos=A_pos, avg_auth=avg_auth,
                S_text=S_text, S_concepts=S_concepts, J=J)


def calc_kanon(comp, theta):
    """KANON v4. theta = [wC, wA, wS, wJ, beta, alpha]."""
    wC, wA, wS, wJ, beta, alpha = theta
    A = np.clip(comp['A_pos'] / np.power(comp['avg_auth'], alpha), 0, 1)
    S = beta * comp['S_text'] + (1.0 - beta) * comp['S_concepts']
    return (wC * comp['C'] + wA * A + wS * S + wJ * comp['J']) * 100.0


def objective(theta, comp, y):
    auc = roc_auc(y, calc_kanon(comp, theta))
    return 1.0 if np.isnan(auc) else -auc


# ============================================================================
# OTIMIZACAO POR FOLD (SLSQP multi-start)
# ============================================================================
def optimize_fold(comp, y, n_restarts=20, seed=42, max_weight=0.50):
    # bounds: 4 pesos em [0.01, max_weight], beta em [0,1], alpha em [0.3,1.0]
    bounds = [(0.01, max_weight)] * 4 + [(0.0, 1.0), (0.3, 1.0)]
    constraints = [{'type': 'eq', 'fun': lambda t: float(np.sum(t[:4]) - 1.0)}]
    rng = np.random.RandomState(seed)

    solutions = []
    for _ in range(n_restarts):
        # Start factivel no simplex respeitando o teto por peso
        for _try in range(100):
            w = rng.dirichlet(np.ones(4))
            if w.max() <= max_weight:
                break
        else:
            w = np.full(4, 0.25)
        theta0 = np.concatenate([w, [rng.uniform(0, 1), rng.uniform(0.3, 1.0)]])
        try:
            res = minimize(objective, theta0, args=(comp, y), method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': 300, 'ftol': 1e-9})
        except Exception:
            continue
        if res.success and np.isfinite(res.fun):
            solutions.append((float(res.fun), np.asarray(res.x, dtype=float)))

    if not solutions:
        return None

    solutions.sort(key=lambda s: s[0])
    best_fun, best_x = solutions[0]

    # DP dos parametros entre os top-5 restarts (estabilidade da convergencia)
    top = np.array([x for _, x in solutions[:min(5, len(solutions))]])
    restart_std = top.std(axis=0)

    # Renormalizar pesos para somar exatamente 1 (guard numerico)
    w = best_x[:4].copy()
    w = w / w.sum()
    theta = np.concatenate([w, best_x[4:]])
    return {
        'theta': theta,
        'train_auc': -best_fun,
        'restart_std': restart_std,
        'n_converged': len(solutions),
    }


def evaluate_on_test(comp, y, theta, h_index):
    kanon = calc_kanon(comp, theta)
    out = {}
    if len(np.unique(y)) >= 2:
        out['auc_kanon'] = round(roc_auc(y, kanon), 4)
        out['auc_hindex'] = round(roc_auc(y, h_index), 4)
        z, p = delong_test(y, kanon, h_index)
        out['delong_z'] = round(z, 4)
        out['delong_p'] = round(p, 6)
    else:
        out.update(auc_kanon=np.nan, auc_hindex=np.nan, delong_z=np.nan, delong_p=np.nan)

    nob = y == 1
    if nob.sum() >= 2 and (~nob).sum() >= 2:
        kn, kr = kanon[nob], kanon[~nob]
        pooled = np.sqrt(((len(kr) - 1) * kr.std(ddof=1) ** 2 +
                          (len(kn) - 1) * kn.std(ddof=1) ** 2) / (len(kr) + len(kn) - 2))
        out['cohens_d_kanon'] = round((kn.mean() - kr.mean()) / pooled, 4) if pooled > 0 else 0.0
        _, p_mw = mannwhitneyu(kn, kr, alternative='greater')
        out['p_mannwhitney'] = round(float(p_mw), 6)
    else:
        out.update(cohens_d_kanon=np.nan, p_mannwhitney=np.nan)
    return out


# ============================================================================
# ANALISE DE SENSIBILIDADE (item 11 do briefing)
# ============================================================================
def sensitivity_analysis(comp, y, theta, delta=0.10, max_weight=0.50):
    base_auc = roc_auc(y, calc_kanon(comp, theta))
    results = {'base_auc': round(float(base_auc), 4), 'delta': delta, 'components': {}}
    for i, name in enumerate(WEIGHT_NAMES):
        w = theta[:4].copy()
        w[i] = min(w[i] * (1 + delta), max_weight)
        w = w / w.sum()
        t_pert = np.concatenate([w, theta[4:]])
        d = roc_auc(y, calc_kanon(comp, t_pert)) - base_auc
        results['components'][name] = round(float(d), 5)
    # beta isolado
    for sign, lab in ((+1, 'beta+'), (-1, 'beta-')):
        t_pert = theta.copy()
        t_pert[4] = float(np.clip(theta[4] * (1 + sign * delta), 0.0, 1.0))
        d = roc_auc(y, calc_kanon(comp, t_pert)) - base_auc
        results['components'][lab] = round(float(d), 5)
    return results


# ============================================================================
# PIPELINE PRINCIPAL
# ============================================================================
def run(data_dir, k_folds, n_restarts, max_weight, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    combined = load_real_data(data_dir)
    ds_hash = dataset_md5(combined)
    print(f"[HASH] dataset MD5: {ds_hash}")

    all_rows = []
    final = {}
    sens_all = {}

    for field in FIELDS:
        dff = combined[combined['field'] == field].copy().reset_index(drop=True)
        n_nob = int((dff['is_nobel'] == 1).sum())
        n_res = int((dff['is_nobel'] == 0).sum())
        if n_nob < k_folds:
            print(f"[SKIP] {field}: apenas {n_nob} Nobel (precisa >= {k_folds})")
            continue
        print(f"\n--- {field} (N={len(dff)}: {n_nob} Nobel + {n_res} pesq) ---")

        y = dff['is_nobel'].values
        comp_all = extract_components(dff)
        h_all = dff['h_index_use'].values
        splits = stratified_kfold_indices(y, k_folds, seed=42)

        fold_thetas, fold_rows, restart_stds = [], [], []
        for fi, (tr, te) in enumerate(splits):
            comp_tr = {kk: vv[tr] for kk, vv in comp_all.items()}
            comp_te = {kk: vv[te] for kk, vv in comp_all.items()}
            opt = optimize_fold(comp_tr, y[tr], n_restarts=n_restarts,
                                seed=1000 * fi + 42, max_weight=max_weight)
            if opt is None:
                print(f"  Fold {fi+1}: FALHOU")
                continue
            theta = opt['theta']
            metrics = evaluate_on_test(comp_te, y[te], theta, h_all[te])
            fold_thetas.append(theta)
            restart_stds.append(opt['restart_std'])
            row = {'field': field, 'fold': fi + 1,
                   'n_train': len(tr), 'n_test': len(te),
                   'train_auc': round(opt['train_auc'], 4),
                   'n_converged': opt['n_converged']}
            for j, nm in enumerate(PARAM_NAMES):
                row[nm] = round(float(theta[j]), 4)
            row.update(metrics)
            fold_rows.append(row)
            all_rows.append(row)
            print(f"  Fold {fi+1}: AUC(K)={metrics.get('auc_kanon')} "
                  f"AUC(h)={metrics.get('auc_hindex')} DeLong p={metrics.get('delong_p')} "
                  f"beta={row['beta']} (restarts conv={opt['n_converged']}/{n_restarts})")

        if not fold_thetas:
            continue

        thetas = np.array(fold_thetas)
        restart_stds = np.array(restart_stds)
        med = np.median(thetas, axis=0)
        # renormalizar pesos medianos
        med_w = med[:4] / med[:4].sum()
        med = np.concatenate([med_w, med[4:]])

        fw = {}
        for j, nm in enumerate(PARAM_NAMES):
            fw[nm] = round(float(med[j]), 4)
            fw[f'{nm}_std_folds'] = round(float(thetas[:, j].std()), 4)
            fw[f'{nm}_std_restarts'] = round(float(restart_stds[:, j].mean()), 4)
            fw[f'{nm}_ci_lo'] = round(float(np.percentile(thetas[:, j], 2.5)), 4)
            fw[f'{nm}_ci_hi'] = round(float(np.percentile(thetas[:, j], 97.5)), 4)

        auc_k = [r['auc_kanon'] for r in fold_rows if not pd.isna(r.get('auc_kanon'))]
        auc_h = [r['auc_hindex'] for r in fold_rows if not pd.isna(r.get('auc_hindex'))]
        dl_p = [r['delong_p'] for r in fold_rows if not pd.isna(r.get('delong_p'))]
        dk = [r['cohens_d_kanon'] for r in fold_rows if not pd.isna(r.get('cohens_d_kanon'))]
        fw['auc_kanon_mean'] = round(float(np.mean(auc_k)), 4) if auc_k else None
        fw['auc_kanon_std'] = round(float(np.std(auc_k)), 4) if auc_k else None
        fw['auc_hindex_mean'] = round(float(np.mean(auc_h)), 4) if auc_h else None
        fw['auc_hindex_std'] = round(float(np.std(auc_h)), 4) if auc_h else None
        fw['delong_p_median'] = round(float(np.median(dl_p)), 6) if dl_p else None
        fw['cohens_d_kanon_mean'] = round(float(np.mean(dk)), 4) if dk else None
        wins = sum(1 for a, b in zip(auc_k, auc_h) if a > b)
        fw['kanon_wins_folds'] = wins
        fw['total_folds'] = len(auc_k)
        fw['kanon_superior'] = wins > len(auc_k) / 2
        final[field] = fw

        # Sensibilidade usando os pesos finais sobre toda a area
        sens_all[field] = sensitivity_analysis(comp_all, y, med, max_weight=max_weight)

        print(f"  RESULTADO {field}: wC={fw['wC']} wA={fw['wA']} wS={fw['wS']} "
              f"wJ={fw['wJ']} beta={fw['beta']} alpha={fw['alpha']}")
        print(f"    AUC(K)={fw['auc_kanon_mean']}+-{fw['auc_kanon_std']} | "
              f"AUC(h)={fw['auc_hindex_mean']}+-{fw['auc_hindex_std']} | "
              f"KANON wins {wins}/{len(auc_k)}")

    # ---- Salvar ----
    cv_df = pd.DataFrame(all_rows)
    cv_path = os.path.join(output_dir, 'kanon_cv_results_v4.csv')
    cv_df.to_csv(cv_path, index=False)

    out_json = dict(final)
    out_json['_meta'] = {
        'version': 'v4',
        'architecture': '4 componentes (C, A, S, J); S = beta*S_text + (1-beta)*S_concepts',
        'optimizer': f'SLSQP multi-start (n={n_restarts}), Stratified {k_folds}-Fold CV',
        'constraint': f'sum(weights)=1, weight in [0.01, {max_weight}], beta in [0,1]',
        's_text_source': 'complexity_keyword_score',
        's_concepts_source': 'complexity_concept_score (migrar p/ Topics/subfield baseline)',
        'criterion': 'AUC-ROC (Nobel vs Researcher)',
        'validation': 'DeLong test vs h-index',
        'dataset_md5': ds_hash,
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'k_folds': k_folds, 'n_restarts': n_restarts, 'max_weight': max_weight,
    }
    json_path = os.path.join(output_dir, 'kanon_optimized_weights_v4.json')
    json.dump(out_json, open(json_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    json.dump(out_json, open(os.path.join(data_dir, 'kanon_optimized_weights_v4.json'),
                             'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    sens_path = os.path.join(output_dir, 'kanon_sensitivity_v4.json')
    json.dump(sens_all, open(sens_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    write_report(final, sens_all, output_dir, k_folds, n_restarts, max_weight, ds_hash, combined)

    print(f"\n[OK] {json_path}")
    print(f"[OK] {cv_path}")
    print(f"[OK] {sens_path}")
    return final, sens_all


def write_report(final, sens_all, output_dir, k_folds, n_restarts, max_weight, ds_hash, combined):
    L = []
    L.append("# KANON-Index Validation Report v4 (Fase 0 — 4 componentes)")
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  dataset MD5: `{ds_hash}`")
    L.append("")
    L.append("## Arquitetura")
    L.append("- Formula: `F = (wC*C + wA*A + wS*S + wJ*J) * 100`")
    L.append("- `S = beta*S_text + (1-beta)*S_concepts` (M e R fundidos em S)")
    L.append(f"- Otimizador: **SLSQP** multi-start (n={n_restarts}), Stratified {k_folds}-Fold CV")
    L.append(f"- Restricoes: soma(pesos)=1, peso in [0.01, {max_weight}], beta in [0,1]")
    L.append("")
    L.append("## Pesos otimizados (mediana entre folds; DP entre folds / restarts)")
    L.append("")
    L.append("| Area | wC | wA | wS | wJ | beta | alpha |")
    L.append("|------|----|----|----|----|------|-------|")
    for f in FIELDS:
        if f in final:
            w = final[f]
            L.append(f"| {f} | {w['wC']}±{w['wC_std_folds']} | {w['wA']}±{w['wA_std_folds']} | "
                     f"{w['wS']}±{w['wS_std_folds']} | {w['wJ']}±{w['wJ_std_folds']} | "
                     f"{w['beta']}±{w['beta_std_folds']} | {w['alpha']}±{w['alpha_std_folds']} |")
    L.append("")
    L.append("## Desempenho out-of-sample (folds de teste)")
    L.append("")
    L.append("| Area | AUC(KANON) | AUC(h-index) | DeLong p | d(KANON) | Vencedor |")
    L.append("|------|-----------|--------------|----------|----------|----------|")
    for f in FIELDS:
        if f in final:
            w = final[f]
            win = "**KANON**" if w.get('kanon_superior') else "h-index"
            ak = f"{w['auc_kanon_mean']}±{w['auc_kanon_std']}" if w['auc_kanon_mean'] is not None else "N/A"
            ah = f"{w['auc_hindex_mean']}±{w['auc_hindex_std']}" if w['auc_hindex_mean'] is not None else "N/A"
            L.append(f"| {f} | {ak} | {ah} | {w.get('delong_p_median')} | "
                     f"{w.get('cohens_d_kanon_mean')} | {win} |")
    L.append("")
    L.append("## Analise de sensibilidade (dAUC ao perturbar cada peso +10%)")
    L.append("")
    L.append("| Area | wC | wA | wS | wJ | beta+ | beta- |")
    L.append("|------|----|----|----|----|-------|-------|")
    for f in FIELDS:
        if f in sens_all:
            c = sens_all[f]['components']
            L.append(f"| {f} | {c.get('wC')} | {c.get('wA')} | {c.get('wS')} | "
                     f"{c.get('wJ')} | {c.get('beta+')} | {c.get('beta-')} |")
    L.append("")
    L.append("## Notas")
    L.append("- `beta` alto => S depende mais de S_text (marcadores no abstract); "
             "`beta` baixo => depende mais de S_concepts (classificacao OpenAlex).")
    L.append("- S_concepts atual vem de `complexity_concept_score` (concepts+topics). "
             "Item 1 da revisao: migrar para Topics/subfield baseline e RE-COLETAR para efetivar.")
    L.append("- Comparacao inter-areas requer normalizacao intra-area (Problema 1) — pendente.")
    path = os.path.join(output_dir, 'kanon_validation_report_v4.md')
    open(path, 'w', encoding='utf-8').write("\n".join(L))
    print(f"[OK] {path}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='KANON-Index Weight Optimizer v4 (4 componentes, SLSQP)')
    ap.add_argument('--k-folds', type=int, default=5)
    ap.add_argument('--n-restarts', type=int, default=20)
    ap.add_argument('--max-weight', type=float, default=0.50)
    ap.add_argument('--data-dir', type=str, default=None)
    ap.add_argument('--output', type=str, default=None)
    args = ap.parse_args()

    data_dir = args.data_dir or os.path.join(SCRIPT_DIR, 'dados_reais')
    date_tag = datetime.now().strftime('%Y-%m-%d')
    output_dir = args.output or os.path.join(data_dir, date_tag, 'otimizacao_v4')
    run(data_dir, args.k_folds, args.n_restarts, args.max_weight, output_dir)
