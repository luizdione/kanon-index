#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KANON audit library (2026-05-27) — pure numpy/pandas, NO scipy.
Centraliza: metricas (AUC, DeLong), componentes (as-implemented e CORRIGIDOS
field-relative), e um otimizador vetorizado de pesos (multi-start no simplex).

Motivo: auditoria revelou que os componentes C (citacoes/10000) e J (journals/50)
saturam em 1.0 para 91-99% dos perfis -> nao discriminam e nao correspondem as
equacoes do paper. Esta lib implementa versoes corrigidas, field-relative.
"""
import math
import numpy as np
import pandas as pd

FIELDS = ['Medicine', 'Physics', 'Chemistry', 'Economics']


# ----------------------------- metricas -----------------------------
def _avg_ranks(s):
    s = np.asarray(s, float)
    order = np.argsort(s, kind='mergesort')
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def roc_auc(y, s):
    y = np.asarray(y)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = _avg_ranks(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def delong_test(y, sa, sb):
    """DeLong (Sun & Xu) — retorna (z, p) para AUC(sa) vs AUC(sb)."""
    y = np.asarray(y); sa = np.asarray(sa, float); sb = np.asarray(sb, float)
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    if n1 < 2 or n0 < 2:
        return 0.0, 1.0
    pa, na = sa[y == 1], sa[y == 0]
    pb, nb = sb[y == 1], sb[y == 0]

    def place(pos, neg):
        v10 = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
        v01 = np.array([np.mean(pos > n) + 0.5 * np.mean(pos == n) for n in neg])
        return v10, v01
    v10a, v01a = place(pa, na)
    v10b, v01b = place(pb, nb)
    s10 = np.cov(v10a, v10b); s01 = np.cov(v01a, v01b)
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n1 + \
          (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n0
    if var <= 0:
        return 0.0, 1.0
    z = (roc_auc(y, sa) - roc_auc(y, sb)) / math.sqrt(var)
    return float(z), float(2 * (1 - _ncdf(abs(z))))


# ----------------------------- dados -----------------------------
def load_combined(data_dir='dados_reais'):
    import os
    r = pd.read_csv(os.path.join(data_dir, 'kanon_real_researchers.csv')); r['is_nobel'] = 0
    n = pd.read_csv(os.path.join(data_dir, 'kanon_real_nobel.csv')); n['is_nobel'] = 1
    comb = pd.concat([r, n], ignore_index=True)
    # h_index_use: corrige h<=0 (raro)
    comb['h_index_use'] = comb['h_index'].clip(lower=0)
    return comb


# ----------------------------- componentes -----------------------------
def comp_as_implemented(df):
    """Exatamente como kanon_weight_optimizer_v4.calc_kanon usa."""
    C = np.minimum(1.0, df['total_citations'].values / 10000.0)
    A_pos = np.clip(1.0 - df['avg_author_position'].values / 20.0, 0, 1)
    avg_auth = np.maximum(df['avg_authors_per_paper'].values, 1.0)
    S_text = np.clip(df['complexity_keyword_score'].values, 0, 1)
    S_concepts = np.clip(df['complexity_concept_score'].values, 0, 1)
    J = np.minimum(1.0, df['journals'].values / 50.0)
    return dict(C=C, A_pos=A_pos, avg_auth=avg_auth, S_text=S_text,
                S_concepts=S_concepts, J=J)


def _field_pctl_log(values, fields):
    """Percentil empirico (0-1) de log(1+x) DENTRO de cada campo -> field-relative, sem saturacao."""
    values = np.asarray(values, float)
    out = np.zeros(len(values))
    lv = np.log1p(np.clip(values, 0, None))
    for f in np.unique(fields):
        m = fields == f
        out[m] = _avg_ranks(lv[m]) / m.sum()  # (1..n)/n -> (0,1]
    return out


def comp_corrected(df):
    """Componentes CORRIGIDOS, field-relative (resolve a saturacao de C e J).
    - C: percentil intra-campo de log(1+citacoes)  (substitui cit/10000)
    - J: percentil intra-campo de log(1+n_journals) (proxy de diversidade de veiculo;
         NAO e fator de impacto - limitacao documentada)
    - A, S: inalterados (ja eram nao-degenerados)."""
    fields = df['field'].values
    C = _field_pctl_log(df['total_citations'].values, fields)
    J = _field_pctl_log(df['journals'].values, fields)
    A_pos = np.clip(1.0 - df['avg_author_position'].values / 20.0, 0, 1)
    avg_auth = np.maximum(df['avg_authors_per_paper'].values, 1.0)
    S_text = np.clip(df['complexity_keyword_score'].values, 0, 1)
    S_concepts = np.clip(df['complexity_concept_score'].values, 0, 1)
    return dict(C=C, A_pos=A_pos, avg_auth=avg_auth, S_text=S_text,
                S_concepts=S_concepts, J=J)


def kanon_score(comp, theta):
    """theta = [wC, wA, wS, wJ, beta, alpha]."""
    wC, wA, wS, wJ, beta, alpha = theta
    A = np.clip(comp['A_pos'] / np.power(comp['avg_auth'], alpha), 0, 1)
    S = beta * comp['S_text'] + (1.0 - beta) * comp['S_concepts']
    return (wC * comp['C'] + wA * A + wS * S + wJ * comp['J']) * 100.0


def components_ASCJ(comp, theta):
    """Retorna C, A, S, J finais (0-1) dado theta (para tabelas/figuras)."""
    wC, wA, wS, wJ, beta, alpha = theta
    A = np.clip(comp['A_pos'] / np.power(comp['avg_auth'], alpha), 0, 1)
    S = beta * comp['S_text'] + (1.0 - beta) * comp['S_concepts']
    return comp['C'], A, S, comp['J']


# ----------------------------- otimizador vetorizado -----------------------------
def stratified_folds(y, k=5, seed=42):
    y = np.asarray(y); rng = np.random.RandomState(seed)
    folds = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]; rng.shuffle(idx)
        for i, s in enumerate(idx):
            folds[i % k].append(int(s))
    alli = set(range(len(y))); out = []
    for i in range(k):
        te = np.array(sorted(folds[i])); tr = np.array(sorted(alli - set(folds[i])))
        out.append((tr, te))
    return out


def _auc_matrix(y, scores_mat):
    """AUC para cada coluna de scores_mat [n x m]. Retorna vetor [m]."""
    y = np.asarray(y); n1 = (y == 1).sum(); n0 = (y == 0).sum()
    m = scores_mat.shape[1]
    aucs = np.empty(m)
    pos = y == 1
    for j in range(m):
        aucs[j] = (_avg_ranks(scores_mat[:, j])[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return aucs


def optimize_field(comp, y, n_weights=3000, betas=None, alphas=None,
                   max_w=0.50, seed=42):
    """Multi-start vetorizado: amostra n_weights pesos no simplex (Dirichlet, teto max_w)
    x grade de (beta, alpha). Maximiza AUC no treino. Retorna melhor theta + train_auc."""
    if betas is None:
        betas = np.linspace(0.05, 0.95, 7)
    if alphas is None:
        alphas = np.linspace(0.30, 1.0, 8)
    rng = np.random.RandomState(seed)
    W = rng.dirichlet(np.ones(4), size=n_weights * 3)
    W = W[(W.max(axis=1) <= max_w)][:n_weights]
    if len(W) < 50:
        W = np.full((1, 4), 0.25)
    best = (-1.0, None)
    Cc = comp['C']; J = comp['J']; A_pos = comp['A_pos']; avg = comp['avg_auth']
    St = comp['S_text']; Sc = comp['S_concepts']
    for alpha in alphas:
        A = np.clip(A_pos / np.power(avg, alpha), 0, 1)
        for beta in betas:
            S = beta * St + (1 - beta) * Sc
            X = np.column_stack([Cc, A, S, J])          # n x 4
            scores = X @ W.T                            # n x m
            aucs = _auc_matrix(y, scores)
            jbest = int(np.nanargmax(aucs))
            if aucs[jbest] > best[0]:
                w = W[jbest]
                best = (float(aucs[jbest]),
                        np.array([w[0], w[1], w[2], w[3], float(beta), float(alpha)]))
    return best[1], best[0]


def cv_evaluate(comp, y, h, theta_per_fold_fn, k=5, seed=42):
    """Roda CV: em cada fold otimiza no treino e avalia no teste. Retorna dict."""
    folds = stratified_folds(y, k, seed)
    rows = []
    thetas = []
    for fi, (tr, te) in enumerate(folds):
        comp_tr = {kk: vv[tr] for kk, vv in comp.items()}
        comp_te = {kk: vv[te] for kk, vv in comp.items()}
        theta, tr_auc = theta_per_fold_fn(comp_tr, y[tr], fi)
        thetas.append(theta)
        sc = kanon_score(comp_te, theta)
        ak = roc_auc(y[te], sc); ah = roc_auc(y[te], h[te])
        z, p = delong_test(y[te], sc, h[te])
        rows.append(dict(fold=fi + 1, auc_kanon=ak, auc_h=ah, delong_p=p,
                         **{n: theta[i] for i, n in enumerate(['wC', 'wA', 'wS', 'wJ', 'beta', 'alpha'])}))
    df = pd.DataFrame(rows)
    med = np.median(np.array(thetas), axis=0)
    med[:4] = med[:4] / med[:4].sum()
    return df, med
