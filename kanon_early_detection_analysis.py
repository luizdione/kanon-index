#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KANON-Index — Analise de DETECCAO PRECOCE
=========================================
Le dados_reais/kanon_early_real.csv (1 linha por autor x horizonte, point-in-time)
e os pesos FIXOS do KANON honesto (kanon_optimized_weights_v4.json). Para cada area
e horizonte (T0, T-5, T-10), aplica os pesos fixos (NAO re-otimiza: e um teste de
rastreio, nao de ajuste) e mede:

  - AUC(KANON) vs AUC(h-index point-in-time)  [laureado vs controle]
  - AUC de cada componente isolado: A (autoria), S (sofisticacao), C (citacoes)
  - AUC de um KANON "content-only" (wC=0, renormalizado) -> sinal sem citacoes
  - Percentil do laureado mediano entre os controles ("parecia mediano em T-10?")
  - DeLong KANON vs h; flag de confiabilidade do point-in-time (cby_min_year)

Saida: dados_reais/<data>/early_analysis/
  - early_detection_table.csv         (area x horizonte x metricas)
  - fig_early_auc_by_horizon.png      (KANON / h / content-only por area)
  - fig_early_components.png          (A / S / C por horizonte)
  - fig_early_percentile.png          (percentil do laureado mediano)
  - early_detection_report.md

USO: python kanon_early_detection_analysis.py
"""

import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kanon_weight_optimizer_v4 as kx  # reusa roc_auc, calc_kanon, extract_components, delong_test

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ['Medicine', 'Physics', 'Chemistry', 'Economics']


def _theta(weights, field):
    w = weights[field]
    return [w['wC'], w['wA'], w['wS'], w['wJ'], w['beta'], w['alpha']]


def _content_only_theta(theta):
    """Zera wC e renormaliza wA,wS,wJ (mantem beta, alpha) -> KANON sem citacoes."""
    wC, wA, wS, wJ, beta, alpha = theta
    s = wA + wS + wJ
    if s <= 0:
        return [0, 1 / 3, 1 / 3, 1 / 3, beta, alpha]
    return [0.0, wA / s, wS / s, wJ / s, beta, alpha]


def _component_scores(comp, theta):
    """A, S e C isolados (mesmas formulas do otimizador)."""
    _, _, _, _, beta, alpha = theta
    A = np.clip(comp['A_pos'] / np.power(comp['avg_auth'], alpha), 0, 1)
    S = beta * comp['S_text'] + (1.0 - beta) * comp['S_concepts']
    return A, S, comp['C']


def _percentile_of_median_laureate(score, y):
    """Percentil (0-100) do laureado MEDIANO em relacao aos controles."""
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    med = np.median(pos)
    return round(100.0 * np.mean(neg <= med), 1)


def run(data_dir):
    early_path = os.path.join(data_dir, 'kanon_early_real.csv')
    wpath = os.path.join(data_dir, 'kanon_optimized_weights_v4.json')
    for p in (early_path, wpath):
        if not os.path.exists(p):
            print(f"ERRO: {p} nao encontrado."); sys.exit(1)
    df = pd.read_csv(early_path)
    weights = json.load(open(wpath, encoding='utf-8'))
    horizons = sorted(df['horizon'].unique())

    rows = []
    for field in FIELDS:
        if field not in weights:
            continue
        theta = _theta(weights, field)
        theta_co = _content_only_theta(theta)
        for k in horizons:
            sub = df[(df['field'] == field) & (df['horizon'] == k)].reset_index(drop=True)
            if sub['is_nobel'].nunique() < 2 or len(sub) < 8:
                continue
            y = sub['is_nobel'].values
            comp = kx.extract_components(sub)
            kanon = kx.calc_kanon(comp, theta)
            kanon_co = kx.calc_kanon(comp, theta_co)
            A, S, C = _component_scores(comp, theta)
            h = sub['h_index'].values.astype(float)
            z, p = kx.delong_test(y, kanon, h)
            cutoff = int(sub['cutoff_year'].median())
            cby_min = pd.to_numeric(sub['cby_min_year'], errors='coerce').median()
            rows.append({
                'field': field, 'horizon': k, 'cutoff_year': cutoff,
                'n_nobel': int((y == 1).sum()), 'n_control': int((y == 0).sum()),
                'AUC_KANON': round(kx.roc_auc(y, kanon), 4),
                'AUC_hindex': round(kx.roc_auc(y, h), 4),
                'AUC_content_only': round(kx.roc_auc(y, kanon_co), 4),
                'AUC_A': round(kx.roc_auc(y, A), 4),
                'AUC_S': round(kx.roc_auc(y, S), 4),
                'AUC_C': round(kx.roc_auc(y, C), 4),
                'delong_p': round(p, 4),
                'pctl_med_laureate_KANON': _percentile_of_median_laureate(kanon, y),
                'pctl_med_laureate_h': _percentile_of_median_laureate(h, y),
                'cby_min_year_median': None if pd.isna(cby_min) else int(cby_min),
                'cit_pit_reliable': '' if pd.isna(cby_min) else ('sim' if cutoff >= cby_min else 'NAO (subestima)'),
            })

    out = pd.DataFrame(rows)
    date_tag = datetime.now().strftime('%Y-%m-%d')
    odir = os.path.join(data_dir, date_tag, 'early_analysis')
    os.makedirs(odir, exist_ok=True)
    out.to_csv(os.path.join(odir, 'early_detection_table.csv'), index=False)
    print(out.to_string(index=False))

    _figures(out, odir, horizons)
    _report(out, odir, horizons)
    print(f"\n[OK] {odir}")
    return out


def _figures(out, odir, horizons):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    xs = sorted(horizons, reverse=True)              # 10,5,0 -> esquerda=cedo
    xlabels = [f"T−{k}" if k else "T0" for k in xs]

    def line(ax, sub, col, label, **kw):
        ys = [sub[sub['horizon'] == k][col].mean() if (sub['horizon'] == k).any() else np.nan for k in xs]
        ax.plot(range(len(xs)), ys, marker='o', label=label, **kw)

    # FIG 1: KANON vs h vs content-only
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, field in zip(axes.ravel(), FIELDS):
        sub = out[out['field'] == field]
        if sub.empty:
            ax.set_visible(False); continue
        line(ax, sub, 'AUC_KANON', 'KANON', color='#27ae60', lw=2)
        line(ax, sub, 'AUC_hindex', 'h-index (point-in-time)', color='#7f8c8d')
        line(ax, sub, 'AUC_content_only', 'KANON sem citações (A+S+J)', color='#2980b9', ls='--')
        ax.axhline(0.5, ls=':', c='gray'); ax.set_ylim(0.3, 1.0)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xlabels)
        ax.set_title(field); ax.set_ylabel('AUC'); ax.legend(fontsize=8)
    fig.suptitle('Detecção precoce: discriminação por horizonte (cedo → reconhecimento)', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(odir, 'fig_early_auc_by_horizon.png'), dpi=150); plt.close()

    # FIG 2: componentes A / S / C
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, field in zip(axes.ravel(), FIELDS):
        sub = out[out['field'] == field]
        if sub.empty:
            ax.set_visible(False); continue
        line(ax, sub, 'AUC_A', 'A (autoria)', color='#8e44ad')
        line(ax, sub, 'AUC_S', 'S (sofisticação)', color='#16a085')
        line(ax, sub, 'AUC_C', 'C (citações)', color='#c0392b', ls='--')
        ax.axhline(0.5, ls=':', c='gray'); ax.set_ylim(0.3, 1.0)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xlabels)
        ax.set_title(field); ax.set_ylabel('AUC'); ax.legend(fontsize=8)
    fig.suptitle('Componentes ao longo do tempo: A e S existem cedo; C só perto do reconhecimento', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(odir, 'fig_early_components.png'), dpi=150); plt.close()

    # FIG 3: percentil do laureado mediano
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, field in zip(axes.ravel(), FIELDS):
        sub = out[out['field'] == field]
        if sub.empty:
            ax.set_visible(False); continue
        line(ax, sub, 'pctl_med_laureate_KANON', 'percentil via KANON', color='#27ae60', lw=2)
        line(ax, sub, 'pctl_med_laureate_h', 'percentil via h-index', color='#7f8c8d')
        ax.axhline(50, ls=':', c='gray'); ax.set_ylim(0, 100)
        ax.set_xticks(range(len(xs))); ax.set_xticklabels(xlabels)
        ax.set_title(field); ax.set_ylabel('percentil do laureado mediano'); ax.legend(fontsize=8)
    fig.suptitle('"Parecia mediano cedo?" — posição do laureado mediano entre os pares contemporâneos', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(odir, 'fig_early_percentile.png'), dpi=150); plt.close()


def _report(out, odir, horizons):
    L = ["# Detecção precoce — relatório", f"Gerado: {datetime.now():%Y-%m-%d %H:%M}", ""]
    L.append("AUC (laureado vs controle contemporâneo), pesos FIXOS do KANON honesto por horizonte.")
    L.append("Horizonte = anos antes do prêmio. `cit_pit_reliable=NAO` sinaliza horizontes onde "
             "o counts_by_year do OpenAlex não cobre o corte (C e h subestimados).\n")
    cols = ['field', 'horizon', 'cutoff_year', 'AUC_KANON', 'AUC_hindex', 'AUC_content_only',
            'AUC_A', 'AUC_S', 'AUC_C', 'delong_p', 'pctl_med_laureate_KANON',
            'pctl_med_laureate_h', 'cit_pit_reliable']
    cols = [c for c in cols if c in out.columns]
    L.append("| " + " | ".join(cols) + " |")
    L.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in out.iterrows():
        L.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    L.append("\n## Leitura sugerida")
    L.append("- Se **AUC_content_only** e **AUC_S/AUC_A** se mantêm > 0.5 em T-10 enquanto "
             "**AUC_C/AUC_hindex** caem para ~0.5, há sinal de detecção precoce vindo de autoria+sofisticação.")
    L.append("- Se **pctl_med_laureate_KANON** em T-10 for alto (ex.: > 70) mas via h-index for ~50, "
             "o KANON enxerga o futuro laureado que ainda 'parecia mediano' em citações.")
    open(os.path.join(odir, 'early_detection_report.md'), 'w', encoding='utf-8').write("\n".join(L))


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', default=os.path.join(HERE, 'dados_reais'))
    run(ap.parse_args().data_dir)
