# KANON-Index

**Beyond the h-index: a multidimensional and editable indicator that credits
authorship structure and methodological sophistication.**

Author: Luiz Dione Barbosa de Melo — Instituto Federal de Educação, Ciência e
Tecnologia do Rio de Janeiro (IFRJ), Brazil
· ORCID [0000-0003-2445-9943](https://orcid.org/0000-0003-2445-9943)

This repository contains the **method, data and reproduction scripts** for the
KANON-Index paper (version 9, *field-relative* C/J pipeline). It is intended to let
reviewers and readers **reproduce the published numbers, tables and figures**. It does
**not** contain manuscript-writing files (LaTeX, drafts, figures-for-typesetting),
which are kept outside this repository.

License: **MIT** (see `LICENSE`). All bibliometric data derive from the open
[OpenAlex](https://openalex.org) API.

---

## The index

For researcher *i*:

```
KANON_i = 100 · ( w_C·C_i + w_A·A_i + w_S·S_i + w_J·J_i ),   Σ w_k = 1,  0.01 ≤ w_k ≤ 0.50
```

| Comp. | Meaning | Definition (canonical, field-relative) |
|------|---------|----------------------------------------|
| **C** | Citation impact | within-field percentile of `log(1+citations)` |
| **A** | Authorship contribution | `clip(P_pos / N̄^α, 0, 1)`, `P_pos = clip(1 − p̄/20, 0, 1)`; penalises hyperauthorship via `1/N̄^α`, `α ∈ [0.3, 1]` |
| **S** | Methodological sophistication | `β·S_text + (1−β)·S_concepts` (curated method markers + MeSH "E" tree + Topics baseline) |
| **J** | Venue breadth | within-field percentile of `log(1+distinct journals)` |

Per-field weights, `α` and `β` are **learned by cross-validation** (not fixed) and are
fully editable. Core formulas live in **`kanon_audit_lib.py`**.

---

## Reproducing the published results

Requires Python 3.11+ and `pip install -r requirements.txt`. Set your contact e-mail for
the OpenAlex polite pool (no personal e-mail is hard-coded):

```bash
export OPENALEX_MAILTO="you@example.org"      # Windows: set OPENALEX_MAILTO=you@example.org
```

The repository ships the collected datasets, so the headline numbers can be reproduced
**without re-collecting** from OpenAlex:

```bash
python _qa_verify_v9.py             # re-derives the v9 headline numbers from real data
python _qa_laureate_vs_elite_auc.py # Table: laureate-vs-elite AUC (h, g, h_I, h_m, KANON, A)
python _qa_tabSv_and_naivepool.py   # Table: S-component audit + naive-pool reconciliation
python _qa_altidx_recompute.py      # alt-index hyperauthorship sensitivity
```

Full pipeline from scratch (hours; depends on the OpenAlex API):

```bash
python kanon_full_collector_v3.py        # 1. collect (imports kanon_full_collector_v2)
python build_researchers_benchmark.py    # 2. field-matched negative class
python kanon_weight_optimizer_v4.py      # 3. weight optimisation (4 components, SLSQP/CV)
python _reopt_fieldrel_v6.py             # 4. field-relative re-optimisation (canonical v9)
```

---

## What is in this repository

```
kanon_audit_lib.py            core formulas (components, KANON, ROC-AUC, DeLong, optimiser)
kanon_full_collector_v2/v3.py OpenAlex collection (authorship, MeSH, Topics, citations)
kanon_mesh_sophistication.py  MeSH "E"-tree sophistication module
build_researchers_benchmark.py field-matched negative class
kanon_weight_optimizer_v4/v5.py weight optimisation
_reopt_fieldrel_v6.py, _reopt_one.py  field-relative re-optimisation -> _reopt_parts/
kanon_early_detection_*.py    point-in-time (early-detection) analysis
_altindices_*.py              Egghe g, Batista h_I, Schreiber h_m
_qa_*.py                      verification scripts that reproduce the published numbers
test_kanon_optimizer_v4.py    unit tests
_per_researcher_full.csv      per-researcher decomposition (source of Supplementary Table S1)
kanon_per_researcher_indices.xlsx  Supplementary Table S1
_fairness_correlation_stats.csv    fairness correlation table (tab:corr)
_reopt_parts/                 cross-validated per-field weights/AUC (tab:matched)
dados_reais/                  collected datasets + MeSH/Topics caches for reproduction
figuras/                      the 13 figures used in the paper
exampleNOBEL_orcids.csv, researchers_benchmark_v2.csv   input cohorts (public ORCIDs)
```

---

## Interactive calculator (`calculadora/`)

A downloadable, runnable calculator is provided in [`calculadora/`](calculadora/): look up a
researcher via OpenAlex, see the C/A/S/J breakdown, and **edit the weights and the
hyperauthorship penalty**. Open `calculadora/kanon_index.html` in a browser, or run the Flask
app (`python calculadora/app.py`). Note: the calculator uses the standalone (saturating)
formulation for client-side computation; the **canonical v9 field-relative method** and exact
paper weights are the scripts in this repository root. See `calculadora/README.md`.

## Citation

> Melo, L. D. B. (2026). *Beyond the h-index: the KANON-Index, a multidimensional and
> editable indicator that credits authorship structure and methodological sophistication.*
> Preprint (v9).

## Responsible use

KANON is a complementary, auditable descriptor — **not** a predictive tool and **not** a
sole decision criterion. Weights are editable and should be published with any deployment;
the index is intended to support, not replace, expert judgement.
