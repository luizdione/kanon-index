# KANON-Index — Interactive Calculator

A small, downloadable calculator to **explore the KANON-Index interactively**: look up a
researcher (via the open OpenAlex API), see the component breakdown (C, A, S, J), and
**edit the weights and the hyperauthorship penalty** to see how the score changes.

> **Important — which method does this calculator implement?**
> This tool uses the **standalone (saturating) formulation** of the components, because it
> must be computable in the browser for a *single* researcher. The **canonical method of the
> paper (v9)** uses *field-relative* C and J (the within-field percentile of `log(1+x)`),
> which requires the full reference cohort and therefore lives in the main repository scripts
> (`kanon_audit_lib.py`, `_reopt_fieldrel_v6.py`, the `_qa_*.py` verifiers), not here.
> The default weights below are the cross-validated **v4 (4-component)** weights; they are a
> reasonable starting point and are **fully editable**. For the exact v9 numbers, tables and
> field-relative weights, use the scripts and data in the repository root.

## Two ways to run it

**1. Open in the browser (no server) — simplest**
Just open `kanon_index.html` in any modern browser. The calculation runs client-side
(`js/kanon.js`) and queries OpenAlex directly. Edit the e-mail for the OpenAlex *polite pool*
at the top of `js/kanon.js` (`POLITE_EMAIL`).

**2. Run the Flask app (local server)**
```bash
pip install -r requirements.txt
export OPENALEX_MAILTO="you@example.org"   # Windows: set OPENALEX_MAILTO=you@example.org
python app.py
# open http://127.0.0.1:5000
```

## What you can edit
- **Per-field weights** `w_C, w_A, w_S, w_J` (must sum to 1) — in `js/kanon.js` (`FIELD_WEIGHTS`).
- **Hyperauthorship penalty** `α` — `FIELD_ALPHA` (credit scales as `1/N^α`).
- **Sophistication blend** `β` — `FIELD_BETA` (`S = β·S_text + (1−β)·S_concepts`).

## Privacy / notes
- No personal e-mail is shipped: set your own via `OPENALEX_MAILTO` (server) or `POLITE_EMAIL`
  (browser). The placeholder is `anonymous@example.com`.
- All data come from the public [OpenAlex](https://openalex.org) API. License: MIT (repository root).
- This calculator is a teaching/exploration aid, **not** the canonical computation behind the
  paper's reported numbers.
