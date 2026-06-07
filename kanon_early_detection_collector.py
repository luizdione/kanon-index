#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KANON-Index — Coletor de DETECCAO PRECOCE (point-in-time, janela de 20 anos)
============================================================================
Pergunta: o KANON identifica o futuro extra-classe ENQUANTO sua producao ainda
parece mediana? Para responder de forma honesta reconstruimos o curriculo de cada
autor "como era" em horizontes anteriores ao reconhecimento, sem vazar o futuro.

Desenho (combinado com Dione, 2026-05-23):
  - Mesmos 164 laureados Nobel (positivos) + 500 controles casados por campo
    (researchers_benchmark_v2.csv, classe negativa honesta).
  - Horizontes k em {0, 5, 10}: ano de corte = ref_year - k.
        ref_year = ano do Nobel (laureados) | pseudo-ano sorteado da distribuicao
        de anos de premio da MESMA area (controles), para casar o estagio temporal.
  - Janela de 20 anos terminando no corte: artigos com pub_year em [corte-19, corte].
  - Metricas POINT-IN-TIME: citacoes que cada artigo tinha acumulado ATE o ano de
    corte, via counts_by_year (NUNCA o total atual). h-index idem (do ano de corte).

LIMITACAO CONHECIDA (importante p/ o paper): counts_by_year do OpenAlex cobre ~10-14
anos (comeca ~2012). Logo, C (citacoes) e h-index point-in-time sao confiaveis em
T0/T-5 mas SUBESTIMAM em T-10 (registra-se `cby_min_year` por linha p/ sinalizar).
Ja A (autoria) e S (S_text/MeSH + S_concepts) sao EXATOS em qualquer horizonte,
pois dependem so do conteudo dos artigos publicados ate o corte.

Saida: dados_reais/<data>/early/kanon_early_real.csv — 1 linha por (autor, horizonte),
com as MESMAS colunas que o kanon_weight_optimizer_v4 consome + identificadores.

USO (maquina do Dione, env datascience):
    python kanon_early_detection_collector.py --self-test
    python kanon_early_detection_collector.py --resume
    python kanon_early_detection_collector.py --horizons 0 5 10 --window 20
Depois: python kanon_early_detection_analysis.py
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kanon_full_collector_v3 as v3
import kanon_mesh_sophistication as meshmod

HERE = os.path.dirname(os.path.abspath(__file__))


def point_in_time_citations(work, cutoff_year):
    """Citacoes que o work tinha ate cutoff_year (soma counts_by_year<=corte).
    Se o corte e >= ano atual, usa o total atual (cited_by_count)."""
    cby = work.get('counts_by_year') or []
    if not cby:
        return work.get('cited_by_count', 0) or 0, None
    years = [c['year'] for c in cby]
    cmin = min(years) if years else None
    if cutoff_year >= max(years):
        # corte no presente: total atual menos o que veio depois (nada)
        ptc = work.get('cited_by_count', 0) or 0
    else:
        ptc = sum(c.get('cited_by_count', 0) or 0 for c in cby if c['year'] <= cutoff_year)
    return ptc, cmin


class KANONEarlyCollector(v3.KANONCollectorV3):
    """Reusa os scorers validados do v3 (palavras-chave, MeSH, Topics) e
    reconstroi features por janela com citacoes point-in-time."""

    def _fetch_all_works(self, author_id, win_from, win_to, hard_cap=500):
        """Busca works do autor em [win_from, win_to] (sort por ano desc), com
        counts_by_year/topics/mesh (campos default do OpenAlex)."""
        works, cursor = [], '*'
        while len(works) < hard_cap:
            time.sleep(self.DELAY)
            url = (self.BASE_URL + "/works?filter=authorships.author.id:" + author_id +
                   f",publication_year:{win_from}-{win_to}"
                   "&sort=publication_year:desc&per_page=50&cursor=" + cursor)
            data = self._fetch_json(url)
            if not data or not data.get('results'):
                break
            works.extend(data['results'])
            cursor = (data.get('meta') or {}).get('next_cursor')
            if not cursor:
                break
        return works[:hard_cap]

    def _features_in_window(self, works, author_id, cutoff_year, window):
        """Calcula as colunas que o otimizador v4 consome, na janela
        [cutoff-window+1, cutoff], com citacoes point-in-time."""
        lo = cutoff_year - window + 1
        win = [w for w in works
               if w.get('publication_year') and lo <= w['publication_year'] <= cutoff_year]
        if not win:
            return None

        total_authors = 0
        positions, s_text, mesh_scores, concept_scores, journals = [], [], [], [], []
        work_ptc, cby_mins = [], []
        works_with_abstract = works_with_mesh = 0

        for w in win:
            auths = w.get('authorships', [])
            total_authors += len(auths)
            for idx, a in enumerate(auths):
                if (a.get('author') or {}).get('id') == author_id:
                    positions.append(idx + 1)
                    break
            abstract = self._reconstruct_abstract(w.get('abstract_inverted_index', {}))
            if abstract and len(abstract) > 20:
                works_with_abstract += 1
            kw, _ = self._estimate_complexity_from_text((w.get('title', '') or '') + ' ' + abstract)
            ms = self._mesh_score(w)
            if ms > 0:
                works_with_mesh += 1
                mesh_scores.append(ms)
            st = max(kw, ms)
            if st > 0:
                s_text.append(st)
            cs = self._estimate_complexity_from_concepts(w)
            if cs > 0:
                concept_scores.append(cs)
            j = (((w.get('primary_location') or {}).get('source') or {}).get('display_name', '')
                 or ((w.get('host_venue') or {}).get('display_name', '')))
            if j:
                journals.append(j)
            ptc, cmin = point_in_time_citations(w, cutoff_year)
            work_ptc.append(ptc)
            if cmin is not None:
                cby_mins.append(cmin)

        n = len(win)
        avg_auth = total_authors / n
        avg_pos = sum(positions) / len(positions) if positions else 1
        avg_s_text = sum(s_text) / len(s_text) if s_text else 0.0
        avg_mesh = sum(mesh_scores) / len(mesh_scores) if mesh_scores else 0.0
        avg_concept = sum(concept_scores) / len(concept_scores) if concept_scores else 0.0
        total_cit = int(sum(work_ptc))
        # h-index point-in-time na janela
        srt = sorted(work_ptc, reverse=True)
        h_asof = 0
        for i, c in enumerate(srt, 1):
            if c >= i:
                h_asof = i
            else:
                break
        return {
            'total_citations': total_cit,
            'avg_author_position': round(avg_pos, 2),
            'avg_authors_per_paper': round(avg_auth, 2),
            'complexity_keyword_score': round(avg_s_text, 4),   # S_text hibrido
            'complexity_concept_score': round(avg_concept, 4),  # S_concepts
            'complexity_mesh_score': round(avg_mesh, 4),
            'mesh_coverage': round(works_with_mesh / n, 4),
            'journals': len(set(journals)),
            'h_index': int(h_asof),
            'total_publications': n,
            'complexity_n_works_analyzed': works_with_abstract,
            'cby_min_year': min(cby_mins) if cby_mins else '',
        }

    def process_author_horizons(self, author, name, orcid, field, ref_year,
                                is_nobel, horizons, window):
        aid = author.get('id', '')
        if not aid:
            return []
        win_lo = (ref_year - max(horizons)) - window + 1
        works = self._fetch_all_works(aid, win_lo, ref_year)
        if not works:
            return []
        rows = []
        for k in horizons:
            cutoff = ref_year - k
            feats = self._features_in_window(works, aid, cutoff, window)
            if feats is None:
                continue
            row = {'name': name, 'orcid': orcid, 'field': field,
                   'is_nobel': is_nobel, 'horizon': k, 'ref_year': ref_year,
                   'cutoff_year': cutoff}
            row.update(feats)
            rows.append(row)
        return rows


FIELDNAMES = ['name', 'orcid', 'field', 'is_nobel', 'horizon', 'ref_year', 'cutoff_year',
              'total_citations', 'avg_author_position', 'avg_authors_per_paper',
              'complexity_keyword_score', 'complexity_concept_score', 'complexity_mesh_score',
              'mesh_coverage', 'journals', 'h_index', 'total_publications',
              'complexity_n_works_analyzed', 'cby_min_year']


def _load_inputs(nobel_csv, researchers_csv, seed=42):
    """Laureados com prize year (col 'year'); controles recebem pseudo-ref-year
    sorteado da distribuicao de anos de premio da MESMA area."""
    import random
    rng = random.Random(seed)
    nob = list(csv.DictReader(open(nobel_csv, encoding='utf-8')))
    res = list(csv.DictReader(open(researchers_csv, encoding='utf-8')))
    years_by_area = defaultdict(list)
    for r in nob:
        try:
            years_by_area[r['field']].append(int(r['year']))
        except Exception:
            pass
    items = []
    for r in nob:
        try:
            ry = int(r['year'])
        except Exception:
            continue
        items.append((r['name'], r.get('orcid', ''), r['field'], ry, 1))
    for r in res:
        pool = years_by_area.get(r['field']) or [y for ys in years_by_area.values() for y in ys]
        ry = rng.choice(pool) if pool else 2020
        items.append((r['name'], r.get('orcid', ''), r['field'], ry, 0))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nobel-csv', default='exampleNOBEL_orcids.csv')
    ap.add_argument('--researchers-csv', default='researchers_benchmark_v2.csv')
    ap.add_argument('--horizons', type=int, nargs='+', default=[0, 5, 10])
    ap.add_argument('--window', type=int, default=20)
    ap.add_argument('--output-dir', default='dados_reais')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()

    if args.self_test:
        _self_test(); return

    col = KANONEarlyCollector.__new__(KANONEarlyCollector)
    v3.KANONCollectorV3.__init__(col, output_dir=args.output_dir, from_year=1990,
                                 to_year=datetime.now().year, max_works=500)

    items = _load_inputs(args.nobel_csv, args.researchers_csv)
    # pasta FIXA (sem data) para que --resume sempre encontre o parcial,
    # mesmo retomando em outro dia
    out_dir = os.path.join(args.output_dir, 'early')
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(out_dir, 'kanon_early_real.csv')

    done = set()
    if args.resume and os.path.exists(out_path):
        for r in csv.DictReader(open(out_path, encoding='utf-8')):
            done.add((r['orcid'] or r['name']))
        print(f"[RESUME] {len(done)} autores ja coletados")
    mode = 'a' if (args.resume and os.path.exists(out_path)) else 'w'
    fout = open(out_path, mode, newline='', encoding='utf-8')
    writer = csv.DictWriter(fout, fieldnames=FIELDNAMES)
    if mode == 'w':
        writer.writeheader()

    t0 = time.time()
    for i, (name, orcid, field, ref_year, is_nobel) in enumerate(items, 1):
        key = orcid or name
        if key in done:
            continue
        author = col._get_author_by_orcid(orcid) or col._get_author_by_name(name, field)
        if author:
            for row in col.process_author_horizons(author, name, orcid, field, ref_year,
                                                    is_nobel, args.horizons, args.window):
                writer.writerow(row)
            fout.flush()
        if i % 20 == 0:
            meshmod.save_cache(col.mesh_cache)
            eta = (len(items) - i) * (time.time() - t0) / i / 60
            print(f"  [{100*i//len(items):3d}%] {i}/{len(items)} | cache MeSH={len(col.mesh_cache)} | ETA={eta:.0f}min", flush=True)
    fout.close()
    meshmod.save_cache(col.mesh_cache)
    # copia ativa
    import shutil
    shutil.copy2(out_path, os.path.join(args.output_dir, 'kanon_early_real.csv'))
    print(f"\n[OK] {out_path}\n[OK] copia ativa: {os.path.join(args.output_dir, 'kanon_early_real.csv')}")


def _self_test():
    # point_in_time_citations
    w = {'cited_by_count': 100, 'counts_by_year': [
        {'year': 2018, 'cited_by_count': 5}, {'year': 2019, 'cited_by_count': 10},
        {'year': 2020, 'cited_by_count': 30}, {'year': 2021, 'cited_by_count': 55}]}
    assert point_in_time_citations(w, 2019)[0] == 15, point_in_time_citations(w, 2019)
    assert point_in_time_citations(w, 2021)[0] == 100  # corte no presente -> total
    assert point_in_time_citations(w, 2025)[0] == 100
    assert point_in_time_citations({'cited_by_count': 7, 'counts_by_year': []}, 2010) == (7, None)
    print("SELF-TEST OK (point_in_time_citations)")


if __name__ == '__main__':
    main()
