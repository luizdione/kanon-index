#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KANON-Index — Reconstrutor da CLASSE NEGATIVA do benchmark (researchers v2)
===========================================================================
Motivacao (achado 2026-05-22): o `example500_orcids.csv` antigo estava quebrado:
  - 500 linhas / 398 ORCIDs unicos; 85 pesquisadores em >1 area;
  - o MESMO autor rotulado em areas diferentes (Freddie Bray como Medicine E Physics);
  - pool biomedico-dominado (IARC, GBD): os "fisicos"/"economistas" eram epidemiologistas.
Isso tornou o MeSH um discriminador ESPURIO de area e invalidou a validacao
estratificada (Nobel vs researcher por area).

Este script monta uma classe negativa HONESTA e CASADA POR CAMPO:
  - profila os laureados (positivos) e mede a distribuicao REAL de campo dominante
    OpenAlex de cada categoria Nobel. Isso importa porque a categoria Nobel != campo
    OpenAlex: os Nobel de "Medicine" (Ambros, Kariko, Paabo...) sao dominantes em
    "Biochemistry, Genetics and Molecular Biology", nao em "Medicine" (clinico).
  - frequency-matching: para cada area, sorteia pesquisadores nao-laureados cuja
    distribuicao de campo dominante REPRODUZ a dos laureados daquela area;
  - muito citados (teste dificil: Nobel vs elite da mesma area), piso de produtividade;
  - 1 area por ORCID, deduplicado; exclui laureados por ORCID E por id OpenAlex.

Rota tecnica (verificada na API em 2026-05-22):
  - `x_concepts` no objeto autor esta ZERADO (deprecado) -> nao serve de pre-filtro.
  - filtro `orcid:` de autor EXISTE, mas muito laureado nao tem ORCID ligado no
    OpenAlex (ex.: Victor Ambros -> count=0); por isso ha fallback por nome (search=).
  - /topics?filter=field.id:F          -> topics de cada campo.
  - /authors?filter=topics.id:T,...    -> autores daquele topic, ordenaveis por citacoes.
  - author.topics[].field              -> campo dominante (current, nao deprecado).

USO (na maquina do Dione, env datascience):
    python build_researchers_benchmark.py --dry-run     # so diagnostico (Nobel vs negativos)
    python build_researchers_benchmark.py               # grava researchers_benchmark_v2.csv
Depois:
    python kanon_full_collector_v3.py --researchers-only \
        --researchers-csv researchers_benchmark_v2.csv
    python kanon_weight_optimizer_v4.py
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import defaultdict

MAILTO = os.environ.get("OPENALEX_MAILTO", "anonymous@example.com")
BASE = 'https://api.openalex.org'
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, 'dados_reais', 'authors_benchmark_cache.json')

SELECT = ('id,display_name,orcid,works_count,cited_by_count,'
          'summary_stats,topics,last_known_institutions')
_SLEEP = 0.12

# Campos OpenAlex (so para rotular saidas; o casamento e data-driven via Nobel)
FIELD_NAMES = {
    11: 'Agricultural and Biological Sciences', 12: 'Arts and Humanities',
    13: 'Biochemistry, Genetics and Molecular Biology', 14: 'Business, Management and Accounting',
    15: 'Chemical Engineering', 16: 'Chemistry', 17: 'Computer Science', 18: 'Decision Sciences',
    19: 'Earth and Planetary Sciences', 20: 'Economics, Econometrics and Finance', 21: 'Energy',
    22: 'Engineering', 23: 'Environmental Science', 24: 'Immunology and Microbiology',
    25: 'Materials Science', 26: 'Mathematics', 27: 'Medicine', 28: 'Neuroscience', 29: 'Nursing',
    30: 'Pharmacology, Toxicology and Pharmaceutics', 31: 'Physics and Astronomy', 32: 'Psychology',
    33: 'Social Sciences', 34: 'Veterinary', 35: 'Dentistry', 36: 'Health Professions'}


def _req(url):
    full = url + ('&' if '?' in url else '?') + 'mailto=' + MAILTO
    req = urllib.request.Request(full, headers={'User-Agent': 'kanon-benchmark/1.0',
                                                'Accept': 'application/json'})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def get(url, retries=3):
    for i in range(retries + 1):
        try:
            return _req(url)
        except Exception as e:
            if i == retries:
                print(f"    [WARN] falha apos {retries} tentativas: {e}", flush=True)
                return None
            time.sleep(0.6 * (i + 1))


def norm_orcid(o):
    if not o:
        return None
    return str(o).strip().rstrip('/').split('/')[-1].upper()


def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            return json.load(open(CACHE_PATH, encoding='utf-8'))
        except Exception:
            return {}
    return {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + '.tmp'
    json.dump(cache, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False)
    os.replace(tmp, CACHE_PATH)


def dominant_field(author):
    """(field_id:int, field_name:str, purity:float) a partir de author.topics."""
    fc, names = {}, {}
    for t in author.get('topics', []) or []:
        fl = t.get('field') or {}
        fid = fl.get('id')
        if not fid:
            continue
        fid = int(str(fid).split('/')[-1])
        fc[fid] = fc.get(fid, 0) + (t.get('count', 0) or 0)
        names[fid] = fl.get('display_name', '')
    if not fc:
        return None, None, 0.0
    top = max(fc, key=fc.get)
    return top, names[top], fc[top] / (sum(fc.values()) or 1)


def author_record(a):
    inst = ''
    lk = a.get('last_known_institutions') or []
    if lk:
        inst = lk[0].get('display_name', '') or ''
    return {
        'id': (a.get('id') or '').split('/')[-1],
        'name': a.get('display_name', ''),
        'orcid': norm_orcid(a.get('orcid')),
        'cited_by_count': a.get('cited_by_count', 0) or 0,
        'works_count': a.get('works_count', 0) or 0,
        'h_index': (a.get('summary_stats') or {}).get('h_index'),
        'institution': inst,
    }


def norm_name(s):
    s = unicodedata.normalize('NFKD', s or '').encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z ]', ' ', s).split()


def best_name_match(name, results):
    """Casa por sobrenome + primeiro nome COMPLETO; aceita so-inicial se o sobrenome
    for unico nos resultados (senao retorna {} -> ambiguo, nao polui a distribuicao)."""
    tp = norm_name(name)
    if not tp:
        return {}
    tlast, tfirst = tp[-1], tp[0]
    full, initial = [], []
    for r in results:
        rp = norm_name(r.get('display_name', ''))
        if not rp or rp[-1] != tlast:
            continue
        if rp[0] == tfirst:
            full.append(r)
        elif rp[0][0] == tfirst[0]:
            initial.append(r)
    if full:
        return max(full, key=lambda x: x.get('works_count', 0) or 0)
    if len(initial) == 1:
        return initial[0]
    return {}


def resolve_author(name, orcid, cache):
    """ORCID (autoritativo) -> fallback por nome validado.
    So reutiliza cache NAO-vazio (evita cache envenenado por run anterior)."""
    key = 'nb:' + (orcid or name)
    cached = cache.get(key)
    if cached:
        return cached
    a = {}
    if orcid:
        d = get(f"{BASE}/authors?filter=orcid:{orcid}&per-page=1&select={SELECT}")
        if d and d.get('results'):
            a = d['results'][0]
    if not a and name:
        d = get(f"{BASE}/authors?search={urllib.parse.quote(name)}&per-page=8&select={SELECT}")
        if d and d.get('results'):
            a = best_name_match(name, d['results'])
    cache[key] = a
    time.sleep(_SLEEP)
    return a


def fetch_nobel_profile(nobel_csv, cache):
    """Retorna (exclude_orcids, exclude_ids, nobel_dist[area]={field_id:count}, resolved[area])."""
    rows = list(csv.DictReader(open(nobel_csv, encoding='utf-8')))
    exclude_orcids, exclude_ids = set(), set()
    dist = defaultdict(lambda: defaultdict(int))
    resolved = defaultdict(int)
    for i, r in enumerate(rows, 1):
        oc = norm_orcid(r.get('orcid'))
        if oc:
            exclude_orcids.add(oc)
        area = r.get('field', '')
        a = resolve_author(r.get('name', ''), oc, cache)
        if a:
            aid = (a.get('id') or '').split('/')[-1]
            if aid:
                exclude_ids.add(aid)
            fid, _, _ = dominant_field(a)
            if fid:
                dist[area][fid] += 1
                resolved[area] += 1
        if i % 40 == 0:
            print(f"    [Nobel] perfil: {i}/{len(rows)}", flush=True)
            save_cache(cache)
    return exclude_orcids, exclude_ids, dist, resolved


def harvest_field(field_id, exclude_orcids, exclude_ids, used, cache,
                  works_floor, topics_per_field, pages_per_topic, pool_cap):
    """Pool ranqueado (por citacoes) de autores cujo campo DOMINANTE == field_id."""
    tp = get(f"{BASE}/topics?filter=field.id:{field_id}&sort=works_count:desc"
             f"&per-page={topics_per_field}&select=id")
    topic_ids = [t['id'].split('/')[-1] for t in (tp['results'] if tp else [])]
    cand = {}
    for tid in topic_ids:
        cursor = '*'
        for _ in range(pages_per_topic):
            url = (f"{BASE}/authors?filter=topics.id:{tid},has_orcid:true,"
                   f"works_count:>{works_floor}&sort=cited_by_count:desc"
                   f"&per-page=200&cursor={urllib.parse.quote(cursor)}&select={SELECT}")
            d = get(url)
            if not d or not d.get('results'):
                break
            for a in d['results']:
                rec = author_record(a)
                oc, aid = rec['orcid'], rec['id']
                if not oc or oc in exclude_orcids or aid in exclude_ids:
                    continue
                if oc in used or oc in cand:
                    continue
                fid, _, _ = dominant_field(a)
                if fid != field_id:
                    continue
                cand[oc] = rec
            cursor = (d.get('meta') or {}).get('next_cursor')
            if not cursor:
                break
            time.sleep(_SLEEP)
        if len(cand) >= pool_cap:
            break
    return sorted(cand.values(), key=lambda r: r['cited_by_count'], reverse=True)


def allocate_targets(dist_area, resolved_n, n_target):
    """Alvo por campo proporcional a distribuicao Nobel da area (soma == n_target)."""
    if resolved_n <= 0:
        return {}
    fields = sorted(dist_area.items(), key=lambda kv: -kv[1])
    targets, allocated = {}, 0
    for fid, cnt in fields:
        t = int(round(cnt / resolved_n * n_target))
        targets[fid] = t
        allocated += t
    diff = n_target - allocated
    if fields:  # ajusta arredondamento no maior campo
        targets[fields[0][0]] += diff
    return {f: max(0, t) for f, t in targets.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nobel-csv', default='exampleNOBEL_orcids.csv')
    ap.add_argument('--out', default='researchers_benchmark_v2.csv')
    ap.add_argument('--n-per-area', type=int, default=125)
    ap.add_argument('--works-floor', type=int, default=30)
    ap.add_argument('--topics-per-field', type=int, default=25)
    ap.add_argument('--pages-per-topic', type=int, default=2)
    ap.add_argument('--min-field-share', type=float, default=0.05,
                    help='ignora campos Nobel raros (< share) no casamento')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    if not os.path.exists(args.nobel_csv):
        print(f"[ERRO] nao achei {args.nobel_csv}"); sys.exit(1)

    cache = load_cache()
    print("[1/3] Profilando laureados (ORCID + fallback nome) e medindo campo dominante...")
    ex_orcids, ex_ids, dist, resolved = fetch_nobel_profile(args.nobel_csv, cache)
    save_cache(cache)
    print(f"      exclusao: {len(ex_orcids)} ORCIDs + {len(ex_ids)} ids OpenAlex de laureados")
    fname = lambda f: FIELD_NAMES.get(f, f'field/{f}')
    for area in dist:
        tot = resolved[area]
        top = sorted(dist[area].items(), key=lambda x: -x[1])
        print(f"      Nobel {area} (resolvidos={tot}): "
              + ', '.join(f"{fname(f)}={c}" for f, c in top[:4]))

    print("\n[2/3] Coletando classe negativa frequency-matched por area...")
    used = set()
    all_rows, report = [], {}
    for area in dist:
        # campos relevantes (>= min share) e alvos proporcionais
        tot = resolved[area]
        rel = {f: c for f, c in dist[area].items() if tot and c / tot >= args.min_field_share}
        if not rel:
            rel = dict(dist[area])
        rel_n = sum(rel.values())
        targets = allocate_targets(rel, rel_n, args.n_per_area)
        chosen, got_by_field = [], {}
        deficit = 0
        pools = {}
        for fid, tgt in sorted(targets.items(), key=lambda x: -x[1]):
            pool = harvest_field(fid, ex_orcids, ex_ids, used, cache,
                                 args.works_floor, args.topics_per_field,
                                 args.pages_per_topic, max(tgt * 4, 40))
            pools[fid] = pool
            take = pool[:tgt]
            for r in take:
                used.add(r['orcid']); chosen.append((fid, r))
            got_by_field[fid] = len(take)
            deficit += max(0, tgt - len(take))
            print(f"  [{area}] {fname(fid):42s} alvo={tgt:3d} obtidos={len(take):3d} (pool={len(pool)})", flush=True)
        # cobre deficit com sobras dos maiores pools (mantendo a area)
        if deficit:
            leftovers = []
            for fid, pool in pools.items():
                leftovers += [(fid, r) for r in pool[got_by_field.get(fid, 0):]]
            leftovers.sort(key=lambda x: -x[1]['cited_by_count'])
            for fid, r in leftovers:
                if deficit <= 0:
                    break
                if r['orcid'] in used:
                    continue
                used.add(r['orcid']); chosen.append((fid, r)); deficit -= 1
        for fid, r in chosen:
            all_rows.append({'name': r['name'], 'orcid': r['orcid'],
                             'field': area, 'institution': r['institution']})
        hs = [r['h_index'] for _, r in chosen if r['h_index'] is not None]
        cs = [r['cited_by_count'] for _, r in chosen]
        med = lambda v: (sorted(v)[len(v) // 2] if v else 0)
        neg_dist = defaultdict(int)
        for fid, _ in chosen:
            neg_dist[fid] += 1
        report[area] = {'n': len(chosen), 'h_med': med(hs), 'cit_med': med(cs),
                        'neg_dist': dict(neg_dist), 'nobel_dist': dict(rel)}
        save_cache(cache)

    print("\n[3/3] Resumo + casamento de campo (Nobel% vs Negativos%)")
    print(f"  Total: {len(all_rows)} | ORCIDs unicos: {len(set(r['orcid'] for r in all_rows))}")
    for area, s in report.items():
        flag = '' if s['n'] >= args.n_per_area else '  <-- COTA NAO PREENCHIDA'
        print(f"\n  {area}: n={s['n']} h_med={s['h_med']} cit_med={s['cit_med']}{flag}")
        nob_tot = sum(s['nobel_dist'].values()) or 1
        neg_tot = sum(s['neg_dist'].values()) or 1
        fids = sorted(set(s['nobel_dist']) | set(s['neg_dist']),
                      key=lambda f: -(s['nobel_dist'].get(f, 0)))
        for f in fids:
            nob = 100 * s['nobel_dist'].get(f, 0) / nob_tot
            neg = 100 * s['neg_dist'].get(f, 0) / neg_tot
            print(f"      {fname(f):42s} Nobel={nob:4.0f}%  Neg={neg:4.0f}%")

    if args.dry_run:
        print("\n[DRY-RUN] nada gravado.")
        return
    with open(args.out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['name', 'orcid', 'field', 'institution'])
        w.writeheader(); w.writerows(all_rows)
    print(f"\n[OK] gravado: {args.out} ({len(all_rows)} pesquisadores)")
    print("Proximo: python kanon_full_collector_v3.py --researchers-only "
          f"--researchers-csv {args.out}  &&  python kanon_weight_optimizer_v4.py")


if __name__ == '__main__':
    main()
