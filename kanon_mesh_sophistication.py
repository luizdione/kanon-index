#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KANON-Index — Modulo de Sofisticacao via MeSH (S_text biomedico)
================================================================
Calcula um score de sofisticacao experimental (0-1) por artigo a partir dos
descritores MeSH que o OpenAlex ja entrega por work. Usa a arvore E
("Analytical, Diagnostic and Therapeutic Techniques, and Equipment"):
    E01 Diagnosis | E02 Therapeutics | E03 Anesthesia | E04 Surgical
    E05 Investigative Techniques  <- nucleo de "metodos"
    E06 Dentistry | E07 Equipment and Supplies

Os tree numbers vem do registro JSON-LD do NLM (id.nlm.nih.gov), cacheados
localmente por descriptor_ui em dados_reais/mesh_tree_cache.json.

USO:
    python kanon_mesh_sophistication.py --self-test
    python kanon_mesh_sophistication.py --live
"""

import json
import os
import sys
import time
import urllib.request

CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'dados_reais', 'mesh_tree_cache.json')

# So a arvore E conta como tecnica/equipamento; pesos por ramo
E_BRANCH_WEIGHTS = {
    'E05': 1.00,  # Investigative Techniques
    'E07': 0.85,  # Equipment and Supplies
    'E01': 0.70,  # Diagnosis
    'E04': 0.65,  # Surgical Procedures
    'E03': 0.60,  # Anesthesia
    'E02': 0.55,  # Therapeutics
    'E06': 0.55,  # Dentistry
}

# Pausa entre requisicoes ao NLM (limitada para nao demorar demais nem agredir o servidor)
_NLM_MAX_DELAY = 0.06
_NLM_RETRIES = 2


def load_cache(path=CACHE_PATH):
    if os.path.exists(path):
        try:
            return json.load(open(path, 'r', encoding='utf-8'))
        except Exception:
            return {}
    return {}


def save_cache(cache, path=CACHE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    json.dump(cache, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False)
    os.replace(tmp, path)   # escrita atomica (nao corrompe se interromper)


def _extract_tree_numbers(obj):
    tn = obj.get('treeNumber') or obj.get('treeNumbers') or []
    if isinstance(tn, str):
        tn = [tn]
    out = []
    for x in tn:
        if isinstance(x, str):
            out.append(x.split('/')[-1])
        elif isinstance(x, dict):
            val = x.get('@id') or x.get('@value') or x.get('label')
            if isinstance(val, dict):
                val = val.get('@value')
            if val:
                out.append(str(val).split('/')[-1])
    return out


def fetch_tree_numbers(descriptor_ui, timeout=15):
    """Retorna lista de tree numbers (sucesso, pode ser vazia) ou None (falha apos retries)."""
    url = f'https://id.nlm.nih.gov/mesh/{descriptor_ui}.json'
    for attempt in range(_NLM_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'kanon/1.0',
                                                       'Accept': 'application/json'})
            obj = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
            return _extract_tree_numbers(obj)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []          # descritor sem registro -> sem tecnica
            time.sleep(0.5 * (attempt + 1))
        except Exception:
            time.sleep(0.5 * (attempt + 1))
    return None                    # falha: NAO cachear (tenta de novo na proxima)


def ensure_cached(descriptor_uis, cache, delay=_NLM_MAX_DELAY, verbose=True, _state={'n': 0}):
    """Garante tree numbers em cache para os UIs (busca os faltantes). So cacheia sucessos."""
    eff_delay = min(delay, _NLM_MAX_DELAY)
    missing = [u for u in dict.fromkeys(descriptor_uis) if u and u not in cache]
    fails = 0
    for ui in missing:
        res = fetch_tree_numbers(ui)
        if res is not None:
            cache[ui] = res
        else:
            fails += 1
        _state['n'] += 1
        if verbose and _state['n'] % 100 == 0:
            print(f"    [MeSH] cache: {len(cache)} descritores ({fails} falhas nesta leva)", flush=True)
        time.sleep(eff_delay)
    return cache


def descriptor_score(tree_numbers, is_major):
    best = 0.0
    for tn in tree_numbers:
        if not tn or tn[0] != 'E':
            continue
        w = E_BRANCH_WEIGHTS.get(tn.split('.')[0], 0.40)
        depth_score = min(1.0, tn.count('.') / 4.0)
        s = w * (0.5 + 0.5 * depth_score)
        if s > best:
            best = s
    if best > 0 and is_major:
        best = min(1.0, best * 1.25)
    return best


def mesh_sophistication_score(mesh_list, cache):
    seen = {}
    for m in mesh_list or []:
        ui = m.get('descriptor_ui')
        if not ui:
            continue
        seen[ui] = seen.get(ui, False) or bool(m.get('is_major_topic'))
    scores = []
    for ui, major in seen.items():
        s = descriptor_score(cache.get(ui, []), major)
        if s > 0:
            scores.append(s)
    if not scores:
        return 0.0
    scores.sort(reverse=True)
    return round(min(1.0, sum(scores[:5]) / 3.0), 4)


def _self_test():
    cache = {'D1': ['E05.595'], 'D2': ['E05.595.402.350.700'],
             'D3': ['G03.123', 'C01.001'], 'D4': ['E07.700']}
    assert descriptor_score(cache['D2'], False) > descriptor_score(cache['D1'], False)
    assert descriptor_score(cache['D3'], True) == 0.0
    assert descriptor_score(cache['D1'], True) > descriptor_score(cache['D1'], False)
    hi = [{'descriptor_ui': 'D2', 'is_major_topic': True},
          {'descriptor_ui': 'D4', 'is_major_topic': True},
          {'descriptor_ui': 'D1', 'is_major_topic': False}]
    lo = [{'descriptor_ui': 'D3', 'is_major_topic': True}]
    assert mesh_sophistication_score(hi, cache) > 0.5
    assert mesh_sophistication_score(lo, cache) == 0.0
    assert mesh_sophistication_score([], cache) == 0.0
    print("SELF-TEST OK: scoring MeSH por arvore E funciona.")


def _live_test():
    def get(u):
        return json.loads(urllib.request.urlopen(
            u + '&mailto=' + os.environ.get("OPENALEX_MAILTO", "anonymous@example.com"), timeout=20).read())
    d = get('https://api.openalex.org/works?filter=has_pmid:true,title.search:CRISPR'
            '&per_page=2&select=id,title,mesh')
    cache = load_cache()
    for w in d['results']:
        uis = [m['descriptor_ui'] for m in w.get('mesh', []) if m.get('descriptor_ui')]
        ensure_cached(uis, cache)
        print(f"  {w['title'][:55]:55s} | MeSH={len(w.get('mesh',[]))} "
              f"| S_text(MeSH)={mesh_sophistication_score(w.get('mesh', []), cache)}")
    save_cache(cache)
    print(f"Cache salvo: {CACHE_PATH} ({len(cache)} descritores)")


if __name__ == '__main__':
    if '--live' in sys.argv:
        _live_test()
    else:
        _self_test()
