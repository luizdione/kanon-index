#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KANON-Index Full Collector v3 — S_text via MeSH + S_concepts via Topics (Fase 0)."""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kanon_full_collector_v2 as v2
import kanon_mesh_sophistication as meshmod


def _load_subfield_baseline(path):
    doc = json.load(open(path, 'r', encoding='utf-8'))
    raw = doc.get('subfields', doc)
    base = {}
    for k, v in raw.items():
        kid = k.split('/')[-1]
        score = v['sophistication_baseline'] if isinstance(v, dict) else v
        base[kid] = float(score)
    return base, doc.get('_meta', {})


def _subfield_id(topic):
    sub = topic.get('subfield') or {}
    sid = sub.get('id') or ''
    return str(sid).split('/')[-1] if sid else None


class KANONCollectorV3(v2.KANONCollector):
    """S_text hibrido (marcadores + MeSH) e S_concepts via Topics+baseline."""

    def __init__(self, *args, baseline_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if baseline_path is None:
            baseline_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'dados_reais', 'subfield_sophistication_baseline_v1.0.json')
        self.subfield_baseline, self.baseline_meta = _load_subfield_baseline(baseline_path)
        self._missing_subfields = set()
        self.FIELDNAMES = list(v2.KANONCollector.FIELDNAMES) + ['complexity_mesh_score', 'mesh_coverage']
        self.mesh_cache = meshmod.load_cache()
        self._mesh_cache_dirty = 0
        print(f"[CONFIG] S_concepts via Topics | baseline subfields: {len(self.subfield_baseline)} "
              f"| sha256={self.baseline_meta.get('sha256','?')[:12]}")
        print(f"[CONFIG] S_text hibrido: marcadores + MeSH (arvore E) | cache MeSH: {len(self.mesh_cache)}")

    def _estimate_complexity_from_concepts(self, work):
        topics = work.get('topics', []) or []
        if not topics:
            return 0.0
        num = den = 0.0
        for t in topics:
            sid = _subfield_id(t)
            if sid is None:
                continue
            base = self.subfield_baseline.get(sid)
            if base is None:
                self._missing_subfields.add(sid)
                continue
            score = t.get('score')
            score = float(score) if score is not None else 0.5
            num += score * base
            den += score
        if den <= 0:
            return 0.0
        return round(min(1.0, (num / den) / 10.0), 4)

    def _mesh_score(self, work):
        mesh = work.get('mesh', []) or []
        if not mesh:
            return 0.0
        uis = [m.get('descriptor_ui') for m in mesh if m.get('descriptor_ui')]
        before = len(self.mesh_cache)
        meshmod.ensure_cached(uis, self.mesh_cache, delay=self.DELAY)
        self._mesh_cache_dirty += len(self.mesh_cache) - before
        if self._mesh_cache_dirty >= 50:
            meshmod.save_cache(self.mesh_cache)
            self._mesh_cache_dirty = 0
        return meshmod.mesh_sophistication_score(mesh, self.mesh_cache)

    def _process_author(self, author_data, name, orcid, field):
        author_id = author_data.get('id', '')
        if not author_id:
            return None
        h_index = self._get_h_index(author_data)
        total_pubs = author_data.get('works_count', 0) or 0
        citations = author_data.get('cited_by_count', 0) or 0
        if total_pubs == 0:
            return None
        works = self._get_author_works(author_id)
        if not works:
            return None

        total_authors = 0
        author_positions, s_text_scores, keyword_scores = [], [], []
        mesh_scores, concept_scores, journals = [], [], []
        works_with_abstract = works_with_mesh = 0

        for work in works:
            total_authors += len(work.get('authorships', []))
            for idx, a in enumerate(work.get('authorships', [])):
                if (a.get('author') or {}).get('id') == author_id:
                    author_positions.append(idx + 1)
                    break
            abstract_text = self._reconstruct_abstract(work.get('abstract_inverted_index', {}))
            full_text = (work.get('title', '') or '') + ' ' + abstract_text
            kw_score, _ = self._estimate_complexity_from_text(full_text)
            if abstract_text and len(abstract_text) > 20:
                works_with_abstract += 1
            if kw_score > 0:
                keyword_scores.append(kw_score)
            mesh_score = self._mesh_score(work)
            if mesh_score > 0:
                works_with_mesh += 1
                mesh_scores.append(mesh_score)
            s_text_work = max(kw_score, mesh_score)
            if s_text_work > 0:
                s_text_scores.append(s_text_work)
            cs = self._estimate_complexity_from_concepts(work)
            if cs > 0:
                concept_scores.append(cs)
            journal = (((work.get('primary_location') or {}).get('source') or {}).get('display_name', '')
                       or ((work.get('host_venue') or {}).get('display_name', '')))
            if journal:
                journals.append(journal)

        avg_auth = total_authors / len(works)
        avg_pos = sum(author_positions) / len(author_positions) if author_positions else 1
        avg_s_text = sum(s_text_scores) / len(s_text_scores) if s_text_scores else 0.0
        avg_mesh = sum(mesh_scores) / len(mesh_scores) if mesh_scores else 0.0
        avg_concept = sum(concept_scores) / len(concept_scores) if concept_scores else 0.0
        mesh_cov = round(works_with_mesh / len(works), 4) if works else 0.0

        if avg_s_text > 0 and avg_concept > 0:
            avg_comp = 0.6 * avg_s_text + 0.4 * avg_concept
        elif avg_s_text > 0:
            avg_comp = avg_s_text
        elif avg_concept > 0:
            avg_comp = avg_concept
        else:
            avg_comp = 0.0

        kanon = self._calculate_kanon(citations, avg_auth, avg_pos, avg_comp, h_index, journals, field)
        result = {
            'name': name, 'orcid': orcid, 'field': field,
            'h_index': h_index, 'total_publications': total_pubs, 'total_citations': citations,
            'avg_authors_per_paper': round(avg_auth, 2), 'avg_author_position': round(avg_pos, 2),
            'avg_complexity': round(avg_comp, 4),
            'complexity_keyword_score': round(avg_s_text, 4),
            'complexity_concept_score': round(avg_concept, 4),
            'complexity_mesh_score': round(avg_mesh, 4), 'mesh_coverage': mesh_cov,
            'complexity_n_works_analyzed': works_with_abstract,
            'works_in_period': len(works), 'top_works': len(works), 'journals': len(set(journals)),
            'from_year': self.from_year, 'to_year': self.to_year,
        }
        result.update(kanon)
        return result


def _self_test():
    import tempfile
    tmp = tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8')
    json.dump({'_meta': {'sha256': 'testhash'}, 'subfields': {'1300': {'sophistication_baseline': 9},
              '3300': {'sophistication_baseline': 3}}}, tmp, ensure_ascii=False)
    tmp.close()
    col = KANONCollectorV3.__new__(KANONCollectorV3)
    col.subfield_baseline, col.baseline_meta = _load_subfield_baseline(tmp.name)
    col._missing_subfields = set()
    w_hi = {'topics': [{'subfield': {'id': 'x/1300'}, 'score': 1.0}]}
    assert abs(col._estimate_complexity_from_concepts(w_hi) - 0.9) < 1e-6
    w_mix = {'topics': [{'subfield': {'id': 'x/1300'}, 'score': 0.8}, {'subfield': {'id': 'x/3300'}, 'score': 0.2}]}
    assert abs(col._estimate_complexity_from_concepts(w_mix) - 0.78) < 1e-6
    assert col._estimate_complexity_from_concepts({'topics': []}) == 0.0
    os.unlink(tmp.name)
    print("SELF-TEST OK")


def main():
    if '--self-test' in sys.argv:
        _self_test(); return
    import argparse
    from datetime import datetime
    from pathlib import Path
    import shutil
    cur = datetime.now().year
    ap = argparse.ArgumentParser()
    ap.add_argument('--nobel-only', action='store_true')
    ap.add_argument('--researchers-only', action='store_true')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--output-dir', default='dados_reais')
    ap.add_argument('--from-year', type=int, default=cur - 20)
    ap.add_argument('--to-year', type=int, default=cur)
    ap.add_argument('--max-works', type=int, default=200)
    ap.add_argument('--baseline', default=None)
    ap.add_argument('--nobel-csv', default='exampleNOBEL_orcids.csv')
    ap.add_argument('--researchers-csv', default='example500_orcids.csv')
    args = ap.parse_args()
    date_tag = datetime.now().strftime('%Y-%m-%d')
    dated_dir = os.path.join(args.output_dir, date_tag, 'coleta_v3')
    Path(dated_dir).mkdir(parents=True, exist_ok=True)
    col = KANONCollectorV3(output_dir=dated_dir, from_year=args.from_year, to_year=args.to_year,
                           max_works=args.max_works, baseline_path=args.baseline)
    if not args.researchers_only and Path(args.nobel_csv).exists():
        print(f"\n[1/2] Nobel ({args.nobel_csv})...")
        col.collect(args.nobel_csv, 'kanon_real_nobel.csv', 'Nobel Laureates', True)
    if not args.nobel_only and Path(args.researchers_csv).exists():
        print(f"\n[2/2] Researchers ({args.researchers_csv})...")
        col.collect(args.researchers_csv, 'kanon_real_researchers.csv', 'Researchers', True)
    meshmod.save_cache(col.mesh_cache)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    for fn in ['kanon_real_nobel.csv', 'kanon_real_researchers.csv']:
        src = os.path.join(dated_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, fn))
    print(f"\nCOLETA v3 COMPLETA. Cache MeSH: {len(col.mesh_cache)} descritores.")


if __name__ == '__main__':
    main()
