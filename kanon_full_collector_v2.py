#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KANON-Index Full Collector v2
=============================
Coleta dados reais de TODOS os pesquisadores (500) E Nobel (164)
via OpenAlex API com:
  - Busca ORCID (primário) + busca por nome (fallback)
  - Correção do bug h_index=0 (summary_stats + estimativa)
  - Salvamento incremental (não perde progresso se interromper)
  - Retomada automática (pula quem já foi coletado)
  - Extração REAL de complexidade metodológica (abstract + concepts + topics)
  - Filtro de período temporal (default: últimos 20 anos)
  - Paginação completa de works (não limita a 50)

Uso:
    python kanon_full_collector_v2.py                        # Coleta tudo (20 anos)
    python kanon_full_collector_v2.py --from-year 2010       # A partir de 2010
    python kanon_full_collector_v2.py --from-year 2000 --to-year 2020
    python kanon_full_collector_v2.py --nobel-only
    python kanon_full_collector_v2.py --researchers-only
    python kanon_full_collector_v2.py --resume
    python kanon_full_collector_v2.py --max-works 200        # Mais works por autor
"""

import os
import sys
import csv
import json
import time
import math
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path

try:
    import urllib.request
    import urllib.error
    import urllib.parse
except ImportError:
    print("ERRO: urllib nao encontrado")
    sys.exit(1)


# ============================================================================
# COLLECTOR
# ============================================================================
class KANONCollector:

    BASE_URL = "https://api.openalex.org"
    EMAIL = "mailto=" + os.environ.get("OPENALEX_MAILTO", "anonymous@example.com")
    DELAY = 0.12          # ~8 req/s (polite pool permite 10/s)
    MAX_RETRIES = 3
    TIMEOUT = 15

    FIELD_WEIGHTS = {
        'Medicine':  {'wC': 0.15, 'wA': 0.60, 'wM': 0.10, 'wR': 0.05, 'wJ': 0.10, 'alpha': 0.7},
        'Physics':   {'wC': 0.20, 'wA': 0.40, 'wM': 0.20, 'wR': 0.05, 'wJ': 0.15, 'alpha': 0.4},
        'Chemistry': {'wC': 0.20, 'wA': 0.50, 'wM': 0.15, 'wR': 0.05, 'wJ': 0.10, 'alpha': 0.4},
        'Economics': {'wC': 0.30, 'wA': 0.30, 'wM': 0.15, 'wR': 0.10, 'wJ': 0.15, 'alpha': 0.4},
    }

    # ----------------------------------------------------------------
    # COMPLEXITY KEYWORDS: expandidos por categoria metodológica
    # Score 1-5: 1=descritivo, 2=analítico, 3=quantitativo,
    #            4=computacional/experimental avançado, 5=alta complexidade
    # ----------------------------------------------------------------
    COMPLEXITY_KEYWORDS_GENERAL = {
        # Experimental techniques
        'experimental': 4, 'in vivo': 5, 'in vitro': 4, 'in situ': 4,
        'assay': 3, 'bioassay': 4, 'immunoassay': 4, 'elisa': 4,
        'chromatography': 4, 'hplc': 5, 'mass spectrometry': 5,
        'spectroscopy': 4, 'nmr': 5, 'x-ray': 4, 'crystallography': 5,
        'microscopy': 3, 'electron microscopy': 5, 'confocal': 5,
        'flow cytometry': 4, 'electrophoresis': 3, 'western blot': 3,
        'pcr': 3, 'qpcr': 4, 'rt-pcr': 4, 'real-time pcr': 4,
        'sequencing': 4, 'next-generation sequencing': 5, 'ngs': 5,
        'whole genome': 5, 'rna-seq': 5, 'chip-seq': 5, 'atac-seq': 5,
        'single-cell': 5, 'proteomics': 5, 'metabolomics': 5,
        'transcriptomics': 5, 'genomics': 4, 'metagenomics': 5,
        'crispr': 5, 'gene editing': 5, 'cloning': 3, 'transfection': 3,
        'cell culture': 3, 'primary culture': 4, 'organoid': 5,
        'animal model': 4, 'mouse model': 4, 'knockout': 5,
        'transgenic': 5, 'clinical trial': 5, 'randomized': 4,
        'double-blind': 5, 'placebo': 4, 'cohort': 3, 'longitudinal': 4,
        'prospective': 3, 'retrospective': 2, 'cross-sectional': 2,
        'meta-analysis': 4, 'systematic review': 3,
        # Computational / mathematical
        'simulation': 4, 'computational': 4, 'algorithm': 4,
        'machine learning': 5, 'deep learning': 5, 'neural network': 5,
        'artificial intelligence': 5, 'reinforcement learning': 5,
        'natural language processing': 5, 'computer vision': 5,
        'random forest': 4, 'support vector': 4, 'gradient boosting': 4,
        'convolutional': 5, 'recurrent': 4, 'transformer': 5,
        'bayesian': 4, 'monte carlo': 5, 'markov chain': 4,
        'finite element': 5, 'molecular dynamics': 5, 'dft': 5,
        'density functional': 5, 'ab initio': 5, 'quantum': 4,
        'numerical': 3, 'optimization': 3, 'stochastic': 4,
        # Statistical methods
        'regression': 3, 'logistic regression': 3, 'linear regression': 2,
        'multivariate': 3, 'principal component': 4, 'pca': 3,
        'cluster analysis': 3, 'factor analysis': 3, 'anova': 3,
        'chi-square': 2, 'mann-whitney': 3, 'kruskal-wallis': 3,
        'kaplan-meier': 3, 'cox regression': 4, 'survival analysis': 4,
        'time series': 4, 'spatial analysis': 4, 'gis': 4,
        'network analysis': 4, 'graph theory': 4,
        # Physics/Chemistry specific
        'spectral analysis': 4, 'interferometry': 5, 'laser': 4,
        'superconductor': 5, 'semiconductor': 4, 'nanoparticle': 4,
        'nanotechnology': 5, 'synthesis': 3, 'catalysis': 4,
        'polymerization': 3, 'titration': 2, 'calorimetry': 3,
        'diffraction': 4, 'scattering': 4, 'neutron': 4,
        'synchrotron': 5, 'accelerator': 5, 'collider': 5,
        'detector': 3, 'sensor': 3, 'biosensor': 4,
        # Economics specific
        'econometric': 4, 'instrumental variable': 5, 'panel data': 3,
        'difference-in-difference': 5, 'regression discontinuity': 5,
        'general equilibrium': 5, 'game theory': 4, 'auction': 3,
        'experiment': 3, 'field experiment': 5, 'natural experiment': 4,
        'survey': 2, 'interview': 1, 'case study': 1, 'qualitative': 1,
        # Low complexity / descriptive
        'review': 1, 'overview': 1, 'commentary': 1, 'editorial': 1,
        'letter': 1, 'opinion': 1, 'perspective': 1, 'descriptive': 1,
    }

    # OpenAlex concept-level scores (concept.level indica profundidade)
    # Level 0 = campo amplo (score baixo), Level 3+ = muito específico (score alto)
    CONCEPT_LEVEL_SCORES = {0: 0.1, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9, 5: 1.0}

    FIELDNAMES = [
        'name', 'orcid', 'field', 'h_index', 'total_publications',
        'total_citations', 'avg_authors_per_paper', 'avg_author_position',
        'avg_complexity', 'complexity_keyword_score', 'complexity_concept_score',
        'complexity_n_works_analyzed', 'works_in_period',
        'top_works', 'journals', 'from_year', 'to_year',
        'KANON_C', 'KANON_A', 'KANON_M', 'KANON_R', 'KANON_J', 'KANON_Index'
    ]

    def __init__(self, output_dir='dados_reais', from_year=None, to_year=None, max_works=200):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        current_year = datetime.now().year
        self.from_year = from_year or (current_year - 20)
        self.to_year = to_year or current_year
        self.max_works = max_works
        print(f"[CONFIG] Período de análise: {self.from_year}-{self.to_year}")
        print(f"[CONFIG] Max works por autor: {self.max_works}")
        print(f"[CONFIG] Keywords de complexidade: {len(self.COMPLEXITY_KEYWORDS_GENERAL)}")

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------
    def _fetch_json(self, url, retry_count=0):
        try:
            full_url = url + ("&" if "?" in url else "?") + self.EMAIL
            req = urllib.request.Request(full_url)
            req.add_header('User-Agent', 'KANON-Index-Collector/2.0')
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if e.code == 429 and retry_count < self.MAX_RETRIES:
                time.sleep(2 ** (retry_count + 1))
                return self._fetch_json(url, retry_count + 1)
            elif retry_count < self.MAX_RETRIES:
                time.sleep(1)
                return self._fetch_json(url, retry_count + 1)
            return None
        except Exception:
            if retry_count < self.MAX_RETRIES:
                time.sleep(1)
                return self._fetch_json(url, retry_count + 1)
            return None

    # ------------------------------------------------------------------
    # AUTHOR LOOKUP
    # ------------------------------------------------------------------
    def _get_author_by_orcid(self, orcid):
        if not orcid:
            return None
        time.sleep(self.DELAY)
        url = self.BASE_URL + "/authors?filter=orcid:" + orcid
        data = self._fetch_json(url)
        if data and data.get('results') and len(data['results']) > 0:
            author = data['results'][0]
            if (author.get('works_count', 0) or 0) > 0:
                return author
        return None

    def _get_author_by_name(self, name, field):
        time.sleep(self.DELAY)
        clean = name.replace(".", " ").strip()
        encoded = urllib.parse.quote(clean)
        url = self.BASE_URL + "/authors?search=" + encoded + "&per_page=10"
        data = self._fetch_json(url)
        if not data or not data.get('results'):
            return None

        candidates = [c for c in data['results'] if (c.get('works_count', 0) or 0) > 3]
        if not candidates:
            return None

        best, best_score = None, -1
        for c in candidates:
            score = 0
            works = c.get('works_count', 0) or 0
            cit = c.get('cited_by_count', 0) or 0
            score += min(works / 100, 5)
            score += min(cit / 10000, 5)

            c_name = (c.get('display_name', '') or '').lower()
            parts = clean.lower().split()
            score += sum(2 for p in parts if p in c_name)

            for concept in (c.get('x_concepts', []) or []):
                if field.lower() in (concept.get('display_name', '') or '').lower():
                    score += 3
                    break

            if c.get('orcid'):
                score += 2

            if score > best_score:
                best_score = score
                best = c

        return best if best and best_score > 4 else None

    # ------------------------------------------------------------------
    # WORKS & METRICS (com filtro temporal e paginação)
    # ------------------------------------------------------------------
    def _get_author_works(self, author_id):
        """Busca works com filtro de período e paginação completa."""
        all_works = []
        per_page = 50
        cursor = '*'
        max_pages = self.max_works // per_page + 1

        for page in range(max_pages):
            time.sleep(self.DELAY)
            url = (self.BASE_URL + "/works?filter=authorships.author.id:" +
                   author_id +
                   ",publication_year:" + str(self.from_year) + "-" + str(self.to_year) +
                   "&sort=cited_by_count:desc&per_page=" + str(per_page) +
                   "&cursor=" + cursor)
            data = self._fetch_json(url)
            if not data or not data.get('results'):
                break
            all_works.extend(data['results'])
            if len(all_works) >= self.max_works:
                all_works = all_works[:self.max_works]
                break
            # Cursor pagination
            meta = data.get('meta', {})
            cursor = meta.get('next_cursor')
            if not cursor:
                break

        return all_works

    def _reconstruct_abstract(self, inverted_index):
        """Reconstrói texto do abstract a partir do inverted_index do OpenAlex."""
        if not inverted_index or not isinstance(inverted_index, dict):
            return ''
        word_positions = []
        for word, positions in inverted_index.items():
            if isinstance(positions, list):
                for pos in positions:
                    word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return ' '.join(w for _, w in word_positions)

    def _estimate_complexity_from_text(self, abstract_text):
        """Calcula complexidade a partir do abstract reconstruído.
        Retorna score 0.0-1.0 baseado em keywords encontradas."""
        if not abstract_text or len(abstract_text) < 20:
            return 0.0, 0  # score 0 = sem dados (NÃO 0.5 falso)
        text = abstract_text.lower()
        found_keywords = {}
        for kw, score in self.COMPLEXITY_KEYWORDS_GENERAL.items():
            if kw in text:
                found_keywords[kw] = score
        if not found_keywords:
            return 0.0, 0
        # Weighted score: soma dos scores / máximo teórico para N keywords
        total_score = sum(found_keywords.values())
        n_found = len(found_keywords)
        # Normalizar: máximo realista ~30 pontos (6 keywords de score 5)
        normalized = min(1.0, total_score / 30.0)
        return round(normalized, 4), n_found

    def _estimate_complexity_from_concepts(self, work):
        """Calcula complexidade a partir dos concepts/topics do OpenAlex.
        Concepts de nível alto (level 3+) indicam especificidade metodológica."""
        concepts = work.get('concepts', []) or []
        topics = work.get('topics', []) or []
        if not concepts and not topics:
            return 0.0
        scores = []
        for c in concepts:
            level = c.get('level', 0)
            conf = c.get('score', 0.5) if c.get('score') else 0.5
            level_score = self.CONCEPT_LEVEL_SCORES.get(level, 0.1)
            scores.append(level_score * conf)
        # Topics (OpenAlex v2) - profundidade implícita
        for t in topics:
            score_val = t.get('score', 0.5)
            # Subfield e topic indicam especificidade
            if t.get('subfield'):
                scores.append(0.6 * score_val)
            if t.get('domain'):
                scores.append(0.3 * score_val)
        if not scores:
            return 0.0
        return round(min(1.0, sum(sorted(scores, reverse=True)[:10]) / 5.0), 4)

    def _get_h_index(self, author_data):
        """Extrair h-index com fallbacks multiplos"""
        h = author_data.get('h_index', 0) or 0
        if h > 0:
            return h
        summary = author_data.get('summary_stats', {}) or {}
        h = summary.get('h_index', 0) or 0
        if h > 0:
            return h
        # Estimar de citations
        cit = author_data.get('cited_by_count', 0) or 0
        pubs = author_data.get('works_count', 0) or 0
        if cit > 0 and pubs > 0:
            return min(int(math.sqrt(cit)), pubs)
        return 0

    def _calculate_kanon(self, citations, avg_authors, avg_position,
                         avg_complexity, h_index, journals, field):
        w = self.FIELD_WEIGHTS.get(field, self.FIELD_WEIGHTS['Medicine'])
        c_norm = min(1.0, citations / 10000.0) if citations > 0 else 0
        a_norm = max(0, min(1.0, 1.0 - (avg_position / 20.0)))
        m_norm = avg_complexity
        r_norm = min(1.0, h_index / 100.0) if h_index > 0 else 0
        j_norm = min(1.0, len(set(journals)) / 50.0) if journals else 0

        kc = w['wC'] * c_norm
        ka = w['wA'] * a_norm
        km = w['wM'] * m_norm
        kr = w['wR'] * r_norm
        kj = w['wJ'] * j_norm
        ki = (kc + ka + km + kr + kj) ** w['alpha']

        return {
            'KANON_C': round(kc, 4), 'KANON_A': round(ka, 4),
            'KANON_M': round(km, 4), 'KANON_R': round(kr, 4),
            'KANON_J': round(kj, 4), 'KANON_Index': round(ki, 4)
        }

    # ------------------------------------------------------------------
    # PROCESS ONE AUTHOR (com complexidade real e filtro temporal)
    # ------------------------------------------------------------------
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
        author_positions = []
        keyword_scores = []
        concept_scores = []
        journals = []
        works_with_abstract = 0

        for work in works:
            n_auth = len(work.get('authorships', []))
            total_authors += n_auth

            # Posição do autor neste work
            for idx, a in enumerate(work.get('authorships', [])):
                if (a.get('author') or {}).get('id') == author_id:
                    author_positions.append(idx + 1)
                    break

            # === COMPLEXIDADE REAL ===
            # 1) Reconstruir abstract e extrair keywords
            inverted_index = work.get('abstract_inverted_index', {})
            abstract_text = self._reconstruct_abstract(inverted_index)
            # Também usar título como fonte de keywords
            title = work.get('title', '') or ''
            full_text = title + ' ' + abstract_text
            kw_score, n_kw = self._estimate_complexity_from_text(full_text)
            if kw_score > 0:
                keyword_scores.append(kw_score)
            if abstract_text and len(abstract_text) > 20:
                works_with_abstract += 1

            # 2) Complexidade por concepts/topics do OpenAlex
            concept_score = self._estimate_complexity_from_concepts(work)
            if concept_score > 0:
                concept_scores.append(concept_score)

            # Journal
            journal = (
                ((work.get('primary_location') or {}).get('source') or {}).get('display_name', '')
                or ((work.get('host_venue') or {}).get('display_name', ''))
            )
            if journal:
                journals.append(journal)

        if not works:
            return None

        avg_auth = total_authors / len(works)
        avg_pos = sum(author_positions) / len(author_positions) if author_positions else 1

        # === COMPLEXIDADE COMBINADA (keyword 60% + concept 40%) ===
        avg_kw = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0
        avg_concept = sum(concept_scores) / len(concept_scores) if concept_scores else 0.0

        if avg_kw > 0 and avg_concept > 0:
            avg_comp = 0.6 * avg_kw + 0.4 * avg_concept
        elif avg_kw > 0:
            avg_comp = avg_kw
        elif avg_concept > 0:
            avg_comp = avg_concept
        else:
            avg_comp = 0.0  # SEM DADOS = 0.0 (NÃO 0.5 falso)

        kanon = self._calculate_kanon(citations, avg_auth, avg_pos, avg_comp, h_index, journals, field)

        result = {
            'name': name, 'orcid': orcid, 'field': field,
            'h_index': h_index, 'total_publications': total_pubs,
            'total_citations': citations,
            'avg_authors_per_paper': round(avg_auth, 2),
            'avg_author_position': round(avg_pos, 2),
            'avg_complexity': round(avg_comp, 4),
            'complexity_keyword_score': round(avg_kw, 4),
            'complexity_concept_score': round(avg_concept, 4),
            'complexity_n_works_analyzed': works_with_abstract,
            'works_in_period': len(works),
            'top_works': len(works), 'journals': len(set(journals)),
            'from_year': self.from_year, 'to_year': self.to_year,
        }
        result.update(kanon)
        return result

    # ------------------------------------------------------------------
    # COLLECT
    # ------------------------------------------------------------------
    def collect(self, csv_path, output_file, label="researchers", use_name_fallback=True):
        """
        Coleta dados de um CSV de entrada e salva incrementalmente.
        Retoma automaticamente se output_file ja existe.
        """
        # Ler input
        with open(csv_path, 'r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        total = len(rows)

        # Ler ja coletados (para retomar)
        out_path = Path(self.output_dir) / output_file
        already_done = set()
        existing_results = []
        if out_path.exists():
            with open(str(out_path), 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    already_done.add(r.get('orcid', ''))
                    already_done.add(r.get('name', ''))
                    existing_results.append(r)
            print(f"[RESUME] {len(existing_results)} ja coletados, pulando...")

        stats = {'success': len(existing_results), 'failed': 0, 'skipped': 0,
                 'orcid': 0, 'name': 0, 'total': total}

        # Abrir arquivo em modo append
        write_header = not out_path.exists() or len(existing_results) == 0
        fout = open(str(out_path), 'a' if not write_header else 'w', newline='', encoding='utf-8')
        writer = csv.DictWriter(fout, fieldnames=self.FIELDNAMES)
        if write_header:
            writer.writeheader()
            # Reescrever existentes se estamos recriando
            for r in existing_results:
                writer.writerow({k: r.get(k, '') for k in self.FIELDNAMES})

        t0 = time.time()
        new_collected = 0

        for idx, row in enumerate(rows):
            name = row.get('name', 'Unknown')
            orcid = row.get('orcid', '')
            field = row.get('field', 'Unknown')
            safe = name.encode('ascii', 'replace').decode('ascii')[:40]

            # Pular se ja coletado
            if orcid in already_done or name in already_done:
                stats['skipped'] += 1
                continue

            pct = int(100.0 * (idx + 1) / total)
            elapsed = time.time() - t0
            rate = new_collected / elapsed if elapsed > 0 and new_collected > 0 else 0
            remaining = total - idx - 1
            eta = remaining / rate / 60 if rate > 0 else 0
            sys.stdout.write(f"\r[{pct:3d}%] {idx+1}/{total} - {safe} "
                           f"(ok={stats['success']}, fail={stats['failed']}, "
                           f"ETA={eta:.0f}min)    ")
            sys.stdout.flush()

            # Buscar no OpenAlex
            author = self._get_author_by_orcid(orcid)
            method = 'orcid'

            if not author and use_name_fallback:
                author = self._get_author_by_name(name, field)
                method = 'name'

            if author:
                result = self._process_author(author, name, orcid, field)
                if result:
                    writer.writerow(result)
                    fout.flush()  # flush incremental
                    stats['success'] += 1
                    new_collected += 1
                    if method == 'orcid':
                        stats['orcid'] += 1
                    else:
                        stats['name'] += 1
                    continue

            stats['failed'] += 1

        fout.close()

        # Sumario
        elapsed = time.time() - t0
        print(f"\n\n{'='*60}")
        print(f"{label.upper()} Collection Summary")
        print(f"{'='*60}")
        print(f"Total no CSV:     {stats['total']}")
        print(f"Ja coletados:     {stats['skipped']}")
        print(f"Novos coletados:  {new_collected}")
        print(f"Total com sucesso:{stats['success']}")
        print(f"  - via ORCID:    {stats['orcid']}")
        print(f"  - via nome:     {stats['name']}")
        print(f"Falhas:           {stats['failed']}")
        print(f"Tempo:            {elapsed/60:.1f} min")
        print(f"Arquivo:          {out_path}")
        print(f"{'='*60}")

        # Gerar summary
        self._write_summary(str(out_path), output_file.replace('.csv', '_summary.csv'))

        return stats

    def _write_summary(self, csv_path, summary_file):
        by_field = defaultdict(list)
        with open(csv_path, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                by_field[r['field']].append(r)

        summary_path = Path(self.output_dir) / summary_file
        with open(str(summary_path), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['Field', 'Count', 'Avg_h_index', 'Avg_Citations', 'Avg_KANON', 'Avg_Complexity'])
            for field in sorted(by_field.keys()):
                recs = by_field[field]
                n = len(recs)
                avg_h = sum(float(r.get('h_index', 0)) for r in recs) / n
                avg_c = sum(float(r.get('total_citations', 0)) for r in recs) / n
                avg_k = sum(float(r.get('KANON_Index', 0)) for r in recs) / n
                avg_x = sum(float(r.get('avg_complexity', 0)) for r in recs) / n
                w.writerow([field, n, round(avg_h, 2), round(avg_c, 2), round(avg_k, 4), round(avg_x, 3)])
        print(f"Summary: {summary_path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    current_year = datetime.now().year
    default_from = current_year - 20

    parser = argparse.ArgumentParser(description='KANON-Index Full Data Collector v2')
    parser.add_argument('--nobel-only', action='store_true', help='Coletar apenas Nobel')
    parser.add_argument('--researchers-only', action='store_true', help='Coletar apenas pesquisadores')
    parser.add_argument('--resume', action='store_true', help='Retomar coleta de onde parou')
    parser.add_argument('--output-dir', default='dados_reais', help='Diretorio base de saida')
    parser.add_argument('--from-year', type=int, default=default_from,
                        help=f'Ano inicial do periodo (default: {default_from})')
    parser.add_argument('--to-year', type=int, default=current_year,
                        help=f'Ano final do periodo (default: {current_year})')
    parser.add_argument('--max-works', type=int, default=200,
                        help='Max works por autor (default: 200)')
    args = parser.parse_args()

    date_tag = datetime.now().strftime('%Y-%m-%d')
    base_dir = args.output_dir
    dated_dir = os.path.join(base_dir, date_tag, 'coleta')
    Path(dated_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("KANON-Index Full Collector v2 — DADOS REAIS")
    print(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Periodo de analise: {args.from_year} - {args.to_year}")
    print(f"Max works/autor: {args.max_works}")
    print(f"Saida datada: {dated_dir}/")
    print(f"Copia ativa:  {base_dir}/  (usada pelo optimizer e analysis)")
    print("Features: h_index fix, nome fallback, complexidade REAL (abstract+concepts),")
    print("          paginacao completa, filtro temporal, salvamento incremental")
    print("=" * 70)

    # Collector saves to dated subdir
    collector = KANONCollector(
        output_dir=dated_dir,
        from_year=args.from_year,
        to_year=args.to_year,
        max_works=args.max_works
    )

    # --- NOBEL ---
    if not args.researchers_only:
        nobel_csv = Path('exampleNOBEL_orcids.csv')
        if nobel_csv.exists():
            print(f"\n[1/2] Coletando Nobel laureates ({nobel_csv})...")
            collector.collect(
                csv_path=str(nobel_csv),
                output_file='kanon_real_nobel.csv',
                label='Nobel Laureates',
                use_name_fallback=True
            )
        else:
            print(f"AVISO: {nobel_csv} nao encontrado, pulando Nobel")

    # --- RESEARCHERS ---
    if not args.nobel_only:
        res_csv = Path('example500_orcids.csv')
        if res_csv.exists():
            print(f"\n[2/2] Coletando 500 Researchers ({res_csv})...")
            collector.collect(
                csv_path=str(res_csv),
                output_file='kanon_real_researchers.csv',
                label='Researchers',
                use_name_fallback=True
            )
        else:
            print(f"AVISO: {res_csv} nao encontrado, pulando Researchers")

    # Copy final CSVs to base dados_reais/ for pipeline consumption
    import shutil
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    for fname in ['kanon_real_nobel.csv', 'kanon_real_researchers.csv']:
        src = os.path.join(dated_dir, fname)
        dst = os.path.join(base_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  [COPY] {src} -> {dst}")

    print("\n" + "=" * 60)
    print("COLETA COMPLETA!")
    print(f"Arquivo datado:  {dated_dir}/")
    print(f"Arquivo ativo:   {base_dir}/ (para optimizer/analysis)")
    print(f"Proximo passo:   python kanon_weight_optimizer_v2.py")
    print("=" * 60)


if __name__ == '__main__':
    main()
