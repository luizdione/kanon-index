"""
KANON-Index Data Collector - Web-optimized version
===================================================
Stripped-down version for web deployment on Hostinger.
Uses OpenAlex API (free, no auth required).
"""

import requests
import numpy as np
import re
import time
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class OpenAlexClient:
    """Client for OpenAlex API"""

    BASE_URL = "https://api.openalex.org"

    def __init__(self, email: Optional[str] = None):
        self.email = email
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'KANON-Index-WebApp/2.0 (mailto:kanon-index@research.org)'
        })
        if email:
            self.session.params = {'mailto': email}

    def search_author(self, name: str) -> List[Dict]:
        """Search for author by name"""
        url = f"{self.BASE_URL}/authors"
        params = {
            'filter': f'display_name.search:{name}',
            'per-page': 10
        }
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data.get('results', [])
        except Exception as e:
            logger.error(f"Error searching author: {e}")
            return []

    def get_author_by_orcid(self, orcid: str) -> Optional[Dict]:
        """Get author by ORCID"""
        url = f"{self.BASE_URL}/authors"
        params = {'filter': f'orcid:{orcid}'}
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            results = data.get('results', [])
            return results[0] if results else None
        except Exception as e:
            logger.error(f"Error getting author by ORCID: {e}")
            return None

    def get_author_works(self, author_id: str, per_page: int = 200, years_limit: int = 10) -> List[Dict]:
        """Get works for an author, limited to recent years."""
        works = []
        cursor = '*'
        from datetime import datetime
        current_year = datetime.now().year
        min_year = current_year - years_limit if years_limit > 0 else 1900

        while True:
            url = f"{self.BASE_URL}/works"
            filter_str = f'author.id:{author_id}'
            if years_limit > 0:
                filter_str += f',publication_year:{min_year}-{current_year}'

            params = {
                'filter': filter_str,
                'per-page': per_page,
                'cursor': cursor
            }

            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                results = data.get('results', [])
                if not results:
                    break
                works.extend(results)
                meta = data.get('meta', {})
                cursor = meta.get('next_cursor')
                if not cursor:
                    break
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error getting works: {e}")
                break

        return works


class MethodComplexityAnalyzer:
    """Analyze methodological complexity from abstracts."""

    TECHNIQUE_SCORES = {
        'pcr': 1, 'qpcr': 2, 'rt-pcr': 2, 'real-time pcr': 2,
        'cloning': 3, 'western blot': 3, 'northern blot': 3, 'southern blot': 3,
        'elisa': 2, 'flow cytometry': 4, 'facs': 4,
        'immunofluorescence': 3, 'immunohistochemistry': 3, 'ihc': 3,
        'chip-seq': 6, 'rna-seq': 5, 'rna sequencing': 5,
        'single-cell': 8, 'single cell sequencing': 8, 'scrna-seq': 8,
        'crispr': 7, 'crispr-cas9': 7, 'genome editing': 7,
        'mass spectrometry': 6, 'proteomics': 6, 'metabolomics': 6,
        'cryo-em': 9, 'cryo-electron microscopy': 9,
        'x-ray crystallography': 8, 'nmr': 7, 'nuclear magnetic resonance': 7,
        'clinical trial': 10, 'randomized controlled trial': 10, 'rct': 10,
        'cohort study': 5, 'case-control': 4, 'meta-analysis': 6,
        'mouse model': 4, 'transgenic': 5, 'knockout': 5,
        'zebrafish': 4, 'drosophila': 4, 'c. elegans': 4,
        'mri': 5, 'fmri': 6, 'pet scan': 6, 'confocal microscopy': 5,
        'two-photon': 7, 'super-resolution': 8,
        'machine learning': 4, 'deep learning': 5, 'neural network': 5,
        'genome-wide association': 6, 'gwas': 6, 'molecular dynamics': 6,
        'bioinformatics': 4,
        'particle accelerator': 9, 'laser spectroscopy': 6, 'quantum': 7,
        'synchrotron': 8, 'neutron scattering': 7,
        'synthesis': 3, 'organic synthesis': 4, 'total synthesis': 7,
        'catalysis': 5, 'electrochemistry': 4,
    }

    PURPOSE_MARKERS = [
        'for genotyping', 'to measure', 'to quantify', 'to detect',
        'to analyze', 'for expression', 'for screening', 'to validate',
        'measuring', 'quantifying', 'detecting', 'analyzing'
    ]

    WORKFLOW_SEPARATORS = [
        'separately', 'independently', 'additionally', 'in parallel',
        'for validation', 'for screening', 'for confirmation',
        'initial', 'subsequent', 'follow-up', 'validation cohort'
    ]

    def __init__(self):
        self.technique_pattern = self._build_pattern()
        self.purpose_pattern = self._build_purpose_pattern()
        self.separator_pattern = self._build_separator_pattern()

    def _build_pattern(self) -> re.Pattern:
        techniques = sorted(self.TECHNIQUE_SCORES.keys(), key=len, reverse=True)
        pattern = '|'.join(re.escape(t) for t in techniques)
        return re.compile(pattern, re.IGNORECASE)

    def _build_purpose_pattern(self) -> re.Pattern:
        markers = sorted(self.PURPOSE_MARKERS, key=len, reverse=True)
        return re.compile('|'.join(re.escape(m) for m in markers), re.IGNORECASE)

    def _build_separator_pattern(self) -> re.Pattern:
        separators = sorted(self.WORKFLOW_SEPARATORS, key=len, reverse=True)
        return re.compile('|'.join(re.escape(s) for s in separators), re.IGNORECASE)

    def calculate_complexity(self, text: str) -> float:
        if not text:
            return 0.0
        technique_workflows = self._extract_techniques_with_context(text)
        if not technique_workflows:
            return 0.0
        technique_counts = {t: len(w) for t, w in technique_workflows.items()}
        total_score = sum(
            self.TECHNIQUE_SCORES.get(t.lower(), 0) * c
            for t, c in technique_counts.items()
        )
        max_possible = sum(10 * c for c in technique_counts.values())
        if max_possible == 0:
            return 0.0
        return min(total_score / max_possible, 1.0)

    def _extract_techniques_with_context(self, text: str) -> Dict[str, Set[str]]:
        if not text:
            return {}
        paragraphs = re.split(r'\n\s*\n', text)
        technique_workflows = defaultdict(set)
        for para_idx, paragraph in enumerate(paragraphs):
            sentences = re.split(r'[.!?]+', paragraph)
            for sentence in sentences:
                if not sentence.strip():
                    continue
                matches = self.technique_pattern.findall(sentence.lower())
                for technique in matches:
                    sig = self._context_signature(sentence, technique, para_idx)
                    technique_workflows[technique].add(sig)
        return technique_workflows

    def _context_signature(self, sentence: str, technique: str, para_idx: int) -> str:
        sl = sentence.lower()
        purposes = self.purpose_pattern.findall(sl)
        separators = self.separator_pattern.findall(sl)
        words = sentence.split()
        targets = [w for w in words if w and w[0].isupper() and len(w) > 3]
        sig = '#'.join(filter(None, [
            technique,
            '|'.join(sorted(purposes)),
            '|'.join(sorted(separators)),
            '|'.join(sorted(targets)[:3])
        ]))
        return f"{para_idx}:{hashlib.md5(sig.encode()).hexdigest()}"


class KanonIndexCalculator:
    """Calculate the KANON-Index for researchers"""

    # Versao atual: 4 componentes (C, A, S, J). M e R unificados/eliminados.
    DEFAULT_WEIGHTS = {
        'citations': 0.25, 'authorship': 0.39,
        'sophistication': 0.22, 'journal': 0.14
    }

    POSITION_WEIGHTS = {
        'first': 1.0, 'co_first': 0.9, 'corresponding': 0.9,
        'last': 0.8, 'middle': 0.4, 'single': 1.2
    }

    def __init__(self, weights=None, alpha=0.5, beta=0.29, s_concepts=0.70):
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.alpha = alpha
        self.beta = beta              # peso de S_text dentro de S = b*S_text + (1-b)*S_concepts
        self.s_concepts = s_concepts  # baseline de sofisticacao por area (~constante no campo)
        self.complexity_analyzer = MethodComplexityAnalyzer()

    def calculate_h_index(self, citation_counts):
        if not citation_counts:
            return 0
        sorted_c = sorted(citation_counts, reverse=True)
        h = 0
        for i, c in enumerate(sorted_c, 1):
            if c >= i:
                h = i
            else:
                break
        return h

    def calculate_authorship_score(self, position, total_authors, alpha=0.8):
        pw = self.POSITION_WEIGHTS.get(position, 0.4)
        return min(pw / (total_authors ** alpha), 1.0)

    def estimate_cost(self, complexity_score, index=0):
        """Deterministic cost estimation based on complexity.
        Uses index-based perturbation instead of random noise for reproducibility."""
        base = complexity_score * 0.7
        perturbation = ((index % 7) - 3) * 0.02
        return float(np.clip(base + perturbation, 0, 1))

    def normalize_citations(self, citations, field_median=50.0):
        if citations <= 0:
            return 0.0
        log_c = np.log1p(citations)
        log_m = np.log1p(field_median)
        return min(log_c / (log_m * 2), 1.0)

    def normalize_journal_impact(self, impact_factor):
        if impact_factor <= 0:
            return 0.0
        return min(np.log1p(impact_factor) / (np.log1p(7) * 2), 1.0)

    def _get_journal_score(self, work):
        """Extract journal impact score from OpenAlex work data."""
        primary_loc = work.get('primary_location', {}) or {}
        source = primary_loc.get('source', {}) or {}
        if not source:
            return 0.3

        source_type = source.get('type', '')
        if source_type != 'journal':
            return 0.2

        # Use 2-year mean citedness as impact factor proxy
        summary = source.get('summary_stats', {}) or {}
        impact_factor = summary.get('2yr_mean_citedness', 0)
        if impact_factor and impact_factor > 0:
            return self.normalize_journal_impact(impact_factor)

        # Fallback: estimate from journal size
        works_count = source.get('works_count', 0)
        if works_count > 50000:
            return 0.7
        if works_count > 10000:
            return 0.5
        if works_count > 1000:
            return 0.4
        return 0.3

    def calculate_kanon_index(self, works, author_id):
        if not works:
            return {
                'kanon_index': 0.0, 'h_index': 0, 'total_citations': 0,
                'total_papers': 0, 'avg_complexity': 0.0, 'avg_authors': 0.0,
                'first_author_fraction': 0.0
            }

        citation_counts = []
        authorship_scores = []
        complexity_scores = []   # S_text (marcadores do abstract)
        journal_scores = []
        first_author_count = 0

        for wi, work in enumerate(works):
            citations = work.get('cited_by_count', 0)
            citation_counts.append(citations)

            authorships = work.get('authorships', [])
            total_authors = len(authorships)

            author_position = 'middle'
            for i, authorship in enumerate(authorships):
                author_info = authorship.get('author', {})
                aid = author_info.get('id', '').replace('https://openalex.org/', '')
                check_id = author_id.replace('https://openalex.org/', '')
                if aid == check_id:
                    if i == 0:
                        author_position = 'first' if total_authors > 1 else 'single'
                        first_author_count += 1
                    elif i == total_authors - 1 and total_authors > 1:
                        author_position = 'last'
                    if authorship.get('is_corresponding', False):
                        author_position = 'corresponding'
                    break

            auth_score = self.calculate_authorship_score(author_position, max(total_authors, 1), alpha=self.alpha)
            authorship_scores.append(auth_score)

            abstract = work.get('abstract_inverted_index', {})
            if abstract:
                words = []
                for word, positions in abstract.items():
                    for pos in positions:
                        words.append((pos, word))
                words.sort()
                abstract_text = ' '.join(w[1] for w in words)
            else:
                abstract_text = work.get('display_name', '')

            complexity = self.complexity_analyzer.calculate_complexity(abstract_text)
            complexity_scores.append(complexity)

            # Extract journal impact from OpenAlex source data
            journal_score = self._get_journal_score(work)
            journal_scores.append(journal_score)

        h_index = self.calculate_h_index(citation_counts)
        total_citations = sum(citation_counts)
        avg_citations = float(np.mean(citation_counts)) if citation_counts else 0

        C = float(np.mean([self.normalize_citations(c) for c in citation_counts]))
        A = float(np.mean(authorship_scores))
        S_text = float(np.mean(complexity_scores))
        S = self.beta * S_text + (1.0 - self.beta) * self.s_concepts
        J = float(np.mean(journal_scores))

        kanon_score = (
            self.weights['citations'] * C +
            self.weights['authorship'] * A +
            self.weights['sophistication'] * S +
            self.weights['journal'] * J
        )

        kanon_index = kanon_score * 100

        return {
            'kanon_index': round(kanon_index, 2),
            'h_index': h_index,
            'total_citations': total_citations,
            'avg_citations': round(avg_citations, 2),
            'total_papers': len(works),
            'avg_complexity': round(S_text, 3),
            'avg_sophistication': round(S, 3),
            's_concepts': round(self.s_concepts, 3),
            'beta': round(self.beta, 3),
            'avg_authors': round(float(np.mean([len(w.get('authorships', [])) for w in works])), 2),
            'first_author_fraction': round(first_author_count / len(works), 3),
            'component_citations': round(C, 3),
            'component_authorship': round(A, 3),
            'component_sophistication': round(S, 3),
            'component_journal': round(J, 3)
        }
