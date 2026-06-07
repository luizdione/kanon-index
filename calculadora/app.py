#!/usr/bin/env python3
"""
KANON-Index Web Calculator - Hostinger Deployment
Flask application for calculating KANON-Index with researcher validation.

Features:
- Search by name or ORCID
- Field selection (Chemistry, Medicine, Physics, Economics)
- Customizable weights
- Researcher validation (shows institution before calculating)
- Progress tracking via SSE
- SQLite caching
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import sys
import os
import json
import sqlite3
import re
import time
import threading
import queue
from datetime import datetime
import logging

# Add parent directory for kanon_data_collector
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kanon_data_collector import (
    OpenAlexClient,
    KanonIndexCalculator,
    MethodComplexityAnalyzer
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
DATABASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'kanon_cache.db')
CACHE_EXPIRY_DAYS = 7

# Pesos otimizados (v4: 4 componentes C, A, S, J; SLSQP + 5-Fold CV; AUC-ROC)
# Benchmark casado por campo. Source: dados_reais/kanon_optimized_weights_v4.json
FIELD_WEIGHTS = {
    'chemistry': {
        'citations': 0.0545, 'authorship': 0.4099,
        'sophistication': 0.4111, 'journal': 0.1245
    },
    'medicine': {
        'citations': 0.4267, 'authorship': 0.3462,
        'sophistication': 0.0492, 'journal': 0.1780
    },
    'physics': {
        'citations': 0.0341, 'authorship': 0.4585,
        'sophistication': 0.3577, 'journal': 0.1497
    },
    'economics': {
        'citations': 0.4664, 'authorship': 0.3628,
        'sophistication': 0.0516, 'journal': 0.1192
    },
    'default': {
        'citations': 0.2454, 'authorship': 0.3944,
        'sophistication': 0.2174, 'journal': 0.1428
    }
}

# alpha (penalidade de hiperautoria) por campo — v4
FIELD_ALPHA = {
    'chemistry': 0.4349,
    'medicine': 0.4767,
    'physics': 0.4349,
    'economics': 0.6886,
    'default': 0.500
}

# beta = peso de S_text dentro de S = beta*S_text + (1-beta)*S_concepts — v4
FIELD_BETA = {
    'chemistry': 0.3046,
    'medicine': 0.2474,
    'physics': 0.3086,
    'economics': 0.3012,
    'default': 0.290
}

# S_concepts ~ constante dentro do campo (baseline de sofisticacao por area).
# Valores representativos do escore medio de subfield observado por area.
FIELD_SCONCEPTS = {
    'chemistry': 0.77,
    'medicine': 0.74,
    'physics': 0.74,
    'economics': 0.62,
    'default': 0.70
}

# ── Database ──────────────────────────────────────────────────────────────

def get_db():
    db = sqlite3.connect(DATABASE)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    db = None
    try:
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS calculations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                orcid TEXT,
                openalex_id TEXT,
                researcher_name TEXT,
                institution TEXT,
                field TEXT,
                kanon_index REAL,
                h_index INTEGER,
                total_citations INTEGER,
                total_papers INTEGER,
                weights_json TEXT,
                result_json TEXT,
                calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.execute('CREATE INDEX IF NOT EXISTS idx_orcid ON calculations(orcid)')
        db.execute('CREATE INDEX IF NOT EXISTS idx_openalex ON calculations(openalex_id)')
        db.commit()
    finally:
        if db:
            db.close()

# ── Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/weights/<field>')
def get_weights(field):
    """Return default weights for a field."""
    weights = FIELD_WEIGHTS.get(field, FIELD_WEIGHTS['default'])
    return jsonify({'success': True, 'weights': weights, 'field': field})

@app.route('/api/search', methods=['POST'])
def search_researcher():
    """
    Search for researcher by name or ORCID.
    Returns list of candidates with institution for validation.
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Dados inválidos'}), 400

    search_type = data.get('type', 'name')  # 'name' or 'orcid'
    query = data.get('query', '').strip()

    if not query:
        return jsonify({'success': False, 'error': 'Campo de busca vazio'}), 400

    client = OpenAlexClient(email=os.environ.get('OPENALEX_MAILTO', 'anonymous@example.com'))

    try:
        if search_type == 'orcid':
            # Validate ORCID format
            if not re.match(r'^\d{4}-\d{4}-\d{4}-\d{3}[0-9X]$', query):
                return jsonify({
                    'success': False,
                    'error': 'Formato ORCID inválido. Use: 0000-0000-0000-0000'
                }), 400

            author = client.get_author_by_orcid(query)
            if not author:
                return jsonify({
                    'success': False,
                    'error': f'Nenhum pesquisador encontrado com ORCID: {query}'
                }), 404

            candidates = [_format_author(author)]

        else:
            # Name search
            if len(query) < 3:
                return jsonify({
                    'success': False,
                    'error': 'Digite pelo menos 3 caracteres'
                }), 400

            authors = client.search_author(query)
            if not authors:
                return jsonify({
                    'success': False,
                    'error': f'Nenhum pesquisador encontrado para: {query}'
                }), 404

            candidates = [_format_author(a) for a in authors[:10]]

        return jsonify({
            'success': True,
            'candidates': candidates,
            'total': len(candidates)
        })

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({
            'success': False,
            'error': f'Erro na busca: {str(e)}'
        }), 500


@app.route('/api/calculate', methods=['POST'])
def calculate_kanon():
    """
    Calculate KANON-Index for a validated researcher.
    Uses Server-Sent Events for progress reporting.
    """
    data = request.get_json()
    if not data:
        return jsonify({'success': False, 'error': 'Dados inválidos'}), 400

    author_id = data.get('author_id', '').strip()
    field = data.get('field', 'default')
    custom_weights = data.get('weights')
    years_limit = int(data.get('years_limit', 10))

    if not author_id:
        return jsonify({'success': False, 'error': 'ID do pesquisador não fornecido'}), 400

    # Determine weights
    if custom_weights:
        weights = {
            'citations': float(custom_weights.get('citations', 0.25)),
            'authorship': float(custom_weights.get('authorship', 0.39)),
            'sophistication': float(custom_weights.get('sophistication', 0.22)),
            'journal': float(custom_weights.get('journal', 0.14))
        }
        # Normalize to sum = 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
    else:
        weights = FIELD_WEIGHTS.get(field, FIELD_WEIGHTS['default'])

    # Check cache
    cached = _get_cached(author_id, json.dumps(weights, sort_keys=True))
    if cached:
        return jsonify({
            'success': True,
            'data': cached,
            'cached': True,
            'message': f'Resultado em cache de {cached["calculated_at"]}'
        })

    try:
        client = OpenAlexClient(email=os.environ.get('OPENALEX_MAILTO', 'anonymous@example.com'))

        # Get author info
        openalex_id = author_id
        if not author_id.startswith('https://'):
            openalex_id = f'https://openalex.org/{author_id}'

        # Fetch works
        short_id = author_id.replace('https://openalex.org/', '')
        works = client.get_author_works(short_id, years_limit=years_limit)

        if not works:
            return jsonify({
                'success': False,
                'error': 'Nenhuma publicação encontrada para este pesquisador'
            }), 404

        # Calculate KANON with field-specific alpha, beta and S_concepts baseline
        alpha = FIELD_ALPHA.get(field, FIELD_ALPHA['default'])
        beta = FIELD_BETA.get(field, FIELD_BETA['default'])
        s_concepts = FIELD_SCONCEPTS.get(field, FIELD_SCONCEPTS['default'])
        calculator = KanonIndexCalculator(weights=weights, alpha=alpha, beta=beta, s_concepts=s_concepts)
        metrics = calculator.calculate_kanon_index(works, short_id)

        # Get author details from first work
        author_name = 'Desconhecido'
        institution = 'Desconhecida'
        orcid_val = None

        # Try to get info from OpenAlex directly
        try:
            resp = client.session.get(
                f"{client.BASE_URL}/authors/{short_id}", timeout=15
            )
            if resp.status_code == 200:
                author_data = resp.json()
                author_name = author_data.get('display_name', author_name)
                # Use last_known_institutions (new API) with fallback
                institutions = author_data.get('last_known_institutions', [])
                if institutions:
                    institution = institutions[0].get('display_name', institution)
                else:
                    inst = author_data.get('last_known_institution', {})
                    institution = inst.get('display_name', institution) if inst else institution
                orcid_val = author_data.get('orcid')
        except Exception:
            pass

        result = {
            'researcher_name': author_name,
            'institution': institution,
            'orcid': orcid_val,
            'openalex_id': short_id,
            'field': field,
            'weights': weights,
            'kanon_index': round(metrics.get('kanon_index', 0), 2),
            'h_index': metrics.get('h_index', 0),
            'total_citations': metrics.get('total_citations', 0),
            'avg_citations': metrics.get('avg_citations', 0),
            'total_papers': metrics.get('total_papers', 0),
            'avg_complexity': metrics.get('avg_complexity', 0),
            'avg_sophistication': metrics.get('avg_sophistication', 0),
            's_concepts': metrics.get('s_concepts', 0),
            'beta': metrics.get('beta', 0),
            'avg_authors': metrics.get('avg_authors', 0),
            'first_author_fraction': metrics.get('first_author_fraction', 0),
            'component_citations': metrics.get('component_citations', 0),
            'component_authorship': metrics.get('component_authorship', 0),
            'component_sophistication': metrics.get('component_sophistication', 0),
            'component_journal': metrics.get('component_journal', 0),
            'calculated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # Cache result
        _cache_result(result)

        return jsonify({
            'success': True,
            'data': result,
            'cached': False
        })

    except Exception as e:
        logger.error(f"Calculation error: {e}")
        return jsonify({
            'success': False,
            'error': f'Erro no cálculo: {str(e)}'
        }), 500


# ── Helpers ───────────────────────────────────────────────────────────────

def _format_author(author):
    """Format OpenAlex author record for frontend."""
    # Use last_known_institutions (new API) with fallback to last_known_institution (deprecated)
    institutions = author.get('last_known_institutions', [])
    if institutions:
        inst = institutions[0]
    else:
        inst = author.get('last_known_institution', {})

    inst_name = inst.get('display_name', 'Instituição desconhecida') if inst else 'Instituição desconhecida'
    inst_country = inst.get('country_code', '') if inst else ''

    # Get research areas: use topics (new API) with fallback to x_concepts (deprecated)
    topics_data = author.get('topics', []) or author.get('x_concepts', [])
    topics = [t.get('display_name', '') for t in topics_data[:3]]

    return {
        'id': author.get('id', '').replace('https://openalex.org/', ''),
        'name': author.get('display_name', 'Desconhecido'),
        'orcid': author.get('orcid', ''),
        'institution': inst_name,
        'country': inst_country,
        'works_count': author.get('works_count', 0),
        'cited_by_count': author.get('cited_by_count', 0),
        'topics': topics
    }


def _get_cached(author_id, weights_key):
    """Check cache for existing result."""
    db = None
    try:
        db = get_db()
        cursor = db.execute('''
            SELECT result_json, calculated_at FROM calculations
            WHERE openalex_id = ? AND weights_json = ?
            AND calculated_at >= datetime('now', '-' || ? || ' days')
            ORDER BY calculated_at DESC LIMIT 1
        ''', (author_id.replace('https://openalex.org/', ''), weights_key, CACHE_EXPIRY_DAYS))
        row = cursor.fetchone()
        if row:
            result = json.loads(row['result_json'])
            result['calculated_at'] = row['calculated_at']
            return result
    except Exception as e:
        logger.error(f"Cache read error: {e}")
    finally:
        if db:
            db.close()
    return None


def _cache_result(result):
    """Cache calculation result."""
    db = None
    try:
        db = get_db()
        db.execute('''
            INSERT INTO calculations
            (orcid, openalex_id, researcher_name, institution, field,
             kanon_index, h_index, total_citations, total_papers,
             weights_json, result_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            result.get('orcid', ''),
            result.get('openalex_id', ''),
            result.get('researcher_name', ''),
            result.get('institution', ''),
            result.get('field', ''),
            result.get('kanon_index', 0),
            result.get('h_index', 0),
            result.get('total_citations', 0),
            result.get('total_papers', 0),
            json.dumps(result.get('weights', {}), sort_keys=True),
            json.dumps(result)
        ))
        db.commit()
    except Exception as e:
        logger.error(f"Cache write error: {e}")
    finally:
        if db:
            db.close()


# ── Error Handlers ────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Recurso não encontrado'}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'success': False, 'error': 'Erro interno do servidor'}), 500

@app.errorhandler(429)
def rate_limited(e):
    return jsonify({'success': False, 'error': 'Muitas requisições. Tente novamente em alguns segundos.'}), 429


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)
    app.run(host='0.0.0.0', port=5000, debug=True)
