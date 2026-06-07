/**
 * KANON-Index Calculator - Client-Side Engine
 * =============================================
 * Chama a API OpenAlex diretamente do navegador.
 * Não precisa de backend Python/Flask.
 *
 * OpenAlex API: https://docs.openalex.org/
 * - Gratuita, sem autenticação
 * - Suporta CORS (chamadas do browser)
 */

// ── Configuração ──────────────────────────────────────────────────────

const OPENALEX_BASE = 'https://api.openalex.org';
const POLITE_EMAIL = 'anonymous@example.com'; // OpenAlex polite pool — troque pelo seu e-mail

// Pesos otimizados (v4: 4 componentes C, A, S, J; SLSQP + 5-Fold CV; benchmark casado por campo)
const FIELD_WEIGHTS = {
    chemistry:  { C: 0.0545, A: 0.4099, S: 0.4111, J: 0.1245 },
    medicine:   { C: 0.4267, A: 0.3462, S: 0.0492, J: 0.1780 },
    physics:    { C: 0.0341, A: 0.4585, S: 0.3577, J: 0.1497 },
    economics:  { C: 0.4664, A: 0.3628, S: 0.0516, J: 0.1192 },
    default:    { C: 0.2454, A: 0.3944, S: 0.2174, J: 0.1428 }
};

// alpha (penalidade de hiperautoria) por campo — v4
const FIELD_ALPHA = {
    chemistry: 0.4349, medicine: 0.4767, physics: 0.4349,
    economics: 0.6886, default: 0.500
};

// beta = peso de S_text dentro de S = beta*S_text + (1-beta)*S_concepts — v4
const FIELD_BETA = {
    chemistry: 0.3046, medicine: 0.2474, physics: 0.3086,
    economics: 0.3012, default: 0.290
};

// S_concepts ~ constante por campo (baseline de sofisticacao por area)
const FIELD_SCONCEPTS = {
    chemistry: 0.77, medicine: 0.74, physics: 0.74,
    economics: 0.62, default: 0.70
};

// Pesos de posição de autoria
const POSITION_WEIGHTS = {
    single: 1.2,
    first: 1.0,
    corresponding: 0.9,
    last: 0.8,
    middle: 0.4
};

// Técnicas experimentais e scores de complexidade (0-10)
const TECHNIQUE_SCORES = {
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
    'catalysis': 5, 'electrochemistry': 4
};

// Build regex for technique detection
const TECHNIQUE_REGEX = new RegExp(
    Object.keys(TECHNIQUE_SCORES)
        .sort((a, b) => b.length - a.length)
        .map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        .join('|'),
    'gi'
);

// ── State ─────────────────────────────────────────────────────────────

let selectedField = null;
let searchType = 'name';
let selectedAuthor = null;
let lastResult = null;
let currentWeights = null;
let selectedPeriodYears = 10;

// ── OpenAlex API ──────────────────────────────────────────────────────

async function apiCall(endpoint, params = {}, retries = 2) {
    params.mailto = POLITE_EMAIL;
    const query = new URLSearchParams(params).toString();
    const url = `${OPENALEX_BASE}${endpoint}?${query}`;

    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            const resp = await fetch(url);
            if (resp.status === 429) {
                // Rate limited - wait and retry
                const waitMs = Math.min(1000 * Math.pow(2, attempt), 5000);
                await sleep(waitMs);
                continue;
            }
            if (!resp.ok) throw new Error(`API error: ${resp.status}`);
            return resp.json();
        } catch (err) {
            if (attempt === retries) throw err;
            await sleep(1000 * (attempt + 1));
        }
    }
}

async function searchAuthorByName(name) {
    const data = await apiCall('/authors', {
        'filter': `display_name.search:${name}`,
        'per-page': 10
    });
    return data.results || [];
}

async function searchAuthorByOrcid(orcid) {
    const data = await apiCall('/authors', {
        'filter': `orcid:${orcid}`
    });
    return data.results || [];
}

async function getAuthorWorks(authorId, onProgress) {
    const works = [];
    let cursor = '*';
    const currentYear = new Date().getFullYear();
    const yearsLimit = selectedPeriodYears;
    const minYear = yearsLimit > 0 ? currentYear - yearsLimit : 1900;
    let page = 0;

    while (true) {
        page++;
        if (onProgress) onProgress(`Buscando publicações (página ${page})...`, Math.min(20 + page * 10, 70));

        const data = await apiCall('/works', {
            'filter': `author.id:${authorId},publication_year:${minYear}-${currentYear}`,
            'per-page': 200,
            'cursor': cursor
        });

        const results = data.results || [];
        if (results.length === 0) break;

        works.push(...results);

        const nextCursor = data.meta?.next_cursor;
        if (!nextCursor) break;
        cursor = nextCursor;

        // Pequena pausa para ser educado com a API
        await sleep(100);
    }

    return works;
}

// ── KANON Calculation ─────────────────────────────────────────────────

function reconstructAbstract(invertedIndex) {
    if (!invertedIndex) return '';
    const words = [];
    for (const [word, positions] of Object.entries(invertedIndex)) {
        for (const pos of positions) {
            words.push([pos, word]);
        }
    }
    words.sort((a, b) => a[0] - b[0]);
    return words.map(w => w[1]).join(' ');
}

function calculateComplexity(text) {
    if (!text) return 0;
    const lower = text.toLowerCase();
    const matches = lower.match(TECHNIQUE_REGEX) || [];

    // Deduplicate by technique name
    const uniqueTechniques = [...new Set(matches.map(m => m.toLowerCase()))];

    if (uniqueTechniques.length === 0) return 0;

    let totalScore = 0;
    let maxPossible = 0;
    for (const tech of uniqueTechniques) {
        totalScore += TECHNIQUE_SCORES[tech] || 0;
        maxPossible += 10;
    }

    return maxPossible > 0 ? Math.min(totalScore / maxPossible, 1.0) : 0;
}

function normalizeCitations(citations, fieldMedian = 50) {
    if (citations <= 0) return 0;
    const logC = Math.log1p(citations);
    const logM = Math.log1p(fieldMedian);
    return Math.min(logC / (logM * 2), 1.0);
}

function getJournalScore(work) {
    // Extract journal impact from OpenAlex source data
    const source = work.primary_location?.source;
    if (!source) return 0.3; // No source = low score

    // Use cited_by_count percentile from source if available
    // Otherwise estimate from source type
    const sourceType = source.type || '';
    const isJournal = sourceType === 'journal';

    if (!isJournal) return 0.2; // Non-journal sources get low score

    // Use the 2-year impact factor proxy from OpenAlex (summary_stats)
    const impactFactor = source.summary_stats?.['2yr_mean_citedness'] || 0;
    if (impactFactor > 0) {
        return normalizeJournalImpact(impactFactor);
    }

    // Fallback: use works_count as a rough proxy for journal prestige
    const worksCount = source.works_count || 0;
    if (worksCount > 50000) return 0.7;
    if (worksCount > 10000) return 0.5;
    if (worksCount > 1000) return 0.4;
    return 0.3;
}

function normalizeJournalImpact(impactFactor) {
    if (impactFactor <= 0) return 0;
    return Math.min(Math.log1p(impactFactor) / (Math.log1p(7) * 2), 1.0);
}

function calculateAuthorshipScore(position, totalAuthors, alpha = 0.8) {
    const pw = POSITION_WEIGHTS[position] || 0.4;
    return Math.min(pw / Math.pow(Math.max(totalAuthors, 1), alpha), 1.0);
}

function estimateCost(complexity, index = 0) {
    // Deterministic cost estimation (no random noise for reproducibility)
    const base = complexity * 0.7;
    // Use a small deterministic perturbation based on index to avoid all-identical values
    const perturbation = ((index % 7) - 3) * 0.02;
    return Math.max(0, Math.min(1, base + perturbation));
}

function calculateHIndex(citationCounts) {
    if (!citationCounts.length) return 0;
    const sorted = [...citationCounts].sort((a, b) => b - a);
    let h = 0;
    for (let i = 0; i < sorted.length; i++) {
        if (sorted[i] >= i + 1) h = i + 1;
        else break;
    }
    return h;
}

function calculateKanon(works, authorId, weights, field) {
    const alpha = FIELD_ALPHA[field] || FIELD_ALPHA.default;
    const beta = FIELD_BETA[field] || FIELD_BETA.default;
    const sConcepts = FIELD_SCONCEPTS[field] || FIELD_SCONCEPTS.default;
    if (!works.length) {
        return {
            kanon_index: 0, h_index: 0, total_citations: 0,
            total_papers: 0, avg_complexity: 0, avg_sophistication: 0, avg_authors: 0,
            first_author_fraction: 0, component_citations: 0,
            component_authorship: 0, component_sophistication: 0,
            component_journal: 0, avg_citations: 0
        };
    }

    const citationCounts = [];
    const authorshipScores = [];
    const complexityScores = [];   // S_text (marcadores do abstract)
    const journalScores = [];
    let firstAuthorCount = 0;

    const cleanId = authorId.replace('https://openalex.org/', '');

    for (let wi = 0; wi < works.length; wi++) {
        const work = works[wi];
        // Citations
        const citations = work.cited_by_count || 0;
        citationCounts.push(citations);

        // Authorship
        const authorships = work.authorships || [];
        const totalAuthors = authorships.length;
        let authorPosition = 'middle';

        for (let i = 0; i < authorships.length; i++) {
            const aid = (authorships[i].author?.id || '').replace('https://openalex.org/', '');
            if (aid === cleanId) {
                if (i === 0) {
                    authorPosition = totalAuthors > 1 ? 'first' : 'single';
                    firstAuthorCount++;
                } else if (i === totalAuthors - 1 && totalAuthors > 1) {
                    authorPosition = 'last';
                }
                if (authorships[i].is_corresponding) {
                    authorPosition = 'corresponding';
                }
                break;
            }
        }

        authorshipScores.push(calculateAuthorshipScore(authorPosition, Math.max(totalAuthors, 1), alpha));

        // Complexity (from abstract)
        const abstractText = reconstructAbstract(work.abstract_inverted_index) || work.display_name || '';
        const complexity = calculateComplexity(abstractText);
        complexityScores.push(complexity);

        // Journal impact from OpenAlex source data
        journalScores.push(getJournalScore(work));
    }

    const hIndex = calculateHIndex(citationCounts);
    const totalCitations = citationCounts.reduce((a, b) => a + b, 0);
    const avgCitations = mean(citationCounts);

    const C = mean(citationCounts.map(c => normalizeCitations(c)));
    const A = mean(authorshipScores);
    const S_text = mean(complexityScores);
    const S = beta * S_text + (1 - beta) * sConcepts;
    const J = mean(journalScores);

    const kanonScore = weights.C * C + weights.A * A + weights.S * S + weights.J * J;
    const kanonIndex = kanonScore * 100;

    const totalAuthorsList = works.map(w => (w.authorships || []).length);

    return {
        kanon_index: round(kanonIndex, 2),
        h_index: hIndex,
        total_citations: totalCitations,
        avg_citations: round(avgCitations, 2),
        total_papers: works.length,
        avg_complexity: round(S_text, 3),
        avg_sophistication: round(S, 3),
        avg_authors: round(mean(totalAuthorsList), 2),
        first_author_fraction: round(firstAuthorCount / works.length, 3),
        component_citations: round(C, 3),
        component_authorship: round(A, 3),
        component_sophistication: round(S, 3),
        component_journal: round(J, 3)
    };
}

// ── UI Functions ──────────────────────────────────────────────────────

function selectField(field) {
    selectedField = field;
    document.querySelectorAll('.field-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`[data-field="${field}"]`).classList.add('active');

    currentWeights = { ...FIELD_WEIGHTS[field] };
    updateWeightInputs(currentWeights);

    show('step2');
    show('step3');
    hide('step4');
    hide('step5');
    hide('searchResults');
    hide('searchError');
}

function updateWeightInputs(w) {
    document.getElementById('w_C').value = w.C.toFixed(2);
    document.getElementById('w_A').value = w.A.toFixed(2);
    document.getElementById('w_S').value = w.S.toFixed(2);
    document.getElementById('w_J').value = w.J.toFixed(2);
    updateWeightBars();
    updateWeightSum();
}

function toggleCustomWeights() {
    const custom = document.getElementById('customWeightsToggle').checked;
    document.querySelectorAll('.weight-input').forEach(inp => inp.disabled = !custom);
}

function updateWeightBars() {
    ['C','A','S','J'].forEach(k => {
        const val = parseFloat(document.getElementById(`w_${k}`).value) || 0;
        document.getElementById(`bar_${k}`).style.width = (val * 100) + '%';
    });
}

function updateWeightSum() {
    let sum = 0;
    ['C','A','S','J'].forEach(k => {
        sum += parseFloat(document.getElementById(`w_${k}`).value) || 0;
    });
    document.getElementById('weightSum').textContent = sum.toFixed(2);
    document.getElementById('weightWarning').style.display =
        Math.abs(sum - 1.0) > 0.02 ? 'inline' : 'none';
}

function getWeights() {
    if (document.getElementById('customWeightsToggle').checked) {
        const w = {
            C: parseFloat(document.getElementById('w_C').value) || 0,
            A: parseFloat(document.getElementById('w_A').value) || 0,
            S: parseFloat(document.getElementById('w_S').value) || 0,
            J: parseFloat(document.getElementById('w_J').value) || 0
        };
        // Normalize
        const total = w.C + w.A + w.S + w.J;
        if (total > 0) {
            w.C /= total; w.A /= total; w.S /= total; w.J /= total;
        }
        return w;
    }
    return currentWeights || FIELD_WEIGHTS.default;
}

function setSearchType(type) {
    searchType = type;
    document.querySelectorAll('.search-type-btn').forEach(b => b.classList.remove('active'));
    document.querySelector(`[data-type="${type}"]`).classList.add('active');

    const inp = document.getElementById('searchInput');
    if (type === 'orcid') {
        inp.placeholder = '0000-0000-0000-0000';
        document.getElementById('searchHint').textContent = 'ORCID garante identificação exata';
    } else {
        inp.placeholder = 'Digite o nome do pesquisador...';
        document.getElementById('searchHint').textContent = 'Dica: Use ORCID para identificação exata';
    }
}

async function doSearch() {
    // Sanitize input: trim and remove potential script injection characters
    const rawQuery = document.getElementById('searchInput').value.trim();
    const query = rawQuery.replace(/[<>]/g, '');
    if (!query) return;

    const btn = document.getElementById('searchBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Buscando...';
    hide('searchResults');
    hide('searchError');
    hide('step4');
    hide('step5');

    try {
        let authors;
        if (searchType === 'orcid') {
            // Accept ORCID with or without URL prefix
            let orcid = query;
            if (orcid.startsWith('https://orcid.org/')) {
                orcid = orcid.replace('https://orcid.org/', '');
            }
            if (!/^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$/.test(orcid)) {
                showError('Formato ORCID inválido. Use: 0000-0000-0000-0000');
                return;
            }
            authors = await searchAuthorByOrcid(orcid);
        } else {
            if (query.length < 3) {
                showError('Digite pelo menos 3 caracteres');
                return;
            }
            authors = await searchAuthorByName(query);
        }

        if (!authors.length) {
            showError('Nenhum pesquisador encontrado. Tente outra grafia ou use ORCID.');
            return;
        }

        renderCandidates(authors);
        show('searchResults');

    } catch (err) {
        if (err.message.includes('429')) {
            showError('Muitas requisições. Aguarde alguns segundos e tente novamente.');
        } else {
            showError('Erro ao conectar com OpenAlex: ' + err.message);
        }
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-search"></i> Buscar';
    }
}

function renderCandidates(candidates) {
    const list = document.getElementById('candidatesList');
    list.innerHTML = '';

    candidates.forEach((c, idx) => {
        // Use last_known_institutions (new API) with fallback to last_known_institution (deprecated)
        const instArr = c.last_known_institutions || [];
        const inst = instArr.length > 0 ? instArr[0] : (c.last_known_institution || null);
        const instName = inst?.display_name || 'Instituição desconhecida';
        const country = inst?.country_code || '';
        // Use topics (new API) with fallback to x_concepts (deprecated)
        const topicsList = c.topics || c.x_concepts || [];
        const topics = topicsList.slice(0, 3).map(t => t.display_name);
        const id = (c.id || '').replace('https://openalex.org/', '');

        const card = document.createElement('div');
        card.className = 'candidate-card';
        card.innerHTML = `
            <div class="cand-info">
                <div class="cand-name">${esc(c.display_name || 'Desconhecido')}</div>
                <div class="cand-inst">
                    <i class="fas fa-building"></i> ${esc(instName)}
                    ${country ? `<span class="cand-country">(${esc(country)})</span>` : ''}
                </div>
                <div class="cand-meta">
                    <span><i class="fas fa-file-lines"></i> ${(c.works_count || 0).toLocaleString()} publicações</span>
                    <span><i class="fas fa-quote-right"></i> ${(c.cited_by_count || 0).toLocaleString()} citações</span>
                    ${c.orcid ? `<span><i class="fab fa-orcid"></i> ${esc(c.orcid)}</span>` : ''}
                </div>
                ${topics.length ? `<div class="cand-topics">${topics.map(t => `<span class="topic-tag">${esc(t)}</span>`).join('')}</div>` : ''}
            </div>
            <button class="btn-select" data-candidate-idx="${idx}">
                <i class="fas fa-check"></i> Este é o pesquisador
            </button>
        `;

        // Bind click safely (avoids XSS via inline onclick with user data)
        card.querySelector('.btn-select').addEventListener('click', () => {
            selectAuthor(id, c.display_name || 'Desconhecido', instName, c.orcid || '');
        });

        list.appendChild(card);
    });
}

async function selectAuthor(authorId, name, institution, orcid) {
    selectedAuthor = { id: authorId, name, institution, orcid };
    hide('searchResults');
    show('step4');
    await startCalculation(authorId, name, institution, orcid);
}

async function startCalculation(authorId, name, institution, orcid) {
    const bar = document.getElementById('progressBar');
    const text = document.getElementById('progressText');
    const pct = document.getElementById('progressPercent');

    function setProgress(msg, percent) {
        bar.style.width = percent + '%';
        pct.textContent = Math.round(percent) + '%';
        text.textContent = msg;
    }

    bar.style.background = '';
    setProgress('Conectando à API OpenAlex...', 5);

    try {
        // Fetch works
        const works = await getAuthorWorks(authorId, setProgress);

        if (!works.length) {
            const periodMsg = selectedPeriodYears > 0
                ? (typeof t === 'function' ? t('progress_no_pubs') : `Nenhuma publicação encontrada nos últimos ${selectedPeriodYears} anos`)
                : (typeof t === 'function' ? t('progress_no_pubs') : 'Nenhuma publicação encontrada');
            setProgress(periodMsg, 100);
            bar.style.background = '#e67e22';
            return;
        }

        setProgress(`Analisando ${works.length} publicações...`, 75);
        await sleep(200);

        setProgress('Calculando complexidade metodológica...', 85);
        await sleep(200);

        // Calculate KANON
        const weights = getWeights();
        const metrics = calculateKanon(works, authorId, weights, selectedField);

        setProgress('Calculando KANON-Index...', 95);
        await sleep(300);

        setProgress('Concluído!', 100);

        lastResult = {
            researcher_name: name,
            institution: institution,
            orcid: orcid,
            openalex_id: authorId,
            field: selectedField,
            weights: weights,
            calculated_at: new Date().toLocaleString('pt-BR'),
            ...metrics
        };

        setTimeout(() => {
            hide('step4');
            displayResults(lastResult);
            show('step5');
        }, 400);

    } catch (err) {
        setProgress('Erro: ' + err.message, 100);
        bar.style.background = '#e74c3c';
    }
}

function displayResults(d) {
    document.getElementById('resultName').textContent = d.researcher_name;
    document.getElementById('resultInst').textContent = d.institution || '';
    document.getElementById('resultOrcid').textContent = d.orcid ? `ORCID: ${d.orcid}` : '';

    document.getElementById('kanonValue').textContent = d.kanon_index.toFixed(2);

    const fieldNames = { chemistry: 'Química', medicine: 'Medicina', physics: 'Física', economics: 'Economia' };
    document.getElementById('kanonField').textContent = fieldNames[d.field] || '';

    const badge = document.getElementById('kanonBadge');
    if (d.kanon_index >= 30) badge.className = 'kanon-badge kanon-high';
    else if (d.kanon_index >= 15) badge.className = 'kanon-badge kanon-mid';
    else badge.className = 'kanon-badge kanon-low';

    document.getElementById('mHindex').textContent = d.h_index;
    document.getElementById('mCitations').textContent = d.total_citations.toLocaleString();
    document.getElementById('mPapers').textContent = d.total_papers;
    document.getElementById('mAuthors').textContent = d.avg_authors.toFixed(1);
    document.getElementById('mFirstAuth').textContent = (d.first_author_fraction * 100).toFixed(0) + '%';
    document.getElementById('mComplexity').textContent = d.avg_complexity.toFixed(3);

    setCompBar('C', d.component_citations);
    setCompBar('A', d.component_authorship);
    setCompBar('S', d.component_sophistication);
    setCompBar('J', d.component_journal);

    document.getElementById('calcTime').textContent = d.calculated_at;
}

function setCompBar(letter, val) {
    document.getElementById('comp' + letter).style.width = (val * 100) + '%';
    document.getElementById('comp' + letter + 'val').textContent = val.toFixed(3);
}

function downloadResult() {
    if (!lastResult) return;
    const str = JSON.stringify(lastResult, null, 2);
    const blob = new Blob([str], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `kanon_${lastResult.researcher_name.replace(/\s+/g, '_')}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

function resetAll() {
    selectedField = null;
    selectedAuthor = null;
    lastResult = null;
    searchType = 'name';
    selectedPeriodYears = 5;

    document.querySelectorAll('.field-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('searchInput').value = '';
    document.getElementById('customWeightsToggle').checked = false;
    document.querySelectorAll('.weight-input').forEach(i => i.disabled = true);

    // Reset period selector
    document.querySelectorAll('.period-btn').forEach(b => b.classList.remove('active'));
    const defaultPeriodBtn = document.querySelector('.period-btn[data-years="5"]');
    if (defaultPeriodBtn) defaultPeriodBtn.classList.add('active');

    hide('step2'); hide('step3'); hide('step4'); hide('step5');
    hide('searchResults'); hide('searchError');

    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('progressBar').style.background = '';
    document.getElementById('progressPercent').textContent = '0%';

    setSearchType('name');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ── Helpers ───────────────────────────────────────────────────────────

function show(id) { document.getElementById(id).style.display = 'block'; }
function hide(id) { document.getElementById(id).style.display = 'none'; }
function esc(str) { const d = document.createElement('div'); d.textContent = str || ''; return d.innerHTML; }
function mean(arr) { return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0; }
function round(val, dec) { const f = Math.pow(10, dec); return Math.round(val * f) / f; }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function showError(msg) {
    document.getElementById('searchErrorMsg').textContent = msg;
    show('searchError');
}

// ── Init ──────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Weight input listeners
    document.querySelectorAll('.weight-input').forEach(inp => {
        inp.addEventListener('input', () => { updateWeightBars(); updateWeightSum(); });
    });

    // Enter key on search
    document.getElementById('searchInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') doSearch();
    });
});
