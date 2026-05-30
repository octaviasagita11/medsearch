import streamlit as st
import numpy as np
import pickle
import re
import scipy.sparse
from collections import defaultdict
from datetime import datetime
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ============================================================
# KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="MedSearch — Tolerant Retrieval",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS — Palet warna klasik elegan (krem, emas, coklat tua, hitam)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Lato:wght@300;400;700&display=swap');

html, body, [class*="css"] { font-family: 'Lato', sans-serif; }

/* Background utama — hitam tinta */
.stApp { background: #0e0c0a; color: #d4c9b8; }

/* ---- SIDEBAR ---- */
[data-testid="stSidebar"] {
    background: #13100d !important;
    border-right: 1px solid #2a221a !important;
}
[data-testid="stSidebar"] * { color: #D4C5A9 !important; }

/* ---- HERO ---- */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    font-weight: 600;
    color: #e8ddc8;
    line-height: 1.15;
    margin: 0;
    letter-spacing: -0.01em;
}
.hero-sub {
    font-size: 0.88rem;
    color: #E0D5C5;
    margin-top: 8px;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.hero-accent { color: #c9a96e; }
.hero-wrap {
    padding: 2.5rem 0 2rem 0;
    border-bottom: 1px solid #1e1812;
    margin-bottom: 2rem;
}

/* ---- INPUT ---- */
.stTextInput > div > div > input {
    background: #13100d !important;
    border: 1.5px solid #2a221a !important;
    border-radius: 8px !important;
    color: #e8ddc8 !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1.2rem !important;
    transition: border-color 0.2s ease;
}
.stTextInput > div > div > input:focus {
    border-color: #c9a96e !important;
    box-shadow: 0 0 0 3px rgba(201,169,110,0.1) !important;
}
.stTextInput > div > div > input::placeholder { color: #A89880 !important; }

/* ---- BUTTON ---- */
.stButton > button {
    background: #c9a96e !important;
    color: #0e0c0a !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Lato', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.72rem 1.8rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: #b8945a !important;
    transform: translateY(-1px) !important;
}

/* ---- BADGE ---- */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 3px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.badge-valid   { background: rgba(144,168,120,0.15); color: #90a878; border: 1px solid rgba(144,168,120,0.3); }
.badge-typo    { background: rgba(201,169,110,0.12); color: #c9a96e; border: 1px solid rgba(201,169,110,0.3); }
.badge-unknown { background: rgba(168,100,80,0.12);  color: #a86450; border: 1px solid rgba(168,100,80,0.3); }

/* ---- CARD DOKUMEN ---- */
.doc-card {
    background: #13100d;
    border: 1px solid #1e1812;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.2s, transform 0.15s;
    position: relative;
    overflow: hidden;
}
.doc-card:hover { border-color: #3a2e20; transform: translateY(-1px); }
.doc-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0; width: 2px;
    background: linear-gradient(180deg, #c9a96e, #8a6a3a);
}
.doc-rank { font-family: 'Playfair Display', serif; font-size: 1.4rem; color: #C9A96E; font-style: italic; }
.doc-id   { font-size: 0.68rem; color: #c9a96e; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; }
.doc-sim  { font-size: 0.78rem; color: #4a3e30; }
.doc-sim span { color: #c9a96e; font-weight: 700; }
.doc-snippet  { font-size: 0.87rem; color: #F0EBE3; line-height: 1.7; margin-top: 6px; }

/* ---- METRIC CARD ---- */
.metric-card {
    background: #13100d;
    border: 1px solid #1e1812;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.metric-val { font-family: 'Playfair Display', serif; font-size: 1.8rem; color: #c9a96e; }
.metric-lbl { font-size: 0.68rem; color: #D4C5A9; letter-spacing: 0.1em; text-transform: uppercase; margin-top: 3px; }

/* ---- ABOUT CARD ---- */
.about-card {
    background: #13100d;
    border: 1px solid #1e1812;
    border-radius: 8px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
}
.about-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    color: #e8ddc8;
    margin-bottom: 0.8rem;
    letter-spacing: 0.01em;
}
.about-body { font-size: 0.85rem; color: #E0D5C5; line-height: 1.8; }
.about-body code {
    background: #0a0804;
    color: #c9a96e;
    padding: 2px 6px;
    border-radius: 3px;
    font-size: 0.78rem;
    border: 1px solid #2a221a;
}
.step-row { display: flex; align-items: flex-start; margin-bottom: 10px; gap: 10px; }
.step-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; min-width: 20px;
    background: rgba(201,169,110,0.12);
    color: #c9a96e;
    border-radius: 3px;
    font-size: 0.68rem; font-weight: 700;
    border: 1px solid rgba(201,169,110,0.2);
}
.step-text { font-size: 0.85rem; color: #E0D5C5; line-height: 1.65; }

/* ---- STATISTIK ---- */
.stat-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #1a1510;
    font-size: 0.85rem;
}
.stat-row:last-child { border-bottom: none; }
.hist-row {
    background: #13100d;
    border: 1px solid #1e1812;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.85rem;
}
.hist-query { color: #d4c9b8; font-style: italic; }
.hist-time  { font-size: 0.7rem; color: #3a3028; letter-spacing: 0.05em; }
.hist-badge-valid { color: #90a878; font-size: 0.7rem; }
.hist-badge-typo  { color: #c9a96e; font-size: 0.7rem; }

hr { border-color: #1e1812 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0e0c0a; }
::-webkit-scrollbar-thumb { background: #2a221a; border-radius: 2px; }

/* Divider warna */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #2a221a, transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource(show_spinner="Memuat model...")
def load_model():
    import pandas as pd
    TFIDF_norm      = scipy.sparse.load_npz('tfidf_matrix.npz')
    IDF_vector      = np.load('idf_vector.npy')
    vocabulary      = pickle.load(open('vocabulary.pkl', 'rb'))
    term2idx        = pickle.load(open('term2idx.pkl', 'rb'))
    bigram_index    = pickle.load(open('bigram_index.pkl', 'rb'))
    term_bigrams    = pickle.load(open('term_bigrams.pkl', 'rb'))
    permuterm_index = pickle.load(open('permuterm_index.pkl', 'rb'))
    stopword_list   = pickle.load(open('stopword_list.pkl', 'rb'))
    kata_dikenal    = pickle.load(open('whitelist.pkl', 'rb'))
    documents       = pd.read_pickle('dataset.pkl')
    stemmer         = StemmerFactory().create_stemmer()
    return (TFIDF_norm, IDF_vector, vocabulary, term2idx,
            bigram_index, term_bigrams, permuterm_index,
            stopword_list, kata_dikenal, documents, stemmer)

(TFIDF_norm, IDF_vector, vocabulary, term2idx,
 bigram_index, term_bigrams, permuterm_index,
 stopword_list, KATA_DIKENAL, documents, stemmer) = load_model()

N_terms = len(vocabulary)


# ============================================================
# SESSION STATE
# ============================================================
if 'halaman'        not in st.session_state: st.session_state['halaman']        = 'Beranda'
if 'query_input'    not in st.session_state: st.session_state['query_input']    = ''
if 'search_history' not in st.session_state: st.session_state['search_history'] = []
if 'query_counter'  not in st.session_state: st.session_state['query_counter']  = {}
if 'total_searches' not in st.session_state: st.session_state['total_searches'] = 0
if 'total_typo'     not in st.session_state: st.session_state['total_typo']     = 0
if 'total_docs'     not in st.session_state: st.session_state['total_docs']     = 0


# ============================================================
# FUNGSI RETRIEVAL
# ============================================================
def case_folding(text):  return text.lower()
def tokenizing(text):    return text.split()

def cleaning(text):
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def stopword_removal(tokens):
    return [t for t in tokens if t not in stopword_list and len(t) > 2]

def stem_tokens(tokens):
    return [stemmer.stem(t) for t in tokens]

def get_bigrams(word):
    p = '$' + word + '$'
    return set(p[i:i+2] for i in range(len(p)-1))

def damerau_levenshtein(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if s1[i-1]==s2[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
            if i>1 and j>1 and s1[i-1]==s2[j-2] and s1[i-2]==s2[j-1]:
                dp[i][j] = min(dp[i][j], dp[i-2][j-2]+1)
    return dp[m][n]

def koreksi_typo(word, max_distance=2, max_candidates=3):
    if word in term2idx or word in KATA_DIKENAL:
        return [word], False
    word_bgs = get_bigrams(word)
    kandidat_set = set()
    for bg in word_bgs:
        kandidat_set.update(bigram_index.get(bg, set()))
    if not kandidat_set:
        return [], True
    kandidat_sim = []
    for term in kandidat_set:
        bg2   = term_bigrams.get(term, get_bigrams(term))
        inter = len(word_bgs & bg2)
        union = len(word_bgs | bg2)
        sim   = inter / union if union > 0 else 0
        if sim >= 0.15:
            kandidat_sim.append((term, sim))
    kandidat_sim.sort(key=lambda x: x[1], reverse=True)
    top100 = [t for t, _ in kandidat_sim[:100]]
    adaptive = max_distance + (1 if len(word) >= 6 else 0)
    hasil_ed = [(t, damerau_levenshtein(word, t)) for t in top100
                if damerau_levenshtein(word, t) <= adaptive
                and abs(len(t)-len(word)) <= 2]
    hasil_ed.sort(key=lambda x: x[1])
    return [t for t, _ in hasil_ed[:max_candidates]], True

def ekspansi_morfologi(token, max_hasil=10):
    hasil = set()
    prefix_key = '$' + token
    kandidat = set()
    for key, terms in permuterm_index.items():
        if key.startswith(prefix_key):
            kandidat.update(terms)
    for term in kandidat:
        if stemmer.stem(term) == token:
            hasil.add(term)
    if token in term2idx:
        hasil.add(token)
    return list(hasil)[:max_hasil]

def vectorize_query(expanded_terms):
    if not expanded_terms:
        return np.zeros(N_terms)
    query_vec  = np.zeros(N_terms)
    term_count = defaultdict(int)
    for t in expanded_terms:
        if t in term2idx: term_count[t] += 1
    total = len(expanded_terms) if expanded_terms else 1
    for term, count in term_count.items():
        idx = term2idx[term]
        query_vec[idx] = (count/total) * IDF_vector[idx]
    norm = np.linalg.norm(query_vec)
    if norm > 0: query_vec = query_vec / norm
    return query_vec

def tolerant_retrieval(query_text, top_k=50):
    text   = case_folding(query_text)
    text   = cleaning(text)
    tokens = tokenizing(text)
    tokens = stopword_removal(tokens)
    tokens = stem_tokens(tokens)
    expanded_terms = []
    retrieval_log  = []
    for token in tokens:
        koreksi_hasil, is_typo = koreksi_typo(token)
        if not is_typo:
            morf = ekspansi_morfologi(token)
            expanded_terms.extend(morf)
            retrieval_log.append({'token': token, 'status': 'valid',
                                  'koreksi': token, 'ekspansi': morf})
        elif koreksi_hasil:
            all_morf = []
            for k in koreksi_hasil:
                all_morf.extend(ekspansi_morfologi(k))
            expanded_terms.extend(all_morf)
            retrieval_log.append({'token': token, 'status': 'typo',
                                  'koreksi': koreksi_hasil, 'ekspansi': all_morf[:8]})
        else:
            retrieval_log.append({'token': token, 'status': 'unknown',
                                  'koreksi': None, 'ekspansi': []})
    if not expanded_terms:
        return [], retrieval_log, tokens
    query_vec = vectorize_query(expanded_terms)
    sims      = TFIDF_norm.dot(query_vec)
    top_idx   = np.argsort(sims)[::-1][:top_k]
    hasil = []
    for rank, idx in enumerate(top_idx, 1):
        score = sims[idx]
        if score > 0:
            hasil.append({
                'rank'      : rank,
                'doc_id'    : documents['doc_id'].iloc[idx],
                'similarity': round(float(score), 4),
                'content'   : documents['content'].iloc[idx]
            })
    return hasil, retrieval_log, tokens


# ============================================================
# SIDEBAR NAVIGASI
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:1.4rem 0.5rem 1.6rem;border-bottom:1px solid #1e1812">
        <div style="font-family:'Playfair Display',serif;font-size:1.5rem;
                    color:#e8ddc8;letter-spacing:0.01em">
            Med<span style="color:#c9a96e">Search</span>
        </div>
        <div style="font-size:0.62rem;color:#D4C5A9;margin-top:4px;
                    letter-spacing:0.12em;text-transform:uppercase">
            Tolerant Retrieval System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.62rem;color:#D4C5A9;letter-spacing:0.1em;text-transform:uppercase;padding:14px 4px 8px'>Navigasi</div>", unsafe_allow_html=True)

    menu_items = [
        ("Beranda",         "○"),
        ("Statistik",       "◎"),
        ("Tentang Sistem",  "◈"),
    ]
    for nama, icon in menu_items:
        is_active = st.session_state['halaman'] == nama
        bg     = "background:rgba(201,169,110,0.08);border:1px solid rgba(201,169,110,0.15);" if is_active else "border:1px solid transparent;"
        color  = "color:#2a2a2a  !important;" if is_active else "color:#D4C5A9 !important;"
        if st.button(f"{icon}  {nama}", key=f"nav_{nama}", use_container_width=True):
            st.session_state['halaman'] = nama
            st.rerun()

    # Info sesi ringkas
    if st.session_state['total_searches'] > 0:
        st.markdown(f"""
        <div style="margin-top:1.5rem;padding:1rem;background:#0a0804;
                    border:1px solid #1e1812;border-radius:6px">
            <div style="font-size:0.62rem;color:#D4C5A9;letter-spacing:0.1em;
                        text-transform:uppercase;margin-bottom:10px">Sesi ini</div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="font-size:0.78rem;color:#D4C5A9">Pencarian</span>
                <span style="font-family:'Playfair Display',serif;color:#c9a96e">
                    {st.session_state['total_searches']}</span>
            </div>
            <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                <span style="font-size:0.78rem;color:#D4C5A9">Typo dikoreksi</span>
                <span style="font-family:'Playfair Display',serif;color:#c9a96e">
                    {st.session_state['total_typo']}</span>
            </div>
            <div style="display:flex;justify-content:space-between">
                <span style="font-size:0.78rem;color:#D4C5A9">Dokumen ditemukan</span>
                <span style="font-family:'Playfair Display',serif;color:#c9a96e">
                    {st.session_state['total_docs']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:fixed;bottom:1.2rem;left:0;width:260px;
                padding:0 1.2rem;border-top:1px solid #1a1510;padding-top:0.8rem">
        <div style="font-size:0.62rem;color:#1e1812;line-height:1.7;letter-spacing:0.05em">
            VSM · TF-IDF · Cosine Similarity<br>
            Bigram · Permuterm · Damerau-Levenshtein
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HALAMAN: BERANDA
# ============================================================
if st.session_state['halaman'] == 'Beranda':

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">Med<span class="hero-accent">Search</span></div>
        <div class="hero-sub">Tolerant Retrieval · Berita Kesehatan Bahasa Indonesia · Morfologi &amp; VSM</div>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            "query",
            value=st.session_state.get('query_input', ''),
            placeholder="Ketik query... boleh typo! contoh: vaksinsi, pnyakit, obaat",
            label_visibility="collapsed",
            key="query_input"
        )
    with col_btn:
        cari = st.button("Cari →", use_container_width=True)

    if (cari or query) and query.strip():
        with st.spinner(""):
            hasil, log, tokens = tolerant_retrieval(query.strip(), top_k=50)

        # Simpan ke statistik sesi
        typo_count = sum(1 for l in log if l['status'] == 'typo')
        now        = datetime.now().strftime("%H:%M:%S")
        st.session_state['search_history'].insert(0, {
            'query'     : query.strip(),
            'waktu'     : now,
            'hasil'     : len(hasil),
            'typo'      : typo_count,
            'ada_typo'  : typo_count > 0
        })
        st.session_state['total_searches'] += 1
        st.session_state['total_typo']     += typo_count
        st.session_state['total_docs']     += len(hasil)
        q = query.strip().lower()
        st.session_state['query_counter'][q] = st.session_state['query_counter'].get(q, 0) + 1

        # Log retrieval
        st.markdown("#### Proses Tolerant Retrieval")
        log_cols = st.columns(len(log)) if log else st.columns(1)
        for i, l in enumerate(log):
            with log_cols[i]:
                if l['status'] == 'valid':
                    st.markdown(f"""
                    <div style="background:#0d0c08;border:1px solid #1e1a12;
                                border-radius:6px;padding:0.8rem;text-align:center">
                        <div class="badge badge-valid">valid</div>
                        <div style="font-size:1rem;color:#e8ddc8;margin:6px 0 3px;
                                    font-family:'Playfair Display',serif;font-style:italic">
                            {l['token']}</div>
                        <div style="font-size:0.7rem;color:#A89880">
                            {', '.join(l['ekspansi'][:4])}{'...' if len(l['ekspansi'])>4 else ''}
                        </div>
                    </div>""", unsafe_allow_html=True)
                elif l['status'] == 'typo':
                    kor = l['koreksi'][0] if l['koreksi'] else '?'
                    st.markdown(f"""
                    <div style="background:#120e08;border:1px solid #2a2010;
                                border-radius:6px;padding:0.8rem;text-align:center">
                        <div class="badge badge-typo">typo dikoreksi</div>
                        <div style="font-size:0.82rem;color:#5a4e38;margin:4px 0 1px;
                                    text-decoration:line-through">{l['token']}</div>
                        <div style="font-size:1rem;color:#c9a96e;
                                    font-family:'Playfair Display',serif;font-style:italic">
                            → {kor}</div>
                        <div style="font-size:0.7rem;color:#4a3e28;margin-top:2px">
                            {', '.join(l['ekspansi'][:3])}{'...' if len(l['ekspansi'])>3 else ''}
                        </div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background:#120a08;border:1px solid #2a1510;
                                border-radius:6px;padding:0.8rem;text-align:center">
                        <div class="badge badge-unknown">tidak dikenal</div>
                        <div style="font-size:1rem;color:#a86450;margin:6px 0;
                                    font-family:'Playfair Display',serif;font-style:italic">
                            {l['token']}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        if hasil:
            c1, c2, c3, c4 = st.columns(4)
            exp_count = sum(len(l['ekspansi']) for l in log)
            for col, val, lbl in [
                (c1, len(hasil),             "Dokumen ditemukan"),
                (c2, hasil[0]['similarity'],  "Similarity tertinggi"),
                (c3, typo_count,             "Token dikoreksi"),
                (c4, exp_count,              "Term ekspansi"),
            ]:
                with col:
                    st.markdown(f"""<div class="metric-card">
                        <div class="metric-val">{val}</div>
                        <div class="metric-lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"#### Hasil &nbsp;<span style='font-size:0.82rem;color:#A89880;font-weight:400;font-family:Lato'>— {len(hasil)} dokumen</span>", unsafe_allow_html=True)

            for doc in hasil:
                snippet = doc['content'][:350].strip()
                for l in log:
                    tok = l['koreksi'] if isinstance(l['koreksi'], str) else (l['koreksi'][0] if l['koreksi'] else '')
                    if tok:
                        snippet = re.compile(re.escape(tok), re.IGNORECASE).sub(
                            f'<mark style="background:rgba(201,169,110,0.18);color:#c9a96e;'
                            f'border-radius:2px;padding:0 2px">{tok}</mark>', snippet)

                sim_w = min(int(doc['similarity'] * 800), 100)
                st.markdown(f"""
                <div class="doc-card">
                    <div style="display:flex;align-items:flex-start;gap:1.2rem">
                        <div style="min-width:38px;text-align:center;padding-top:2px">
                            <div class="doc-rank">#{doc['rank']}</div>
                        </div>
                        <div style="flex:1">
                            <div style="display:flex;align-items:center;
                                        justify-content:space-between;margin-bottom:5px">
                                <span class="doc-id">{doc['doc_id']}</span>
                                <span class="doc-sim">similarity <span>{doc['similarity']}</span></span>
                            </div>
                            <div class="doc-snippet">{snippet}...</div>
                            <div style="margin-top:8px;background:#0a0804;
                                        border-radius:2px;height:2px;overflow:hidden">
                                <div style="background:linear-gradient(90deg,#c9a96e,#8a6a3a);
                                            height:100%;width:{sim_w}%;border-radius:2px"></div>
                            </div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#13100d;border:1px dashed #1e1812;border-radius:8px;
                        padding:3rem;text-align:center;margin-top:1rem">
                <div style="font-family:'Playfair Display',serif;font-size:1.1rem;
                            color:#A89880;font-style:italic;margin-bottom:6px">
                    Tidak ada dokumen relevan ditemukan</div>
                <div style="color:#A89880;font-size:0.85rem">Coba kata lain.</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#13100d;border:1px dashed #1e1812;border-radius:8px;
                    padding:3.5rem 2rem;text-align:center;margin-top:1rem">
            <div style="font-family:'Playfair Display',serif;font-size:1.3rem;
                        color:#A89880;font-style:italic;margin-bottom:8px">
                Mulai pencarian</div>
            <div style="color:#A89880;font-size:0.88rem;line-height:1.8">
                Ketik query di atas — boleh kata dasar, berimbuhan, atau dengan typo.<br>
                Sistem akan mengoreksi dan menemukan dokumen yang relevan secara otomatis.
            </div>
        </div>""", unsafe_allow_html=True)


# ============================================================
# HALAMAN: STATISTIK
# ============================================================
elif st.session_state['halaman'] == 'Statistik':

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">Statistik <span class="hero-accent">Pencarian</span></div>
        <div class="hero-sub">Aktivitas pencarian selama sesi ini berlangsung</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state['total_searches'] == 0:
        st.markdown("""
        <div style="background:#13100d;border:1px dashed #1e1812;border-radius:8px;
                    padding:3.5rem 2rem;text-align:center">
            <div style="font-family:'Playfair Display',serif;font-size:1.1rem;
                        color:#2a2218;font-style:italic;margin-bottom:6px">
                Belum ada aktivitas pencarian</div>
            <div style="color:#2a2218;font-size:0.85rem">
                Kembali ke Beranda dan mulai pencarian pertama kamu.
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        # Ringkasan sesi
        s1, s2, s3, s4 = st.columns(4)
        typo_rate = round(st.session_state['total_typo'] /
                          max(st.session_state['total_searches'], 1) * 100)
        avg_docs  = round(st.session_state['total_docs'] /
                          max(st.session_state['total_searches'], 1), 1)
        for col, val, lbl in [
            (s1, st.session_state['total_searches'], "Total pencarian"),
            (s2, st.session_state['total_typo'],     "Typo dikoreksi"),
            (s3, f"{typo_rate}%",                    "Tingkat typo"),
            (s4, avg_docs,                           "Rata-rata dokumen"),
        ]:
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-val">{val}</div>
                    <div class="metric-lbl">{lbl}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns([3, 2])

        with col_l:
            st.markdown("#### Riwayat pencarian")
            if st.session_state['search_history']:
                for item in st.session_state['search_history'][:20]:
                    tipe    = "⚠ typo" if item['ada_typo'] else "✓ valid"
                    t_color = "#c9a96e" if item['ada_typo'] else "#90a878"
                    st.markdown(f"""
                    <div class="hist-row">
                        <div>
                            <span class="hist-query">"{item['query']}"</span>
                            <span style="font-size:0.7rem;color:#2a2218;margin-left:8px">
                                → {item['hasil']} dok</span>
                        </div>
                        <div style="display:flex;align-items:center;gap:10px">
                            <span style="font-size:0.7rem;color:{t_color}">{tipe}</span>
                            <span class="hist-time">{item['waktu']}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#2a2218;font-size:0.85rem'>Belum ada riwayat.</div>", unsafe_allow_html=True)

        with col_r:
            st.markdown("#### Query terpopuler")
            if st.session_state['query_counter']:
                sorted_q = sorted(
                    st.session_state['query_counter'].items(),
                    key=lambda x: x[1], reverse=True
                )
                max_count = sorted_q[0][1] if sorted_q else 1
                for i, (q, count) in enumerate(sorted_q[:10], 1):
                    bar_w = int((count / max_count) * 100)
                    st.markdown(f"""
                    <div style="background:#13100d;border:1px solid #1e1812;border-radius:6px;
                                padding:0.7rem 1rem;margin-bottom:0.4rem">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:5px">
                            <span style="font-size:0.85rem;color:#d4c9b8;font-style:italic">
                                {i}. {q}</span>
                            <span style="font-family:'Playfair Display',serif;
                                         font-size:0.9rem;color:#c9a96e">{count}×</span>
                        </div>
                        <div style="background:#0a0804;border-radius:2px;height:2px">
                            <div style="background:linear-gradient(90deg,#c9a96e,#8a6a3a);
                                        height:100%;width:{bar_w}%;border-radius:2px"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.markdown("<div style='color:#2a2218;font-size:0.85rem'>Belum ada data.</div>", unsafe_allow_html=True)

        # Tombol reset
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("↺  Reset statistik sesi", key="reset_stat"):
            st.session_state['search_history'] = []
            st.session_state['query_counter']  = {}
            st.session_state['total_searches'] = 0
            st.session_state['total_typo']     = 0
            st.session_state['total_docs']     = 0
            st.rerun()


# ============================================================
# HALAMAN: TENTANG SISTEM
# ============================================================
elif st.session_state['halaman'] == 'Tentang Sistem':

    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-title">Tentang <span class="hero-accent">Sistem</span></div>
        <div class="hero-sub">Metodologi · Teknologi · Informasi model</div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown("""
        <div class="about-card">
            <div class="about-title">Deskripsi sistem</div>
            <div class="about-body">
                <b style="color:#d4c9b8">MedSearch</b> adalah sistem pencarian dokumen
                berbasis <code>Tolerant Retrieval</code> yang dirancang khusus untuk
                dokumen berita kesehatan Bahasa Indonesia. Sistem mampu menangani
                variasi penulisan, kesalahan ketik (typo), serta variasi morfologi
                kata secara otomatis menggunakan pendekatan
                <code>Vector Space Model (VSM)</code>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <div class="about-title">Alur metode penelitian</div>
            <div class="about-body">
                <div class="step-row"><span class="step-num">1</span>
                    <span class="step-text"><b style="color:#d4c9b8">Pengumpulan data</b> —
                    958 dokumen berita kesehatan Bahasa Indonesia dari Liputan6.com</span></div>
                <div class="step-row"><span class="step-num">2</span>
                    <span class="step-text"><b style="color:#d4c9b8">Preprocessing teks</b> —
                    Case folding · Cleaning · Tokenizing · Stopword removal · Stemming (Sastrawi)</span></div>
                <div class="step-row"><span class="step-num">3</span>
                    <span class="step-text"><b style="color:#d4c9b8">Analisis morfologi</b> —
                    Permuterm Index untuk ekspansi variasi imbuhan dari setiap kata dasar</span></div>
                <div class="step-row"><span class="step-num">4</span>
                    <span class="step-text"><b style="color:#d4c9b8">Tolerant Retrieval</b> —
                    Bigram Index · Damerau-Levenshtein Distance · Whitelist istilah medis</span></div>
                <div class="step-row"><span class="step-num">5</span>
                    <span class="step-text"><b style="color:#d4c9b8">VSM — Perhitungan similaritas</b> —
                    TF Matrix · IDF Vector · TF-IDF (958×10.333) · Normalisasi L2 · Cosine Similarity</span></div>
                <div class="step-row"><span class="step-num">6</span>
                    <span class="step-text"><b style="color:#d4c9b8">Pengujian</b> —
                    15 query uji · Precision · Recall · F1-Score · Analisis variasi top-k</span></div>
                <div class="step-row"><span class="step-num">7</span>
                    <span class="step-text"><b style="color:#d4c9b8">Analisis hasil</b> —
                    top-k optimal = 50 · Mean F1-Score = 0.6243</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <div class="about-title">Toleransi typo</div>
            <div class="about-body">
                Sistem menggunakan <code>Damerau-Levenshtein Distance</code> yang mendukung
                4 operasi: sisip, hapus, ganti, dan <b style="color:#d4c9b8">transposisi</b>
                (tukar 2 huruf bersebelahan = 1 operasi). Ini memastikan kasus seperti
                <code>pulhi → pulih</code> terdeteksi dengan jarak 1, bukan 2.
                Whitelist memastikan istilah medis seperti
                <code>covid</code>, <code>hiv</code>, <code>aids</code>, <code>tbc</code>
                tidak salah dikoreksi.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        <div class="about-card">
            <div class="about-title">Statistik model</div>
            <div class="about-body">
        """, unsafe_allow_html=True)
        stats = [
            ("958",    "Dokumen",          "#c9a96e"),
            ("10.333", "Term vocabulary",  "#c9a96e"),
            ("843",    "Stopword",         "#c9a96e"),
            ("15",     "Query uji",        "#a09070"),
            ("50",     "Top-K optimal",    "#a09070"),
            ("0.9032", "Mean Precision",   "#90a878"),
            ("0.6048", "Mean Recall",      "#90a878"),
            ("0.6243", "Mean F1-Score",    "#90a878"),
        ]
        for val, lbl, color in stats:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:8px 0;border-bottom:1px solid #1a1510">
                <span style="font-size:0.82rem;color:#D4C5A9">{lbl}</span>
                <span style="font-family:'Playfair Display',serif;
                             font-size:1rem;color:{color}">{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card" style="margin-top:1rem">
            <div class="about-title">Teknologi</div>
            <div class="about-body">
                <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:4px">
        """, unsafe_allow_html=True)
        for tech in ["Python", "PySastrawi", "NumPy", "SciPy", "Streamlit", "Google Colab"]:
            st.markdown(f"""<span style="background:#0a0804;color:#c9a96e;
                padding:4px 10px;border-radius:3px;font-size:0.75rem;
                border:1px solid #2a221a;display:inline-block;margin:2px">{tech}</span>""",
                unsafe_allow_html=True)
        st.markdown("</div></div></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="about-card">
            <div class="about-title">Dataset</div>
            <div class="about-body">
                <div class="stat-row">
                    <span>Sumber</span>
                    <span style="color:#d4c9b8">Liputan6.com</span>
                </div>
                <div class="stat-row">
                    <span>Format</span>
                    <span style="color:#d4c9b8">doc_id · content</span>
                </div>
                <div class="stat-row">
                    <span>Rata-rata panjang</span>
                    <span style="color:#d4c9b8">1.325 karakter</span>
                </div>
                <div class="stat-row">
                    <span>Topik</span>
                    <span style="color:#d4c9b8">Kesehatan Indonesia</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
