import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

from predictor import load_benchmark, load_teacher
from utils import LABEL_COLS, LABEL_META, toxicity_level

# ── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='AMANE — Détection de toxicité',
    page_icon='🛡️',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── CSS global ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.stTextArea textarea { font-size: 14px; }
.label-fr { font-size: 13px; font-weight: 500; }
.model-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 500;
}
.badge-available { background: #E1F5EE; color: #085041; }
.badge-pending   { background: #FAEEDA; color: #633806; }
div[data-testid="stProgress"] > div { border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Chargement des modèles (cached) ─────────────────────────────────────────
@st.cache_resource(show_spinner='Chargement du modèle Benchmark...')
def get_benchmark():
    return load_benchmark()

@st.cache_resource(show_spinner='Chargement de XLM-R Teacher...')
def get_teacher():
    return load_teacher()

benchmark = get_benchmark()
teacher   = get_teacher()

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 🛡️ AMANE')
    st.markdown('*Détection de discours toxiques*')
    st.divider()

    page = st.radio(
        'Navigation',
        ['Analyse', 'Batch CSV', 'Comparaison modèles'],
        label_visibility='collapsed',
    )

    st.divider()
    st.markdown('**Statut des modèles**')

    if benchmark.loaded:
        st.markdown('<span class="model-badge badge-available">✓ Benchmark</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="model-badge badge-pending">⚠ Benchmark</span>', unsafe_allow_html=True)
        st.caption(benchmark.error)

    if teacher.loaded:
        st.markdown('<span class="model-badge badge-available">✓ XLM-R Teacher</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="model-badge badge-pending">⏳ XLM-R Teacher</span>', unsafe_allow_html=True)
        st.caption('Disponible après entraînement')

    st.divider()
    st.caption('Projet AMANE — Master 4 IA & Data Science')


# ════════════════════════════════════════════════════════════════════════════
# Helpers visuels
# ════════════════════════════════════════════════════════════════════════════

def render_probs(probs: dict, model_name: str):
    """Affiche les probabilités par label avec barres de progression."""
    level, level_color = toxicity_level(probs)

    # Niveau global
    st.markdown(
        f'<div style="margin-bottom:12px;">'
        f'<span style="font-size:13px;color:#666;">Niveau de toxicité global : </span>'
        f'<span style="font-size:14px;font-weight:500;color:{level_color};">{level}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Barres par label
    for col in LABEL_COLS:
        prob  = probs[col]
        meta  = LABEL_META[col]
        color = meta['color']
        label = meta['fr']
        flag  = '●' if prob > 0.5 else '○'

        col1, col2, col3 = st.columns([3, 6, 1])
        with col1:
            st.markdown(
                f'<span class="label-fr" style="color:{color if prob>0.5 else "inherit"};">'
                f'{flag} {label}</span>',
                unsafe_allow_html=True
            )
        with col2:
            st.progress(float(prob))
        with col3:
            st.markdown(
                f'<span style="font-size:13px;font-weight:500;">{prob*100:.0f}%</span>',
                unsafe_allow_html=True
            )


def radar_chart(probs_dict: dict) -> go.Figure:
    """Radar chart pour une prédiction."""
    labels = [LABEL_META[c]['fr'] for c in LABEL_COLS]
    values = [probs_dict[c] for c in LABEL_COLS]
    values += values[:1]
    labels += labels[:1]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels,
        fill='toself',
        line=dict(color='#534AB7', width=2),
        fillcolor='rgba(83,74,183,0.15)',
        name='Probabilités',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10))),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        height=300,
    )
    return fig


def comparison_radar(probs_bench: dict, probs_teacher: dict) -> go.Figure:
    """Radar chart comparatif deux modèles."""
    labels = [LABEL_META[c]['fr'] for c in LABEL_COLS]
    v_b    = [probs_bench[c]   for c in LABEL_COLS] + [probs_bench[LABEL_COLS[0]]]
    v_t    = [probs_teacher[c] for c in LABEL_COLS] + [probs_teacher[LABEL_COLS[0]]]
    l      = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=v_b, theta=l, fill='toself', name='Benchmark',
        line=dict(color='#888780', width=2),
        fillcolor='rgba(136,135,128,0.10)',
    ))
    fig.add_trace(go.Scatterpolar(
        r=v_t, theta=l, fill='toself', name='XLM-R Teacher',
        line=dict(color='#534AB7', width=2),
        fillcolor='rgba(83,74,183,0.15)',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10))),
        showlegend=True,
        legend=dict(x=1.1, y=1.0),
        margin=dict(l=40, r=80, t=30, b=30),
        height=320,
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Analyse commentaire unique
# ════════════════════════════════════════════════════════════════════════════
if page == 'Analyse':
    st.title('Analyse de commentaire')
    st.caption('Entrez un commentaire pour détecter la toxicité avec le modèle sélectionné.')

    # Sélecteur de modèle
    model_options = ['Benchmark (OVR LogReg)']
    if teacher.loaded:
        model_options.append('XLM-R Teacher')

    selected_model = st.selectbox('Modèle', model_options)

    if not teacher.loaded and len(model_options) == 1:
        st.info(
            '⏳ **XLM-R Teacher** sera disponible après l\'entraînement. '
            'Exécute `AMANE_Teacher.ipynb` puis place les fichiers dans `models/teacher/`.',
            icon='ℹ️'
        )

    # Zone de saisie
    default_text = 'You are a complete idiot and I hope something terrible happens to you.'
    comment = st.text_area(
        'Commentaire à analyser',
        value=default_text,
        height=120,
        placeholder='Entrez un commentaire en anglais ou en Darija...',
    )

    col_btn, col_clear = st.columns([1, 5])
    with col_btn:
        analyze = st.button('Analyser', type='primary', use_container_width=True)
    with col_clear:
        clear = st.button('Effacer', use_container_width=False)
    if clear:
        comment = ''

    if analyze and comment.strip():
        with st.spinner('Analyse en cours...'):
            try:
                if selected_model == 'Benchmark (OVR LogReg)':
                    probs = benchmark.predict(comment)
                else:
                    probs = teacher.predict(comment)

                st.divider()
                col_left, col_right = st.columns([3, 2])

                with col_left:
                    st.subheader('Résultats par label')
                    render_probs(probs, selected_model)

                with col_right:
                    st.subheader('Radar')
                    st.plotly_chart(radar_chart(probs), use_container_width=True)

                # Détails JSON (expandable)
                with st.expander('Voir les probabilités brutes'):
                    st.json({col: f'{v*100:.2f}%' for col, v in probs.items()})

            except Exception as e:
                st.error(f'Erreur : {e}')

    elif analyze and not comment.strip():
        st.warning('Veuillez entrer un commentaire.')


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Analyse Batch CSV
# ════════════════════════════════════════════════════════════════════════════
elif page == 'Batch CSV':
    st.title('Analyse en lot — CSV')
    st.caption(
        'Uploadez un fichier CSV avec une colonne `comment_text`. '
        'Le système analysera chaque ligne et vous renverra un CSV annoté.'
    )

    # Format attendu
    with st.expander('Format du CSV attendu'):
        st.markdown('Le fichier doit contenir **au minimum** une colonne `comment_text` :')
        st.dataframe(pd.DataFrame({
            'comment_text': [
                'You are an idiot',
                'Have a great day!',
                'I will find you and hurt you',
            ]
        }), use_container_width=True)

    # Sélecteur modèle
    batch_model_options = ['Benchmark (OVR LogReg)']
    if teacher.loaded:
        batch_model_options.append('XLM-R Teacher')
    batch_model = st.selectbox('Modèle pour le batch', batch_model_options, key='batch_model')

    # Upload
    uploaded = st.file_uploader('Choisir un fichier CSV', type=['csv'])

    if uploaded:
        df_input = pd.read_csv(uploaded)

        if 'comment_text' not in df_input.columns:
            st.error('❌ La colonne `comment_text` est manquante dans le fichier.')
        else:
            st.success(f'✅ Fichier chargé — {len(df_input):,} commentaires')
            st.dataframe(df_input.head(5), use_container_width=True)

            if st.button('Lancer l\'analyse', type='primary'):
                progress_bar = st.progress(0, text='Analyse en cours...')
                texts = df_input['comment_text'].fillna('').tolist()
                n     = len(texts)

                try:
                    # Prédictions batch
                    if batch_model == 'Benchmark (OVR LogReg)':
                        all_probs = benchmark.predict_batch(texts)
                    else:
                        all_probs = teacher.predict_batch(texts)
                    progress_bar.progress(80, text='Assemblage des résultats...')

                    # Construire le DataFrame résultat
                    df_result = df_input.copy()
                    for col in LABEL_COLS:
                        df_result[f'prob_{col}'] = [p[col] for p in all_probs]
                    df_result['toxicity_level'] = [
                        toxicity_level(p)[0] for p in all_probs
                    ]
                    progress_bar.progress(100, text='Terminé ✓')

                    st.divider()
                    st.subheader('Résultats')

                    # Statistiques rapides
                    m1, m2, m3, m4 = st.columns(4)
                    n_toxic = sum(1 for p in all_probs if p['toxic'] > 0.5)
                    n_threat= sum(1 for p in all_probs if p['threat'] > 0.5)
                    n_hate  = sum(1 for p in all_probs if p['identity_hate'] > 0.5)
                    avg_tox = np.mean([p['toxic'] for p in all_probs])

                    m1.metric('Total analysés', f'{n:,}')
                    m2.metric('Toxiques détectés', f'{n_toxic:,}', f'{n_toxic/n*100:.1f}%')
                    m3.metric('Menaces détectées', f'{n_threat:,}')
                    m4.metric('Haine identitaire', f'{n_hate:,}')

                    # Tableau
                    prob_cols = [f'prob_{c}' for c in LABEL_COLS]
                    display_df = df_result[['comment_text', 'toxicity_level'] + prob_cols].copy()
                    for c in prob_cols:
                        display_df[c] = display_df[c].apply(lambda x: f'{x*100:.1f}%')

                    st.dataframe(display_df, use_container_width=True, height=350)

                    # Distribution des niveaux
                    level_counts = df_result['toxicity_level'].value_counts()
                    fig_bar = go.Figure(go.Bar(
                        x=level_counts.index,
                        y=level_counts.values,
                        marker_color=['#A32D2D','#D85A30','#EF9F27','#854F0B','#0F6E56'],
                    ))
                    fig_bar.update_layout(
                        title='Distribution des niveaux de toxicité',
                        xaxis_title='Niveau', yaxis_title='Nombre de commentaires',
                        height=300, margin=dict(l=40, r=20, t=40, b=40),
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Téléchargement
                    csv_bytes = df_result.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label='⬇️ Télécharger les résultats (CSV)',
                        data=csv_bytes,
                        file_name='amane_resultats.csv',
                        mime='text/csv',
                        type='primary',
                    )

                except Exception as e:
                    st.error(f'Erreur durant l\'analyse : {e}')


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Comparaison des modèles
# ════════════════════════════════════════════════════════════════════════════
elif page == 'Comparaison modèles':
    st.title('Comparaison Benchmark vs XLM-R Teacher')

    if not teacher.loaded:
        st.warning(
            '⏳ **XLM-R Teacher non disponible.** '
            'Cette page sera pleinement fonctionnelle après l\'entraînement du Teacher. '
            'Pour l\'instant, tu peux voir la structure de la comparaison avec le Benchmark seul.',
            icon='⚠️'
        )

    # Zone de saisie partagée
    comment = st.text_area(
        'Commentaire à comparer',
        value='You stupid idiot, I will destroy you.',
        height=100,
    )

    if st.button('Comparer', type='primary'):
        if not comment.strip():
            st.warning('Veuillez entrer un commentaire.')
        else:
            with st.spinner('Analyse comparative...'):
                probs_bench = benchmark.predict(comment) if benchmark.loaded else None
                probs_teacher_res = teacher.predict(comment) if teacher.loaded else None

            st.divider()

            col_b, col_t = st.columns(2)

            # ── Colonne Benchmark ──────────────────────────────────────────
            with col_b:
                st.markdown('### Benchmark')
                st.markdown(
                    '<span class="model-badge badge-available">OVR LogReg + TF-IDF</span>',
                    unsafe_allow_html=True
                )
                st.write('')
                if probs_bench:
                    render_probs(probs_bench, 'Benchmark')
                else:
                    st.error('Benchmark non chargé.')

            # ── Colonne Teacher ────────────────────────────────────────────
            with col_t:
                st.markdown('### XLM-R Teacher')
                if teacher.loaded:
                    st.markdown(
                        '<span class="model-badge badge-available">XLM-RoBERTa fine-tuné</span>',
                        unsafe_allow_html=True
                    )
                    st.write('')
                    render_probs(probs_teacher_res, 'Teacher')
                else:
                    st.markdown(
                        '<span class="model-badge badge-pending">⏳ Non entraîné</span>',
                        unsafe_allow_html=True
                    )
                    st.info(
                        'Place les fichiers du Teacher dans `models/teacher/` '
                        'après avoir exécuté `AMANE_Teacher.ipynb`.'
                    )
                    # Placeholder visuel
                    st.write('')
                    for col in LABEL_COLS:
                        meta = LABEL_META[col]
                        c1, c2, c3 = st.columns([3, 6, 1])
                        with c1:
                            st.markdown(
                                f'<span class="label-fr" style="color:#ccc;">○ {meta["fr"]}</span>',
                                unsafe_allow_html=True
                            )
                        with c2:
                            st.progress(0.0)
                        with c3:
                            st.markdown('<span style="font-size:13px;color:#ccc;">—</span>', unsafe_allow_html=True)

            # ── Radar comparatif ───────────────────────────────────────────
            if probs_bench and probs_teacher_res:
                st.divider()
                st.subheader('Radar comparatif')
                st.plotly_chart(
                    comparison_radar(probs_bench, probs_teacher_res),
                    use_container_width=True
                )

                # Tableau des différences
                st.subheader('Différences par label')
                diff_rows = []
                for col in LABEL_COLS:
                    b = probs_bench[col]
                    t = probs_teacher_res[col]
                    d = t - b
                    diff_rows.append({
                        'Label'        : LABEL_META[col]['fr'],
                        'Benchmark'    : f'{b*100:.1f}%',
                        'XLM-R Teacher': f'{t*100:.1f}%',
                        'Différence'   : f'{d*100:+.1f}%',
                        'Avantage'     : 'XLM-R ↑' if d > 0.05 else ('Benchmark ↑' if d < -0.05 else '≈ Égal'),
                    })
                st.dataframe(pd.DataFrame(diff_rows), use_container_width=True, hide_index=True)

            elif probs_bench and not probs_teacher_res:
                st.divider()
                st.subheader('Radar — Benchmark uniquement')
                st.plotly_chart(radar_chart(probs_bench), use_container_width=True)

    # ── Métriques statiques de performance (depuis les notebooks) ──────────
    st.divider()
    st.subheader('Performances sur le set de validation Jigsaw')
    st.caption('Remplir après exécution des deux notebooks.')

    perf_data = {
        'Modèle'        : ['Benchmark (baseline)', 'Benchmark (balanced)', 'XLM-R Teacher'],
        'Macro F1'      : ['—', '—', '⏳'],
        'Micro F1'      : ['—', '—', '⏳'],
        'Hamming Loss'  : ['—', '—', '⏳'],
        'Temps entraîn.': ['< 5 min', '< 5 min', '~2-3h GPU'],
        'Complexité'    : ['Faible', 'Faible', 'Élevée'],
    }
    st.dataframe(pd.DataFrame(perf_data), use_container_width=True, hide_index=True)
    st.caption(
        'Remplace les "—" et "⏳" en modifiant le dict `perf_data` dans `app.py` '
        'une fois les notebooks exécutés.'
    )