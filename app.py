import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from dateutil.relativedelta import relativedelta

# --- CONFIGURATION ---
st.set_page_config(page_title="Sézane Analytics Pro", layout="wide")

MOIS_FR = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", 
           "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]

# Style Sézane (Ajusté pour laisser le mode nuit fonctionner sur le fond)
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #D4AF37 !important; font-weight: bold; }
    div[data-testid="stMetric"] {
        border: 1px solid #d4af37 !important;
        padding: 15px !important;
        border-radius: 10px !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    df = pd.read_csv('SUIVI SERVICES CONCIERGERIE _ PARIS 2 - SUIVI RETOUCHE SEZANE.csv', skiprows=1)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['DATE CLIENT', 'NOM'], how='all').copy()
    
    # Conversion Dates & Correction année "25"
    df['DATE CLIENT'] = pd.to_datetime(df['DATE CLIENT'], dayfirst=True, errors='coerce')
    df['DATE DISPO'] = pd.to_datetime(df['DATE DISPO'], dayfirst=True, errors='coerce')
    
    mask_c = df['DATE CLIENT'].dt.year < 100
    df.loc[mask_c, 'DATE CLIENT'] += pd.offsets.DateOffset(years=2000)
    
    df = df.dropna(subset=['DATE CLIENT']).copy()
    df['MOIS_NUM'] = df['DATE CLIENT'].dt.month
    df['ANNEE'] = df['DATE CLIENT'].dt.year
    
    # Nettoyage Textes
    df['NOM'] = df['NOM'].fillna('').astype(str).str.upper().str.strip()
    df['CLIENT_FULL'] = df['NOM'] + " " + df['PRENOM'].fillna('').astype(str).str.strip()
    df['NOM ARTICLE'] = df['NOM ARTICLE'].fillna('Inconnu').astype(str)
    
    # Logique Financière & Récupération
    df['CATE_PRIX'] = df['MONTANT À REGLER'].apply(lambda x: "Payant" if 'PAY' in str(x).upper() else "Offert")
    df['RECUPERE'] = df['RECEPTIONNÉ PAR LE CLIENT'].astype(str).str.upper().str.contains('TRUE')
    
    return df

try:
    df = load_and_clean_data()
    st.title("📊 Dashboard Conciergerie Sézane")

    # --- SIDEBAR : FILTRES ANNUELS ---
    st.sidebar.header("Options d'affichage")
    years_available = sorted(df['ANNEE'].unique(), reverse=True)
    year_target = st.sidebar.selectbox("Année à analyser", years_available, index=0)

    # --- SECTION 1 : BILAN ANNUEL ---
    st.header(f"📈 Bilan Annuel {year_target}")
    
    df_year = df[df['ANNEE'] == year_target]
    df_prev_year = df[df['ANNEE'] == (year_target - 1)]
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        total_y = len(df_year)
        total_py = len(df_prev_year)
        delta = ((total_y - total_py) / total_py * 100) if total_py > 0 else 0
        st.metric("Total Annuel", total_y, delta=f"{delta:.1f}% vs {year_target-1}")
    with c2:
        st.metric("Payantes (An)", len(df_year[df_year['CATE_PRIX'] == 'Payant']))
    with c3:
        st.metric("Offertes (An)", len(df_year[df_year['CATE_PRIX'] == 'Offert']))
    with c4:
        retrait = (len(df_year[df_year['RECUPERE']]) / total_y * 100) if total_y > 0 else 0
        st.metric("Taux de Retrait", f"{retrait:.1f}%")

    # --- SECTION 2 : TENDANCE MENSUELLE (N vs N-1) ---
    st.subheader("Tendance Mensuelle : Comparaison des Volumes")
    
    stats_n = df_year.groupby('MOIS_NUM').size().reindex(range(1, 13), fill_value=0)
    stats_n1 = df_prev_year.groupby('MOIS_NUM').size().reindex(range(1, 13), fill_value=0)
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=MOIS_FR, y=stats_n.values, name=f"Année {year_target}",
                                   line=dict(color='#D4AF37', width=4), mode='lines+markers'))
    fig_trend.add_trace(go.Scatter(x=MOIS_FR, y=stats_n1.values, name=f"Année {year_target-1}",
                                   line=dict(color='#E5D3B3', width=2, dash='dash'), mode='lines+markers'))
    
    fig_trend.update_layout(xaxis_title="Mois", yaxis_title="Nombre de retouches",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_trend, width='stretch')

    # --- SECTION 3 : FOCUS MENSUEL DÉTAILLÉ ---
    st.markdown("---")
    st.header("🔍 Focus Mensuel")
    
    latest_month = int(df_year['MOIS_NUM'].max()) if not df_year.empty else 1
    nom_mois = st.selectbox("Choisir un mois", MOIS_FR, index=latest_month - 1)
    month_num = MOIS_FR.index(nom_mois) + 1
    
    df_m = df_year[df_year['MOIS_NUM'] == month_num]
    
    if df_m.empty:
        st.warning(f"Aucune donnée pour {nom_mois} {year_target}")
    else:
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.subheader("🏆 Top Articles")
            top_articles = df_m['NOM ARTICLE'].value_counts().head(5).reset_index()
            top_articles.columns = ['Article', 'Nombre']
            st.table(top_articles)
        with col_m2:
            st.subheader("💸 Répartition")
            fig_pie = px.pie(df_m, names='CATE_PRIX', hole=0.5, 
                             color_discrete_map={'Payant':'#D4AF37','Offert':'#E5D3B3'})
            st.plotly_chart(fig_pie, width='stretch')

    # --- SECTION 4 : ALERTES FLUX (Conditionnel 2024+) ---
    if year_target >= 2024:
        st.markdown("---")
        st.subheader("🚨 Suivi des Flux (Boutique)")
        t1, t2 = st.tabs(["⏳ En attente de DISPO", "📦 En stock (Non réceptionné)"])
        with t1:
            attente = df_m[df_m['DATE DISPO'].isna()]
            st.dataframe(attente[['DATE CLIENT', 'NOM', 'NOM ARTICLE', 'N° SOUCHE']], width='stretch')
        with t2:
            en_stock = df[df['DATE DISPO'].notna() & (df['RECUPERE'] == False)]
            st.dataframe(en_stock[['DATE DISPO', 'NOM', 'N° SOUCHE', 'NOM ARTICLE']], width='stretch')

except Exception as e:
    st.error(f"Erreur de chargement : {e}")