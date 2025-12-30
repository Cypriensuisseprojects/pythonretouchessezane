import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="Sézane Analytics Pro", layout="wide")

MOIS_FR = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", 
           "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"]

# Style Sézane
st.markdown("""
    <style>
    [data-testid="stMetricValue"] { color: #D4AF37 !important; font-weight: bold; }
    div[data-testid="stMetric"] {
        border: 1px solid #d4af37 !important;
        padding: 15px !important;
        border-radius: 10px !important;
    }
    .stAlert { border-left: 8px solid #ff4b4b !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    # Chargement
    df = pd.read_csv('SUIVI SERVICES CONCIERGERIE _ PARIS 2 - SUIVI RETOUCHE SEZANE.csv', skiprows=1)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(subset=['DATE CLIENT', 'NOM'], how='all').copy()
    
    # Conversion Dates
    df['DATE CLIENT'] = pd.to_datetime(df['DATE CLIENT'], dayfirst=True, errors='coerce')
    df['DATE DISPO'] = pd.to_datetime(df['DATE DISPO'], dayfirst=True, errors='coerce')
    
    # Correction année
    mask_c = (df['DATE CLIENT'].notna()) & (df['DATE CLIENT'].dt.year < 100)
    df.loc[mask_c, 'DATE CLIENT'] += pd.offsets.DateOffset(years=2000)
    mask_d = (df['DATE DISPO'].notna()) & (df['DATE DISPO'].dt.year < 100)
    df.loc[mask_d, 'DATE DISPO'] += pd.offsets.DateOffset(years=2000)
    
    df = df.dropna(subset=['DATE CLIENT']).copy()
    
    # Nouvelles colonnes KPIs
    df['DELAI'] = (df['DATE DISPO'] - df['DATE CLIENT']).dt.days
    df['MOIS_NUM'] = df['DATE CLIENT'].dt.month
    df['ANNEE'] = df['DATE CLIENT'].dt.year
    df['NOM'] = df['NOM'].fillna('').astype(str).str.upper().str.strip()
    df['PRENOM'] = df['PRENOM'].fillna('').astype(str).str.strip()
    df['CLIENT_FULL'] = df['NOM'] + " " + df['PRENOM']

    def clean_article_name(name):
        name = str(name).capitalize()
        patterns = [
            r" - (FR\s?)?\d{2}$", r" - (T)?\d{2}$", 
            r" - (XS|S|M|L|XL|XXL)$", r" (T|Size\s?)\d{2}$", r" T[3-5][0-9]$"
        ]
        for p in patterns:
            name = re.sub(p, "", name).strip()
        return name

    df['NOM ARTICLE'] = df['NOM ARTICLE'].apply(clean_article_name)
    df['CATE_PRIX'] = df['MONTANT À REGLER'].apply(lambda x: "Payant" if 'PAY' in str(x).upper() or any(char.isdigit() for char in str(x)) else "Offert")
    df['RECUPERE'] = df['RECEPTIONNÉ PAR LE CLIENT'].astype(str).str.upper().str.contains('TRUE|OUI|RECU')
    
    return df

try:
    df = load_and_clean_data()
    st.title("📊 KPI's retouches CG P2")

    # --- FILTRES ---
    st.sidebar.header("Période d'analyse")
    years = sorted(df['ANNEE'].unique(), reverse=True)
    year_target = st.sidebar.selectbox("Année", years)
    month_name = st.sidebar.selectbox("Mois", MOIS_FR, index=datetime.now().month - 1)
    month_target = MOIS_FR.index(month_name) + 1

    tab_year, tab_month, tab_flux, tab_anomalies = st.tabs([
        "📅 Vision Annuelle", "🎯 Focus Mensuel", 
        "🚨 Suivi Flux", "🚩 Anomalies"
    ])

    # --- TAB 1 : VISION ANNUELLE ---
    with tab_year:
        df_year = df[df['ANNEE'] == year_target]
        df_prev = df[df['ANNEE'] == (year_target - 1)]
        
        vol_n, vol_n1 = len(df_year), len(df_prev)
        fidele_n = (df_year['CLIENT_FULL'].value_counts() > 1).sum()
        fidele_n1 = (df_prev['CLIENT_FULL'].value_counts() > 1).sum()
        delai_n = df_year['DELAI'].mean()
        delai_n1 = df_prev['DELAI'].mean()
        payant_n = (len(df_year[df_year['CATE_PRIX'] == 'Payant']) / vol_n * 100) if vol_n > 0 else 0
        payant_n1 = (len(df_prev[df_prev['CATE_PRIX'] == 'Payant']) / vol_n1 * 100) if vol_n1 > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Annuel", f"{vol_n} pces", delta=f"{vol_n - vol_n1} vs N-1")
        c2.metric("Clients Fidèles", f"{fidele_n}", delta=f"{fidele_n - fidele_n1} vs N-1")
        delta_delai = None if pd.isna(delai_n) or pd.isna(delai_n1) else round(delai_n - delai_n1, 1)
        c3.metric("Délai Moyen", f"{delai_n:.1f} j" if not pd.isna(delai_n) else "-", delta=f"{delta_delai} j vs N-1" if delta_delai is not None else None, delta_color="inverse")
        c4.metric("% Service Payant", f"{payant_n:.1f}%", delta=f"{payant_n - payant_n1:.1f}% vs N-1")

        st.markdown("---")
        st.subheader("📈 Saisonnalité des Retouches")
        stats_n = df_year.groupby('MOIS_NUM').size().reindex(range(1, 13), fill_value=0)
        stats_n1 = df_prev.groupby('MOIS_NUM').size().reindex(range(1, 13), fill_value=0)
        fig_season = go.Figure()
        fig_season.add_trace(go.Scatter(x=MOIS_FR, y=stats_n.values, name=f"Année {year_target}", line=dict(color='#D4AF37', width=4), mode='lines+markers'))
        fig_season.add_trace(go.Scatter(x=MOIS_FR, y=stats_n1.values, name=f"Année {year_target-1}", line=dict(color='#E5D3B3', dash='dash')))
        fig_season.update_layout(hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', height=300)
        st.plotly_chart(fig_season, use_container_width=True)

        st.markdown("---")
        col_prod, col_amb = st.columns(2)
        with col_prod:
            st.subheader("👔 Top 10 Catégories (Modèles)")
            top_articles = df_year['NOM ARTICLE'].value_counts().head(10)
            fig_bar = px.bar(top_articles, orientation='h', color_continuous_scale='Gold')
            fig_bar.update_layout(showlegend=False, height=400, xaxis_title="Nombre", yaxis_title="")
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_amb:
            st.subheader("🌟 Top 10 Ambassadeurs")
            top_10_df = df_year['CLIENT_FULL'].value_counts().head(10).reset_index()
            top_10_df.columns = ['Client', 'Nombre de Retouches']
            st.table(top_10_df)

        st.markdown("---")
        st.subheader("💎 Analyses de Structure (Mix Client & Prix)")
        col_mix1, col_mix2 = st.columns(2)
        with col_mix1:
            st.write("**💰 Répartition Payant / Offert**")
            fig_pie_pay = px.pie(df_year, names='CATE_PRIX', hole=0.5, color_discrete_map={'Payant':'#D4AF37','Offert':'#E5D3B3'})
            fig_pie_pay.update_traces(textinfo='percent')
            st.plotly_chart(fig_pie_pay, use_container_width=True)
        with col_mix2:
            st.write("**👥 Nouveaux vs Récurrents**")
            clients_counts = df_year['CLIENT_FULL'].value_counts()
            fig_pie_fid = px.pie(names=['Nouveaux', 'Récurrents'], values=[(clients_counts == 1).sum(), (clients_counts > 1).sum()], hole=0.5, color_discrete_sequence=['#E5D3B3', '#D4AF37'])
            fig_pie_fid.update_traces(textinfo='percent')
            st.plotly_chart(fig_pie_fid, use_container_width=True)

    # --- TAB 2 : FOCUS MENSUEL ---
    with tab_month:
        df_m = df_year[df_year['MOIS_NUM'] == month_target]
        if df_m.empty:
            st.info(f"Aucune donnée pour {month_name} {year_target}.")
        else:
            avg_delai_month = df_m['DELAI'].mean()
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Retouches du mois", len(df_m))
            col_m2.metric("⏳ Délai Moyen", f"{avg_delai_month:.1f} j" if not pd.isna(avg_delai_month) else "-")
            col_m3.metric("Part Payante", f"{(len(df_m[df_m['CATE_PRIX'] == 'Payant'])/len(df_m)*100):.1f}%")

            st.markdown("---")
            cl, cr = st.columns(2)
            with cl:
                st.subheader("🏆 Top Articles du mois")
                st.table(df_m['NOM ARTICLE'].value_counts().head(5))
            with cr:
                st.subheader("💸 Répartition Offert/Payant")
                fig_pie_m = px.pie(df_m, names='CATE_PRIX', hole=0.5, color_discrete_map={'Payant':'#D4AF37','Offert':'#E5D3B3'})
                st.plotly_chart(fig_pie_m, use_container_width=True)

    # --- TAB 3 : SUIVI DES FLUX ---
    with tab_flux:
        st.subheader("📦 Alertes relances & Retouches en cours")
        un_mois_ago = datetime.now() - timedelta(days=30)
        alertes_stock = df[(df['DATE DISPO'].notna()) & (df['RECUPERE'] == False) & (df['DATE DISPO'] < un_mois_ago)].copy()

        if not alertes_stock.empty:
            st.error(f"⚠️ **{len(alertes_stock)} retouches sont en boutique depuis plus de 30 jours !**")
            st.dataframe(alertes_stock[['DATE DISPO', 'NOM', 'N° SOUCHE', 'NOM ARTICLE']].sort_values('DATE DISPO'), use_container_width=True)
            csv = alertes_stock.to_csv(index=False).encode('utf-8')
            st.download_button("📩 Télécharger la liste des relances", csv, "relances.csv", "text/csv")
        else:
            st.success("✅ Aucun article en stock depuis plus de 30 jours.")

        st.markdown("---")
        attente_globale = df[df['DATE DISPO'].isna()].copy()
        st.subheader(f"🧵 Toutes les retouches chez le retoucheur ({len(attente_globale)} pièces)")
        if not attente_globale.empty:
            st.dataframe(attente_globale[['DATE CLIENT', 'NOM', 'N° SOUCHE', 'NOM ARTICLE', 'DESCRIPTIF DE LA RETOUCHE']].sort_values('DATE CLIENT'), use_container_width=True)

    # --- TAB 4 : ANOMALIES ---
    with tab_anomalies:
        st.subheader("🚩 Détection des Anomalies")
        err_chrono = df[df['DATE DISPO'] < df['DATE CLIENT']]
        err_delai = df[df['DELAI'] > 21]
        err_souche = df[df['N° SOUCHE'].isna() | (df['N° SOUCHE'] == "")]
        err_recup = df[(df['RECUPERE'] == True) & (df['DATE DISPO'].isna())]

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            st.write(f"⚠️ **Dates illogiques** : {len(err_chrono)}")
            if not err_chrono.empty: st.dataframe(err_chrono[['DATE CLIENT', 'DATE DISPO', 'NOM', 'N° SOUCHE']])
            st.write(f"⚠️ **Délais > 21 jours** : {len(err_delai)}")
            if not err_delai.empty: st.dataframe(err_delai[['DATE CLIENT', 'DELAI', 'NOM', 'N° SOUCHE']])
        with col_a2:
            st.write(f"⚠️ **Souches manquantes** : {len(err_souche)}")
            if not err_souche.empty: st.dataframe(err_souche[['DATE CLIENT', 'NOM', 'NOM ARTICLE']])
            st.write(f"⚠️ **Récupéré sans date retour** : {len(err_recup)}")
            if not err_recup.empty: st.dataframe(err_recup[['DATE CLIENT', 'NOM', 'N° SOUCHE']])

        st.markdown("---")
        st.subheader("📊 Score de Qualité de Saisie")
        total_erreurs = len(err_chrono) + len(err_souche) + len(err_recup)
        score = max(0, 100 - (total_erreurs / len(df) * 100)) if len(df) > 0 else 100
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            title = {'text': "Fiabilité des données (%)"},
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#D4AF37"}}
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.metric("Taux de fiabilité du fichier", f"{score:.1f}%")

except Exception as e:
    st.error(f"Erreur lors de l'analyse : {e}")