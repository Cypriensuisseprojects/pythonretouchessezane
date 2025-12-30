import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import os

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
def load_and_clean_data(file_source):
    # Chargement dynamique
    df = pd.read_csv(file_source, skiprows=1)
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

# --- GESTION DES FICHIERS ---
st.sidebar.header("📁 Données Source")
fichiers_locaux = [f for f in os.listdir('.') if f.endswith('.csv')]
fichier_defaut = 'SUIVI SERVICES CONCIERGERIE _ PARIS 2 - SUIVI RETOUCHE SEZANE.csv'

# On propose les fichiers présents ou l'upload
selection_fichier = st.sidebar.selectbox("Choisir un fichier projet", fichiers_locaux if fichiers_locaux else ["Aucun fichier trouvé"])
uploaded_file = st.sidebar.file_uploader("Ou importer un nouveau CSV", type="csv")

# Priorité : 1. Upload, 2. Sélection, 3. Fichier par défaut
final_source = uploaded_file if uploaded_file else selection_fichier

try:
    df = load_and_clean_data(final_source)
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
        fig_season.add_trace(go.Scatter(x=MOIS_FR, y=stats_n.values, name=f"Année {year_target}", line=dict(color='#D4AF37', width=4), mode='lines+markers+text', text=stats_n.values, textposition="top center"))
        fig_season.add_trace(go.Scatter(x=MOIS_FR, y=stats_n1.values, name=f"Année {year_target-1}", line=dict(color='#E5D3B3', width=2, dash='dot'), mode='lines+markers'))
        
        fig_season.update_layout(hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)', height=450, yaxis=dict(title="Nombre de pièces", showgrid=True, gridcolor='rgba(200, 200, 200, 0.3)', dtick=50, minor=dict(dtick=10, showgrid=True, gridcolor='rgba(200, 200, 200, 0.1)')), xaxis=dict(showgrid=False), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
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

    # --- TAB 2 : FOCUS MENSUEL ---
    with tab_month:
        df_m = df_year[df_year['MOIS_NUM'] == month_target]
        if df_m.empty:
            st.info(f"Aucune donnée pour {month_name} {year_target}.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            payant_pct = (len(df_m[df_m['CATE_PRIX'] == 'Payant']) / len(df_m) * 100) if len(df_m) > 0 else 0
            avg_delai = df_m['DELAI'].mean()
            m1.metric("Volume Traité", f"{len(df_m)} pces")
            m2.metric("Délai de Livraison", f"{avg_delai:.1f} j" if not pd.isna(avg_delai) else "-")
            m3.metric("Performance Vente", f"{payant_pct:.1f}% Payant")
            m4.metric("En attente client", f"{len(df_m[df_m['RECUPERE'] == False])} pces")

            st.markdown("---")
            cl, cr = st.columns(2)
            with cl:
                st.subheader("⏱️ Respect du contrat délai (SLA)")
                bins = [-1, 3, 7, 10, 15, 1000]
                labels = ['Express (0-3j)', 'Standard (4-7j)', 'Tendu (8-10j)', 'Retard (11-15j)', 'Critique (>15j)']
                df_temp = df_m.dropna(subset=['DELAI']).copy()
                if not df_temp.empty:
                    df_temp['TRANCHE_DELAI'] = pd.cut(df_temp['DELAI'], bins=bins, labels=labels)
                    fig_delai = px.bar(df_temp['TRANCHE_DELAI'].value_counts().reindex(labels), color_discrete_sequence=['#D4AF37'])
                    st.plotly_chart(fig_delai, use_container_width=True)
            with cr:
                st.subheader("👗 Top 5 Modèles retouchés")
                top_m = df_m['NOM ARTICLE'].value_counts().head(5).reset_index()
                top_m.columns = ['Modèle', 'Volume']
                fig_p = px.pie(top_m, values='Volume', names='Modèle', hole=0.4, color_discrete_sequence=px.colors.sequential.YlOrBr)
                st.plotly_chart(fig_p, use_container_width=True)

            st.markdown("---")
            st.subheader("📑 Détail des opérations du mois")
            search = st.text_input("Rechercher client, souche ou article...", key="search_m")
            display_df = df_m[['DATE CLIENT', 'DATE DISPO', 'NOM', 'N° SOUCHE', 'NOM ARTICLE', 'CATE_PRIX', 'DELAI']]
            if search:
                display_df = display_df[display_df.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
            st.dataframe(display_df.sort_values('DATE CLIENT', ascending=False), use_container_width=True)

    # --- TAB 3 : SUIVI DES FLUX ---
    with tab_flux:
        st.subheader("📦 Alertes relances & Retouches en cours")
        un_an_ago = datetime.now() - timedelta(days=365)
        un_mois_ago = datetime.now() - timedelta(days=30)
        
        alertes_critiques = df[(df['DATE DISPO'].notna()) & (df['RECUPERE'] == False) & (df['DATE DISPO'] < un_an_ago)].copy()
        alertes_stock = df[(df['DATE DISPO'].notna()) & (df['RECUPERE'] == False) & (df['DATE DISPO'] < un_mois_ago) & (df['DATE DISPO'] >= un_an_ago)].copy()

        if not alertes_critiques.empty:
            st.error(f"🚨 **ALERTE CRITIQUE : {len(alertes_critiques)} retouches de plus d'un AN !**")
            st.dataframe(alertes_critiques[['DATE DISPO', 'NOM', 'N° SOUCHE', 'NOM ARTICLE']].sort_values('DATE DISPO'), use_container_width=True)

        if not alertes_stock.empty:
            st.warning(f"⚠️ **{len(alertes_stock)} retouches de plus de 30 jours.**")
            st.dataframe(alertes_stock[['DATE DISPO', 'NOM', 'N° SOUCHE', 'NOM ARTICLE']].sort_values('DATE DISPO'), use_container_width=True)

        # --- EXPORT INTELLIGENT ---
        if not alertes_stock.empty or not alertes_critiques.empty:
            df_relances = pd.concat([alertes_critiques, alertes_stock])
            col_email = next((c for c in df.columns if 'EMAIL' in c.upper() or 'MAIL' in c.upper()), None)
            
            def format_relance(row):
                email = str(row[col_email]).strip() if col_email and pd.notna(row[col_email]) else ""
                if email and "@" in email:
                    return pd.Series({'NOM': row['NOM'], 'PRENOM': row.get('PRENOM', ''), 'EMAIL': email, 'SOUCHE': row['N° SOUCHE'], 'ACTION': 'EMAIL'})
                else:
                    res = row.copy()
                    res['ACTION'] = '⚠️ APPEL (Email manquant)'
                    return res

            df_export = df_relances.apply(format_relance, axis=1)
            csv = df_export.to_csv(index=False).encode('utf-8-sig')
            st.download_button("📩 Télécharger la liste de contacts (Emails prioritaires)", csv, f"relances_sezane_{datetime.now().strftime('%d_%m')}.csv", "text/csv")
        else:
            st.success("✅ Aucun article en attente de relance.")

        st.markdown("---")
        attente_globale = df[df['DATE DISPO'].isna()].copy()
        st.subheader(f"🧵 Chez le retoucheur ({len(attente_globale)} pièces)")
        if not attente_globale.empty:
            st.dataframe(attente_globale[['DATE CLIENT', 'NOM', 'N° SOUCHE', 'NOM ARTICLE']].sort_values('DATE CLIENT'), use_container_width=True)

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
        total_erreurs = len(err_chrono) + len(err_souche) + len(err_recup)
        score = max(0, 100 - (total_erreurs / len(df) * 100)) if len(df) > 0 else 100
        fig_gauge = go.Figure(go.Indicator(mode = "gauge+number", value = score, title = {'text': "Fiabilité des données (%)"}, gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#D4AF37"}}))
        st.plotly_chart(fig_gauge, use_container_width=True)

except Exception as e:
    st.error(f"Erreur lors de l'analyse : {e}")