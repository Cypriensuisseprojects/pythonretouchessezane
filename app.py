import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

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
    df['NOM ARTICLE'] = df['NOM ARTICLE'].fillna('Inconnu').astype(str).str.capitalize()
    
    # Catégorisation Prix
    df['CATE_PRIX'] = df['MONTANT À REGLER'].apply(lambda x: "Payant" if 'PAY' in str(x).upper() or any(char.isdigit() for char in str(x)) else "Offert")
    
    # Gestion Récupération
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

    tab_year, tab_month, tab_flux = st.tabs(["📅 Vision Annuelle", "🎯 Focus Mensuel", "🚨 Alertes relances & suivi retouches chez AM retouches"])

# --- TAB 1 : VISION ANNUELLE ---
    with tab_year:
        df_year = df[df['ANNEE'] == year_target]
        df_prev = df[df['ANNEE'] == (year_target - 1)]
        
        # --- CALCULS N vs N-1 ---
        # 1. Volume
        vol_n = len(df_year)
        vol_n1 = len(df_prev)
        
        # 2. Fidélité (Clients récurrents)
        fidele_n = len(df_year['CLIENT_FULL'].value_counts()[df_year['CLIENT_FULL'].value_counts() > 1])
        fidele_n1 = len(df_prev['CLIENT_FULL'].value_counts()[df_prev['CLIENT_FULL'].value_counts() > 1])
        
        # 3. Délai Moyen
        delai_n = df_year['DELAI'].mean()
        delai_n1 = df_prev['DELAI'].mean()
        
        # 4. Part Payante (%)
        payant_n = (len(df_year[df_year['CATE_PRIX'] == 'Payant']) / vol_n * 100) if vol_n > 0 else 0
        payant_n1 = (len(df_prev[df_prev['CATE_PRIX'] == 'Payant']) / vol_n1 * 100) if vol_n1 > 0 else 0

        # --- Row 1 : Métriques Clés avec Deltas ---
        c1, c2, c3, c4 = st.columns(4)
        
        c1.metric("Total Annuel", f"{vol_n} pces", 
                  delta=f"{vol_n - vol_n1} vs N-1")
        
        c2.metric("Clients Fidèles", f"{fidele_n}", 
                  delta=f"{fidele_n - fidele_n1} vs N-1",
                  help="Nombre de clients revenus au moins 2 fois dans l'année")
        
        # Delta délai : négatif = vert (car délai plus court)
        delta_delai = None if pd.isna(delai_n) or pd.isna(delai_n1) else round(delai_n - delai_n1, 1)
        c3.metric("Délai Moyen Annuel", f"{delai_n:.1f} j" if not pd.isna(delai_n) else "-", 
                  delta=f"{delta_delai} j vs N-1" if delta_delai is not None else None,
                  delta_color="inverse")
        
        c4.metric("% Service Payant", f"{payant_n:.1f}%", 
                  delta=f"{payant_n - payant_n1:.1f}% vs N-1")

        # --- Row 2 : Graphiques de tendance ---
        st.markdown("---")
        col_g1, col_g2 = st.columns([2, 1])
        
        with col_g1:
            st.subheader("📈 Saisonnalité des Retouches")
            stats_n = df_year.groupby('MOIS_NUM').size().reindex(range(1, 13), fill_value=0)
            stats_n1 = df_prev.groupby('MOIS_NUM').size().reindex(range(1, 13), fill_value=0)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=MOIS_FR, y=stats_n.values, name=f"Année {year_target}", line=dict(color='#D4AF37', width=4), mode='lines+markers'))
            fig.add_trace(go.Scatter(x=MOIS_FR, y=stats_n1.values, name=f"Année {year_target-1}", line=dict(color='#E5D3B3', dash='dash')))
            fig.update_layout(hovermode="x unified", plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

        with col_g2:
            st.subheader("👔 Top Catégories")
            top_articles = df_year['NOM ARTICLE'].value_counts().head(8)
            fig_bar = px.bar(top_articles, orientation='h', color_continuous_scale='Gold')
            fig_bar.update_layout(showlegend=False, xaxis_title="Nombre", yaxis_title="")
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- Row 3 : Analyse de la Valeur ---
        st.markdown("---")
        st.subheader("💎 Analyse de la Fidélité Client")
        c_fid1, c_fid2 = st.columns(2)
        
        clients_counts = df_year['CLIENT_FULL'].value_counts()
        with c_fid1:
            # Répartition Nouveaux vs Récurrents
            labels = ['Nouveaux Clients', 'Clients Récurrents']
            values = [len(clients_counts[clients_counts == 1]), len(clients_counts[clients_counts > 1])]
            fig_rec = px.pie(names=labels, values=values, hole=0.6, title="Répartition de la base client",
                             color_discrete_sequence=['#E5D3B3', '#D4AF37'])
            st.plotly_chart(fig_rec, use_container_width=True)
            
        with c_fid2:
            # Top clients de l'année
            st.write("🌟 **Top 10 Ambassadeurs (Nb retouches)**")
            top_10_clients = clients_counts.head(10).reset_index()
            top_10_clients.columns = ['Client', 'Nombre de Retouches']
            st.table(top_10_clients)

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
                fig_pie = px.pie(df_m, names='CATE_PRIX', hole=0.5, color_discrete_map={'Payant':'#D4AF37','Offert':'#E5D3B3'})
                st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 3 : SUIVI DES FLUX ---
    with tab_flux:
        st.subheader("📦 Alertes relances & Retouches en cours")
        
        # 1. ALERTES RELANCES (Articles prêts non récupérés > 30j)
        un_mois_ago = datetime.now() - timedelta(days=30)
        alertes_stock = df[(df['DATE DISPO'].notna()) & 
                           (df['RECUPERE'] == False) & 
                           (df['DATE DISPO'] < un_mois_ago)].copy()

        if not alertes_stock.empty:
            st.error(f"⚠️ **{len(alertes_stock)} retouches sont en boutique depuis plus de 30 jours !**")
            st.dataframe(alertes_stock[['DATE DISPO', 'NOM', 'N° SOUCHE', 'NOM ARTICLE']].sort_values('DATE DISPO'), use_container_width=True)
            csv = alertes_stock.to_csv(index=False).encode('utf-8')
            st.download_button("📩 Télécharger la liste des relances", csv, "relances_retouches_sézane.csv", "text/csv")
        else:
            st.success("✅ Aucun article en stock depuis plus de 30 jours.")

        st.markdown("---")
        
        # 2. TOUTES LES RETOUCHES EN ATTENTE (Chez le retoucheur)
        # On utilise 'df' (toutes les données) et non 'df_m' (mois filtré)
        attente_globale = df[df['DATE DISPO'].isna()].copy()
        
        st.subheader(f"🧵 Toutes les retouches chez le retoucheur ({len(attente_globale)} pièces)")
        st.warning("Ces articles ont été envoyés mais n'ont pas encore de date de retour enregistrée.")
        
        if not attente_globale.empty:
            # On trie par date client pour voir les plus anciennes en premier
            st.dataframe(
                attente_globale[['DATE CLIENT', 'NOM', 'N° SOUCHE', 'NOM ARTICLE', 'DESCRIPTIF DE LA RETOUCHE']]
                .sort_values('DATE CLIENT'), 
                use_container_width=True
            )
        else:
            st.success("✅ Aucune retouche en attente de retour chez le retoucheur.")

        st.markdown("---")
        
        # 3. STOCK GLOBAL PRÊT (Pour info)
        with st.expander("📦 Voir tout le stock prêt en boutique (tous mois confondus)"):
            stock_boutique = df[(df['DATE DISPO'].notna()) & (df['RECUPERE'] == False)]
            st.dataframe(stock_boutique[['DATE DISPO', 'NOM', 'N° SOUCHE', 'NOM ARTICLE']], use_container_width=True)

except Exception as e:
    st.error(f"Erreur lors de l'analyse : {e}")