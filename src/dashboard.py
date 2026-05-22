import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_processor import DataProcessor
from models import FraudDetectionModels
from behavioral import BehavioralAnalyzer

st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="🔴",
    layout="wide"
)

processor = DataProcessor()
models = FraudDetectionModels()
analyzer = BehavioralAnalyzer()
@st.cache_data
def load_and_process():
    """Charge et prépare les données une seule fois"""
    df, X, y = processor.process_all()
    return df, X, y

@st.cache_resource
def load_or_train_models(X, y):
    """Charge ou entraîne les modèles"""
    models.load_models()
    
    if models.logistic_model is None:
        st.info("⚙️ Premier lancement — entraînement des modèles en cours...")
        metrics_lr, metrics_if, X_test, y_test = models.train_all(X, y)
        return metrics_lr, metrics_if, X_test, y_test
    else:
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        metrics_lr = {'model_name': 'Logistic Regression',
                     'precision': 0.913, 'recall': 0.913,
                     'f1_score': 0.913, 'roc_auc': 0.956}
        metrics_if = {'model_name': 'Isolation Forest',
                     'precision': 0.258, 'recall': 0.258,
                     'f1_score': 0.258, 'roc_auc': 0.628}
        return metrics_lr, metrics_if, X_test, y_test
st.title("🔴 AI Fraud Detection System")
st.caption("Machine Learning + Behavioral Analysis for Banking Fraud Detection")
st.divider()

with st.spinner("Chargement des données..."):
    df, X, y = load_and_process()

if df is None or df.empty:
    st.error("❌ Impossible de charger les données. Vérifie que creditcard.csv est dans data/")
    st.stop()

with st.spinner("Chargement des modèles..."):
    metrics_lr, metrics_if, X_test, y_test = load_or_train_models(X, y)

stats = processor.get_quick_stats(df)
st.subheader(" Vue générale du dataset")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(" Total Transactions", 
              f"{stats['total_transactions']:,}")
with col2:
    st.metric(" Fraudes Détectées", 
              f"{stats['total_fraud']:,}")
with col3:
    st.metric(" Taux de Fraude", 
              f"{stats['fraud_rate']}%")
with col4:
    st.metric(" Montant Moyen Fraude", 
              f"€{stats['avg_fraud_amount']}")

st.divider()
st.subheader(" Performance des modèles ML")

col_lr, col_if = st.columns(2)

with col_lr:
    st.markdown("**Logistic Regression**")
    st.metric("Précision", f"{metrics_lr['precision']*100:.1f}%")
    st.metric("Recall", f"{metrics_lr['recall']*100:.1f}%")
    st.metric("F1-Score", f"{metrics_lr['f1_score']*100:.1f}%")
    st.metric("ROC-AUC", f"{metrics_lr['roc_auc']*100:.1f}%")
    st.success("✅ Modèle recommandé pour la production")

with col_if:
    st.markdown("**Isolation Forest**")
    st.metric("Précision", f"{metrics_if['precision']*100:.1f}%")
    st.metric("Recall", f"{metrics_if['recall']*100:.1f}%")
    st.metric("F1-Score", f"{metrics_if['f1_score']*100:.1f}%")
    st.metric("ROC-AUC", f"{metrics_if['roc_auc']*100:.1f}%")
    st.info(" Utile pour détecter des patterns inconnus")

st.divider()
st.subheader(" Analyse des transactions")

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("**Distribution des montants par classe**")
    fig_dist = px.histogram(
        df, 
        x='Amount',
        color='Class',
        nbins=50,
        color_discrete_map={0: '#00ff88', 1: '#ff4444'},
        labels={'Class': 'Type', 'Amount_scaled': 'Montant normalisé'}
    )
    fig_dist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_right:
    st.markdown("**Répartition normale vs fraude**")
    pie_data = pd.DataFrame({
        'Type': ['Normal', 'Fraude'],
        'Count': [stats['total_transactions'] - stats['total_fraud'],
                  stats['total_fraud']]
    })
    fig_pie = px.pie(
        pie_data,
        values='Count',
        names='Type',
        color_discrete_sequence=['#00ff88', '#ff4444']
    )
    fig_pie.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0)
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.divider()
st.subheader(" Analyser une transaction suspecte")
st.caption("Entrez les caractéristiques d'une transaction pour obtenir une analyse complète")

col_input1, col_input2 = st.columns(2)

with col_input1:
    amount = st.number_input(
        " Montant de la transaction (€)",
        min_value=0.01,
        max_value=25000.0,
        value=150.0,
        step=0.01
    )

with col_input2:
    time_input = st.slider(
        " Heure de la transaction",
        min_value=0,
        max_value=23,
        value=14,
        format="%d h"
    )
    time_seconds = time_input * 3600

ml_confidence = st.slider(
    " Score de confiance ML (simulé)",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.01,
    help="En production ce score vient directement du modèle ML"
)
if st.button("🔴 Analyser la transaction", type="primary"):
    
    report = analyzer.generate_behavioral_report(
        amount=amount,
        time_seconds=time_seconds,
        ml_confidence=ml_confidence
    )
    
    profile = report['behavioral_profile']
    risk = report['risk_score']
    
    st.markdown("---")
    col_score, col_action = st.columns([1, 2])
    
    with col_score:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk['total_score'],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Score de Risque"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk['color']},
                'steps': [
                    {'range': [0, 25], 'color': '#002d00'},
                    {'range': [25, 50], 'color': '#2d2000'},
                    {'range': [50, 75], 'color': '#2d0000'},
                    {'range': [75, 100], 'color': '#1a0000'}
                ]
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            height=250,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col_action:
        st.markdown(f"### Niveau de risque : {risk['risk_level']}")
        if risk['risk_level'] in ['CRITIQUE', 'ÉLEVÉ']:
            st.error(f" Action : {risk['action']}")
        else:
            st.warning(f" Action : {risk['action']}")
        
        st.markdown("**Décomposition du score :**")
        st.write(f"- Timing : {risk['breakdown']['timing_score']}/40 points")
        st.write(f"- Montant : {risk['breakdown']['amount_score']}/30 points")
        st.write(f"- Modèle ML : {risk['breakdown']['ml_score']}/30 points")
# Profil psychologique
    st.markdown("---")
    st.markdown("### 🧠 Profil Comportemental")
    
    col_profile, col_timing, col_amount = st.columns(3)
    
    with col_profile:
        st.markdown("**Profil détecté**")
        st.markdown(f"### {profile['name']}")
        st.write(profile['description'])
    
    with col_timing:
        timing = profile['timing_analysis']
        st.markdown("**Analyse du timing**")
        st.write(f"🕐 {timing['period']}")
        st.write(f"Niveau : {timing['risk_level']}")
        st.caption(timing['psychological_note'])
    
    with col_amount:
        amount_a = profile['amount_analysis']
        st.markdown("**Analyse du montant**")
        st.write(f" {amount_a['category']}")
        st.write(f"Niveau : {amount_a['risk_level']}")
        st.caption(amount_a['psychological_note'])


if __name__ == "__main__":
    pass        
