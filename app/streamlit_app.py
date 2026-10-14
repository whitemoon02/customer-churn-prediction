"""Application de démonstration : streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from predict import score  # noqa: E402

st.set_page_config(page_title="Prédiction du churn", page_icon="📉", layout="wide")
bundle = joblib.load(ROOT / "models" / "churn_model.joblib")

st.title("📉 Prédiction du churn client")
tab1, tab2 = st.tabs(["Client individuel", "Scoring d'un fichier"])

with tab1:
    c1, c2, c3 = st.columns(3)
    client = {
        "age": c1.slider("Âge", 18, 85, 40),
        "tenure_months": c1.slider("Ancienneté (mois)", 1, 72, 12),
        "contract": c1.selectbox("Contrat", ["Mensuel", "1 an", "2 ans"]),
        "monthly_charges": c2.number_input("Facture mensuelle (€)", 10.0, 120.0, 55.0),
        "payment_method": c2.selectbox("Paiement", ["Prélèvement", "Carte", "Chèque", "Virement"]),
        "internet_service": c2.selectbox("Internet", ["Fibre", "ADSL", "Aucun"]),
        "has_streaming": c3.selectbox("Option streaming", ["Oui", "Non"]),
        "support_calls": c3.number_input("Appels au support (an)", 0, 15, 1),
        "late_payments": c3.number_input("Retards de paiement", 0, 10, 0),
        "data_usage_gb": c3.number_input("Data (Go / mois)", 0.0, 300.0, 40.0),
    }
    res = score(pd.DataFrame([client]), bundle).iloc[0]
    st.metric("Probabilité de départ", f"{res.churn_proba:.0%}", res.niveau_risque)
    if res.a_cibler:
        st.warning("Client à cibler par une offre de rétention.")
    else:
        st.success("Risque sous le seuil de ciblage.")

with tab2:
    f = st.file_uploader("CSV clients (mêmes colonnes que data/customers.csv)", type="csv")
    if f:
        scored = score(pd.read_csv(f), bundle)
        st.bar_chart(scored.niveau_risque.value_counts())
        st.dataframe(scored.head(200), use_container_width=True)
        st.download_button("Télécharger les scores", scored.to_csv(index=False), "scores.csv")
