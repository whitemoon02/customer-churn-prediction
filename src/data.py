"""Génération d'un jeu de données d'abonnés (opérateur télécom) avec un churn expliqué par des facteurs réalistes."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "customers.csv"
TARGET = "churn"
NUMERIC = ["tenure_months", "monthly_charges", "support_calls", "data_usage_gb", "late_payments", "age"]
CATEGORICAL = ["contract", "payment_method", "internet_service", "has_streaming"]


def generate(n: int = 10_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["Mensuel", "1 an", "2 ans"], n, p=[.55, .25, .2])
    df = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "age": rng.integers(18, 85, n),
        "contract": contract,
        "tenure_months": np.where(contract == "Mensuel", rng.gamma(2, 8, n), rng.gamma(4, 9, n)).round().clip(1, 72),
        "payment_method": rng.choice(["Prélèvement", "Carte", "Chèque", "Virement"], n, p=[.45, .3, .1, .15]),
        "internet_service": rng.choice(["Fibre", "ADSL", "Aucun"], n, p=[.5, .35, .15]),
        "has_streaming": rng.choice(["Oui", "Non"], n),
        "support_calls": rng.poisson(1.3, n),
        "late_payments": rng.poisson(0.4, n),
        "data_usage_gb": rng.gamma(2, 25, n).round(1),
    })
    df["monthly_charges"] = (20 + 25 * (df.internet_service == "Fibre") + 12 * (df.internet_service == "ADSL")
                             + 10 * (df.has_streaming == "Oui") + rng.normal(0, 6, n)).round(2).clip(10)
    logit = (-1.6 + 1.5 * (df.contract == "Mensuel") - 0.9 * (df.contract == "2 ans")
             - 0.045 * df.tenure_months + 0.035 * (df.monthly_charges - 50) + 0.45 * df.support_calls
             + 0.5 * df.late_payments + 0.6 * (df.payment_method == "Chèque")
             + 0.4 * (df.internet_service == "Fibre") - 0.01 * (df.age - 45))
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    # Valeurs manquantes réalistes
    df.loc[rng.random(n) < 0.02, "data_usage_gb"] = np.nan
    df.loc[rng.random(n) < 0.01, "payment_method"] = np.nan
    return df


def load() -> pd.DataFrame:
    if not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        generate().to_csv(DATA, index=False)
    return pd.read_csv(DATA)
