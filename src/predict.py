"""Scoring d'un fichier clients : python src/predict.py data/customers.csv -o reports/scores.csv"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "churn_model.joblib"


def score(df: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    bundle = bundle or joblib.load(MODEL_PATH)
    out = df.copy()
    out["churn_proba"] = bundle["model"].predict_proba(df[bundle["features"]])[:, 1].round(4)
    out["a_cibler"] = out.churn_proba >= bundle["threshold"]
    out["niveau_risque"] = pd.cut(out.churn_proba, [0, .3, .6, 1], labels=["Faible", "Moyen", "Élevé"],
                                  include_lowest=True)
    return out.sort_values("churn_proba", ascending=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input")
    p.add_argument("-o", "--output", default="reports/scores.csv")
    a = p.parse_args()
    res = score(pd.read_csv(a.input))
    res.to_csv(a.output, index=False)
    print(res.niveau_risque.value_counts().to_string())
    print(f"{res.a_cibler.sum()} clients à cibler → {a.output}")
