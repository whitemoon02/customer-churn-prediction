"""Entraînement : comparaison de modèles en validation croisée, tuning, choix du seuil métier, explicabilité.

Usage : python src/train.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             roc_auc_score, RocCurveDisplay)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CATEGORICAL, NUMERIC, TARGET, load

ROOT = Path(__file__).resolve().parents[1]
MODELS, REPORTS = ROOT / "models", ROOT / "reports"

# Hypothèses métier pour le choix du seuil
CUSTOMER_VALUE = 12 * 45      # marge annuelle d'un client conservé (€)
RETENTION_COST = 40           # coût d'une offre de rétention (€)
RETENTION_SUCCESS = 0.35      # part des clients ciblés qui restent grâce à l'offre


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
    ])


CANDIDATES = {
    "Régression logistique": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                            n_jobs=-1, random_state=0),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=0),
}


def business_gain(y_true, proba, threshold: float) -> float:
    """Gain net de la campagne de rétention si l'on cible les clients au-dessus du seuil."""
    target = proba >= threshold
    saved = (target & (y_true == 1)).sum() * RETENTION_SUCCESS * CUSTOMER_VALUE
    return float(saved - target.sum() * RETENTION_COST)


def main() -> None:
    MODELS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    df = load()
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Taux de churn : {y.mean():.1%} sur {len(df):,} clients")

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = {}
    for name, model in CANDIDATES.items():
        s = cross_val_score(Pipeline([("prep", preprocessor()), ("model", model)]), X_train, y_train,
                            cv=cv, scoring="roc_auc", n_jobs=-1)
        scores[name] = (s.mean(), s.std())
        print(f"  {name:24s} ROC-AUC = {s.mean():.3f} ± {s.std():.3f}")

    # Tuning du modèle de boosting
    search = RandomizedSearchCV(
        Pipeline([("prep", preprocessor()), ("model", HistGradientBoostingClassifier(random_state=0))]),
        {"model__learning_rate": [0.02, 0.05, 0.1], "model__max_depth": [3, 4, 6, None],
         "model__max_leaf_nodes": [15, 31, 63], "model__l2_regularization": [0, 0.1, 1.0],
         "model__max_iter": [200, 400]},
        n_iter=20, scoring="roc_auc", cv=cv, n_jobs=-1, random_state=0)
    search.fit(X_train, y_train)
    print(f"Boosting tuné : {search.best_params_} (CV AUC {search.best_score_:.3f})")

    # Sélection finale : on garde le modèle le plus simple sauf si le boosting tuné fait mieux
    lr_auc = scores["Régression logistique"][0]
    if search.best_score_ > lr_auc + 0.002:
        best, best_name = search.best_estimator_, "Gradient Boosting tuné"
    else:
        best = Pipeline([("prep", preprocessor()), ("model", CANDIDATES["Régression logistique"])]).fit(X_train, y_train)
        best_name = "Régression logistique"
    print(f"Modèle retenu : {best_name}")

    proba = best.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.05, 0.9, 86)
    gains = [business_gain(y_test.values, proba, t) for t in thresholds]
    t_best = float(thresholds[int(np.argmax(gains))])
    pred = (proba >= t_best).astype(int)

    metrics = {
        "model": best_name,
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "threshold": round(t_best, 2),
        "business_gain_eur": round(max(gains)),
        "gain_vs_target_all_eur": round(business_gain(y_test.values, proba, 0.0)),
        "cv_scores": {**{k: round(v[0], 4) for k, v in scores.items()},
                      "Gradient Boosting tuné": round(search.best_score_, 4)},
    }
    print("\n" + classification_report(y_test, pred, target_names=["Reste", "Churn"]))
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Explicabilité : importance par permutation sur les variables d'origine
    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0], name=best_name)
    axes[0].set_title("Courbe ROC")
    axes[1].plot(thresholds, np.array(gains) / 1e3, color="#1f5f8b")
    axes[1].axvline(t_best, ls="--", color="grey")
    axes[1].set(title=f"Gain net de la campagne (seuil optimal {t_best:.2f})", xlabel="Seuil", ylabel="k€")
    axes[2].barh(importance.index, importance.values, color="#2e8b57")
    axes[2].set_title("Importance des variables (permutation)")
    fig.tight_layout(); fig.savefig(REPORTS / "evaluation.png", dpi=120); plt.close(fig)

    cm = confusion_matrix(y_test, pred)
    metrics["confusion_matrix"] = cm.tolist()
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    joblib.dump({"model": best, "threshold": t_best, "features": NUMERIC + CATEGORICAL}, MODELS / "churn_model.joblib")
    print(f"Modèle sauvegardé : {MODELS / 'churn_model.joblib'}")


if __name__ == "__main__":
    main()
