from data import generate
from train import business_gain

import numpy as np


def test_generated_data_is_consistent():
    df = generate(n=2000, seed=1)
    assert df.customer_id.is_unique
    assert 0.1 < df.churn.mean() < 0.6
    assert df.tenure_months.between(1, 72).all()
    # les contrats mensuels partent davantage
    rates = df.groupby("contract").churn.mean()
    assert rates["Mensuel"] > rates["2 ans"]


def test_business_gain():
    y = np.array([1, 0, 1, 0])
    proba = np.array([0.9, 0.8, 0.2, 0.1])
    # seuil 0.5 : 2 clients ciblés, 1 vrai churner
    assert business_gain(y, proba, 0.5) == 0.35 * 540 - 2 * 40
