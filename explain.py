"""
explain.py
----------
SHAP explainability for SOC models.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import shap

def shap_tree(model, X_test: pd.DataFrame):
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X_test)

def rank_features(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
