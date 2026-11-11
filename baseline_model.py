"""
baseline_model.py
-----------------
Trains a Random Forest baseline to predict SOC and generates SHAP plots.
This satisfies the baseline comparison for H1 and the SHAP analysis for H4.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Generate a full synthetic dataset (Features + Target)
def generate_full_synthetic_dataset(n_farms=30, samples_per_farm=7, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for farm_id in range(n_farms):
        farm_bias = rng.normal(0, 0.3) # Farm-specific soil type effect
        for _ in range(samples_per_farm):
            # Satellite & Sensor Features
            ndvi = rng.uniform(0.2, 0.9)
            moisture = rng.uniform(10, 45)
            temp = rng.uniform(5, 30)
            # Synthetic "Truth" (SOC %)
            soc = 0.5 + 1.2 * ndvi + 0.02 * moisture - 0.01 * temp + farm_bias + rng.normal(0, 0.1)
            soc = max(0.1, soc) # SOC cannot be negative
            
            rows.append({
                "farm_id": farm_id,
                "ndvi": ndvi,
                "moisture": moisture,
                "temperature": temp,
                "soc": soc
            })
    return pd.DataFrame(rows)

print("Generating full synthetic dataset...")
df = generate_full_synthetic_dataset()

# 2. Setup Features and Target
X = df[["ndvi", "moisture", "temperature"]]
y = df["soc"]
groups = df["farm_id"]

# 3. Train Baseline Model (Random Forest) using GroupKFold
print("Training Random Forest baseline...")
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# GroupKFold ensures we test on farms the model has never seen
gkf = GroupKFold(n_splits=5)
r2_scores = []

for train_idx, test_idx in gkf.split(X, y, groups):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    r2_scores.append(r2_score(y_test, preds))

print(f"Baseline R² across folds: {np.mean(r2_scores):.3f}")

# 4. SHAP Analysis (for Hypothesis H4)
print("Generating SHAP plots...")
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X)

# Save the SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
print("Saved SHAP plot as 'shap_summary.png'")

# 5. Print feature rankings
mean_shap = np.abs(shap_values).mean(axis=0)
ranking = pd.DataFrame({"Feature": X.columns, "Importance": mean_shap}).sort_values(by="Importance", ascending=False)
print("\nFeature Importance Ranking:")
print(ranking.to_string(index=False))
