"""
HelioCast — Solar Power Generation Forecasting
train_model.py

Training pipeline for real-world NREL/Kaggle PVDAQ Data:
1. Time-series data fusion (Generation + Weather)
2. Manual Polynomial Ridge Regression (NumPy baseline)
3. XGBoost Regressor (Optimized for real-world weather manifolds)
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ─────────────────────────────────────────────
# 1. Bulletproof Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

SAVED_MODELS_DIR = BASE_DIR
PLOTS_DIR        = os.path.join(ROOT_DIR, "plots")
RESULTS_DIR      = os.path.join(ROOT_DIR, "results") # Fixed to save in root/results

GEN_DATA_PATH    = os.path.join(ROOT_DIR, "data", "Generation_data.csv")
WTHR_DATA_PATH   = os.path.join(ROOT_DIR, "data", "Weather_Sensor_Data.csv")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 2. Load, Merge & Feature Engineering
# ─────────────────────────────────────────────
print("="*60)
print(" STEP 1: Data Fusion & Preprocessing")
print("="*60)

gen_df = pd.read_csv(GEN_DATA_PATH)
weather_df = pd.read_csv(WTHR_DATA_PATH)

# Standardize DateTime
gen_df['DATE_TIME'] = pd.to_datetime(gen_df['DATE_TIME'], format='%d-%m-%Y %H:%M')
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'], format='%Y-%m-%d %H:%M:%S')

# Extract weather metrics
weather_metrics = weather_df[['DATE_TIME', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]

# Merge datasets
df = pd.merge(gen_df, weather_metrics, on='DATE_TIME', how='inner')

# Filter out night-time data (Irradiation = 0)
df = df[df['IRRADIATION'] > 0.0]

# Feature Engineering: Thermal Drift
df['TEMP_DELTA'] = df['MODULE_TEMPERATURE'] - df['AMBIENT_TEMPERATURE']

print(f"Merged Dataset Shape (Daylight Only): {df.shape}")

# Define X and y
features = ['AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION', 'TEMP_DELTA']
X = df[features]
y = df['DC_POWER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ─────────────────────────────────────────────
# 3. Manual Polynomial Ridge Regression
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 2: Polynomial Ridge Regression (Manual)")
print("="*60)

X_train_np, X_test_np = X_train.to_numpy(), X_test.to_numpy()
y_train_np, y_test_np = y_train.to_numpy(), y_test.to_numpy()

# Standardize
mean_features = np.mean(X_train_np, axis=0)
std_features  = np.std(X_train_np, axis=0)
std_features[std_features == 0] = 1

X_train_std = (X_train_np - mean_features) / std_features
X_test_std  = (X_test_np - mean_features) / std_features

def polynomial_features(X, degree=2):
    n_samples, n_features = X.shape
    feats = [np.ones(n_samples)]
    for deg in range(1, degree + 1):
        for i in range(n_features):
            feats.append(X[:, i] ** deg)
    if degree >= 2:
        for i in range(n_features):
            for j in range(i + 1, n_features):
                feats.append(X[:, i] * X[:, j])
    return np.vstack(feats).T

degree = 2
ridge_alpha = 10.0

X_train_poly = polynomial_features(X_train_std, degree)
X_test_poly  = polynomial_features(X_test_std, degree)

def ridge_fit(X, y, alpha):
    I = np.eye(X.shape[1])
    I[0, 0] = 0 # Don't regularize bias
    return np.linalg.inv(X.T @ X + alpha * I) @ (X.T @ y)

w_ridge = ridge_fit(X_train_poly, y_train_np, ridge_alpha)

# Ridge Metrics
y_pred_test_ridge = X_test_poly @ w_ridge
r2_ridge   = r2_score(y_test_np, y_pred_test_ridge)
mae_ridge  = mean_absolute_error(y_test_np, y_pred_test_ridge)
mse_ridge  = mean_squared_error(y_test_np, y_pred_test_ridge)
rmse_ridge = np.sqrt(mse_ridge)

print(f"Ridge Test R²: {r2_ridge:.4f}")

# Save Ridge Artifacts
np.save(os.path.join(SAVED_MODELS_DIR, "ridge_weights.npy"), w_ridge)
np.save(os.path.join(SAVED_MODELS_DIR, "ridge_mean.npy"), mean_features)
np.save(os.path.join(SAVED_MODELS_DIR, "ridge_std.npy"), std_features)

# Format & Save Ridge JSON
ridge_results = {
    "Best Params": {
        "degree": degree,
        "alpha": ridge_alpha
    },
    "R2": r2_ridge,
    "MAE": mae_ridge,
    "MSE": mse_ridge,
    "RMSE": rmse_ridge
}
with open(os.path.join(RESULTS_DIR, "ridge_results.json"), "w") as f:
    json.dump(ridge_results, f, indent=4)

# Plot Parity for Ridge
plt.figure(figsize=(6,6))
plt.scatter(y_test_np, y_pred_test_ridge, alpha=0.3, edgecolor='k')
plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--', lw=2)
plt.title(f"Manual Ridge Parity (R²={r2_ridge:.3f})")
plt.xlabel("Actual DC Power")
plt.ylabel("Predicted DC Power")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "parity_ridge_manual.png"), dpi=150)
plt.close()

# ─────────────────────────────────────────────
# 4. XGBoost Regressor
# ─────────────────────────────────────────────
print("\n" + "="*60)
print(" STEP 3: XGBoost Regressor")
print("="*60)

xgb_model = XGBRegressor(objective="reg:squarederror", tree_method="hist", random_state=42)

param_grid_xgb = {
    "n_estimators": [300, 500],
    "max_depth": [4, 5],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8, 1.0],
}

grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring="r2", n_jobs=-1)
grid_xgb.fit(X_train, y_train)

best_xgb = grid_xgb.best_estimator_

# XGBoost Metrics
y_pred_test_xgb = best_xgb.predict(X_test)
r2_xgb   = r2_score(y_test, y_pred_test_xgb)
mae_xgb  = mean_absolute_error(y_test, y_pred_test_xgb)
mse_xgb  = mean_squared_error(y_test, y_pred_test_xgb)
rmse_xgb = np.sqrt(mse_xgb)

print("Calculating Cross-Validation R2 (This takes a few seconds)...")
cv_r2_xgb = cross_val_score(best_xgb, X, y, cv=5, scoring="r2", n_jobs=-1).mean()

print(f"Best Params: {grid_xgb.best_params_}")
print(f"XGBoost Test R²: {r2_xgb:.4f}")

joblib.dump(best_xgb, os.path.join(SAVED_MODELS_DIR, "best_xgb_model.pkl"))

# Format & Save XGBoost JSON
xgb_results = {
    "Best Params": grid_xgb.best_params_,
    "R2": r2_xgb,
    "MAE": mae_xgb,
    "MSE": mse_xgb,
    "RMSE": rmse_xgb,
    "CV_R2": cv_r2_xgb
}
with open(os.path.join(RESULTS_DIR, "xgb_results.json"), "w") as f:
    json.dump(xgb_results, f, indent=4)

# Plot Parity for XGBoost
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_test_xgb, alpha=0.3, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f"XGBoost Parity (R²={r2_xgb:.3f})")
plt.xlabel("Actual DC Power")
plt.ylabel("Predicted DC Power")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "parity_xgb.png"), dpi=150)
plt.close()

# ─────────────────────────────────────────────
# 5. Final Selection & Meta JSON (For Dashboard)
# ─────────────────────────────────────────────
if r2_ridge > r2_xgb:
    bundle = {"type": "ridge", "weights": w_ridge, "mean": mean_features, "std": std_features, "degree": degree}
    with open(os.path.join(SAVED_MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump(bundle, f)
    best_model_name = "Ridge (Manual)"
    best_r2 = r2_ridge
    best_rmse = rmse_ridge
    print("\nSaved Ridge (Manual) as primary model.")
else:
    joblib.dump(best_xgb, os.path.join(SAVED_MODELS_DIR, "best_model.pkl"))
    best_model_name = "XGBoost"
    best_r2 = r2_xgb
    best_rmse = rmse_xgb
    print("\nSaved XGBoost as primary model.")

# Save Meta JSON for the Web Dashboard
meta = {
    "Model_Type": best_model_name,
    "R2_Score": round(best_r2, 4),
    "MSE": round(best_rmse**2, 4), # Converting RMSE back to MSE for the UI
    "RMSE": round(best_rmse, 4)
}
with open(os.path.join(RESULTS_DIR, "model_meta.json"), "w") as f:
    json.dump(meta, f, indent=4)