import os
import pickle
import joblib
import numpy as np
from flask import Flask, render_template, request

# Import your newly updated preprocessing utility
from utils.preprocess import preprocess_input

app = Flask(__name__)

# ─────────────────────────────────────────────
# 1. Load Model (Handles both Ridge & XGBoost)
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'best_model.pkl')

try:
    # Attempt to load with joblib first (standard for XGBoost)
    model_data = joblib.load(MODEL_PATH)
except Exception:
    try:
        # Fallback to pickle (used for the manual Ridge dictionary)
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
    except Exception as e:
        model_data = None
        print(f"CRITICAL ERROR: Could not load model. {e}")

# Helper function to reconstruct polynomial features for Ridge
def polynomial_features_inference(X, degree=2):
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

# ─────────────────────────────────────────────
# 2. Flask Routes
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Use the preprocess utility to extract and engineer features
        # This handles the TEMP_DELTA math and returns a (1, 4) NumPy array
        input_features = preprocess_input(
            request.form["ambient_temp"],
            request.form["module_temp"],
            request.form["irradiation"]
        )

        # 2. Prediction Logic Routing
        if isinstance(model_data, dict) and model_data.get('type') == 'ridge':
            # --- "White-Box" Manual Ridge Inference ---
            mean_f = model_data['mean']
            std_f  = model_data['std']
            w      = model_data['weights']
            deg    = model_data['degree']
            
            # Standardize using the training mean/std
            X_std = (input_features - mean_f) / std_f
            
            # Apply polynomial expansion
            X_poly = polynomial_features_inference(X_std, deg)
            
            # Mathematical dot product for prediction
            prediction = X_poly @ w
            prediction = prediction[0]
            
        elif model_data is not None:
            # --- XGBoost Inference ---
            prediction = model_data.predict(input_features)[0]
            
        else:
            raise ValueError("Model failed to load on server startup.")

        # 3. Post-Process: Physical constraints (DC Power cannot be negative)
        final_prediction = max(0.0, round(float(prediction), 2))

        return render_template("result.html", prediction=final_prediction)

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)