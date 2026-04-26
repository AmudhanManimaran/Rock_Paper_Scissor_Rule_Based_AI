# HelioCast — Solar Power Generation Forecasting via Polynomial Ridge Regression & XGBoost

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![NumPy](https://img.shields.io/badge/NumPy-Manual_Ridge-013243?style=flat-square&logo=numpy)
![XGBoost](https://img.shields.io/badge/XGBoost-R²=0.967-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> A real-world solar DC power forecasting system trained on **Kaggle Solar Power Generation (PVDAQ) plant sensor data**. Implements **Polynomial Ridge Regression from scratch** using NumPy closed-form solution and benchmarks it against **XGBoost with GridSearchCV** — achieving **R² = 0.967** on held-out test data. Deployed as a Flask web application.

---

## 📊 Model Results

| Metric | Manual Polynomial Ridge | XGBoost (Best) |
|--------|------------------------|----------------|
| **R² Score** | **0.9607** | **0.9675** |
| MAE (Watts) | 391.28 | 336.22 |
| RMSE (Watts) | 771.85 | 702.18 |
| CV R² (5-fold) | — | 0.9609 |
| Polynomial Degree | 2 | — |
| Ridge Alpha | 10.0 | — |
| Best XGB Params | — | lr=0.1, depth=5, n=500, subsample=0.8 |

**Both models achieve R² > 0.96** — XGBoost deployed as primary model. Manual Ridge implementation demonstrates mathematical equivalence with competitive accuracy.

---

## 🎯 Key Features

- **Real-world PVDAQ dataset** — 68,778 generation records + 3,182 weather sensor readings from an operational solar plant
- **Time-series data fusion** — Generation_Data merged with Weather_Sensor_Data on timestamp
- **Night-time filtering** — rows with Irradiation = 0 excluded (daylight-only training)
- **TEMP_DELTA feature engineering** — thermal drift (Module Temp − Ambient Temp) captures panel heating effect on power output
- **Manual Polynomial Ridge** — degree-2 expansion + closed-form NumPy solution (no sklearn Ridge)
- **Manual metrics** — R², MAE, MSE, RMSE all implemented from scratch
- **XGBoost GridSearchCV** — 5-fold CV across learning rate, depth, estimators, subsampling
- **Physical constraint** — prediction clipped to ≥ 0W (DC power cannot be negative)
- **Flask web interface** — 3-input form (Ambient Temp, Module Temp, Irradiation) → DC Power prediction

---

## 🏗️ System Architecture

```
Generation_Data.csv (68,778 rows)   Weather_Sensor_Data.csv (3,182 rows)
  [DATE_TIME, DC_POWER, ...]          [DATE_TIME, TEMP, IRRADIATION, ...]
            │                                      │
            └──────────── Merge on DATE_TIME ──────┘
                                    │
                                    ▼
                    Filter: Irradiation > 0 (daylight only)
                                    │
                                    ▼
                    Feature Engineering:
                    TEMP_DELTA = MODULE_TEMP − AMBIENT_TEMP
                                    │
                    Features: [AMBIENT_TEMPERATURE, MODULE_TEMPERATURE,
                               IRRADIATION, TEMP_DELTA]
                    Target:   DC_POWER (Watts)
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        Manual Polynomial Ridge            XGBoost GridSearchCV
        (NumPy Closed-Form)                (5-fold CV, R²=0.967)
        R² = 0.961                         R² = 0.967 ← Deployed
                    │                               │
                    └───────────── Flask App ───────┘
                                    │
                             Predict DC Power (W)
                          Physical Clip: max(0, pred)
```

---

## 🧠 Technical Details

### Dataset
| Source | Rows | Description |
|--------|------|-------------|
| Generation_Data.csv | 68,778 | 15-min interval DC/AC power readings from inverters |
| Weather_Sensor_Data.csv | 3,182 | 15-min interval ambient temp, module temp, irradiation |
| **Merged (Daylight)** | **~2,800** | Inner join on timestamp, Irradiation > 0 filtered |

### Input Features
| Feature | Description | Unit |
|---------|-------------|------|
| AMBIENT_TEMPERATURE | Air temperature at plant site | °C |
| MODULE_TEMPERATURE | Solar panel surface temperature | °C |
| IRRADIATION | Solar irradiance (W/m²) | W/m² |
| TEMP_DELTA | MODULE_TEMP − AMBIENT_TEMP (thermal drift) | °C |

### Manual Polynomial Ridge Regression
Implemented entirely in NumPy without sklearn's Ridge class:

```python
# Closed-form Ridge solution
def ridge_fit(X, y, alpha):
    I = np.eye(X.shape[1])
    I[0, 0] = 0   # Don't regularize bias term
    return np.linalg.inv(X.T @ X + alpha * I) @ (X.T @ y)
```

**Pipeline:**
1. Standardize features (training mean/std)
2. Polynomial expansion to degree 2 with pairwise interactions
3. Solve: `w = (XᵀX + αI)⁻¹ Xᵀy`
4. All evaluation metrics (R², MAE, MSE, RMSE) implemented manually

### XGBoost GridSearchCV
```python
param_grid = {
    "n_estimators":     [300, 500],
    "max_depth":        [4, 5],
    "learning_rate":    [0.05, 0.1],
    "subsample":        [0.8],
    "colsample_bytree": [0.8, 1.0],
}
# Best: lr=0.1, depth=5, n=500, colsample=1.0
```

---

## 📁 Project Structure

```
HelioCast/
│
├── solar_efficiency_webapp/
│   ├── app.py                      # Flask app + dual-model inference routing
│   │
│   ├── data/
│   │   ├── Generation_Data.csv     # Real PVDAQ inverter power readings
│   │   └── Weather_Sensor_Data.csv # Real plant weather sensor readings
│   │
│   ├── model/
│   │   ├── train_model.py          # Full training pipeline
│   │   ├── best_model.pkl          # Deployed model (XGBoost)
│   │   ├── best_xgb_model.pkl      # XGBoost model
│   │   ├── ridge_weights.npy       # Manual Ridge weight vector
│   │   ├── ridge_mean.npy          # Standardization mean
│   │   └── ridge_std.npy           # Standardization std
│   │
│   ├── plots/
│   │   ├── parity_ridge_manual.png # Ridge parity plot (R²=0.961)
│   │   └── parity_xgb.png          # XGBoost parity plot (R²=0.967)
│   │
│   ├── results/
│   │   ├── ridge_results.json      # Ridge evaluation metrics
│   │   ├── xgb_results.json        # XGBoost evaluation metrics
│   │   └── model_meta.json         # Best model metadata for dashboard
│   │
│   ├── utils/
│   │   ├── preprocess.py           # Input preprocessing + TEMP_DELTA
│   │   └── visualize.py            # Plot loading utility
│   │
│   ├── templates/
│   │   ├── index.html              # Prediction input form
│   │   ├── result.html             # Prediction result display
│   │   └── dashboard.html          # Model comparison dashboard
│   │
│   └── static/css/ + static/js/
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/AmudhanManimaran/HelioCast.git
cd HelioCast/solar_efficiency_webapp
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Models (Optional — pre-trained models included)
```bash
cd model
python train_model.py
```

### 5. Run the Application
```bash
cd ..   # back to solar_efficiency_webapp/
python app.py
```
Visit `http://localhost:5000` in your browser.

---

## 🚀 Usage

1. Open `http://localhost:5000`
2. Enter 3 weather inputs:
   - **Ambient Temperature** (°C) — air temperature at plant
   - **Module Temperature** (°C) — solar panel surface temperature
   - **Irradiation** (W/m²) — solar irradiance (0 = night-time, no prediction)
3. Click **Predict**
4. View predicted **DC Power Output (Watts)**

---

## 📈 Parity Plots

### Manual Ridge Regression (R² = 0.961)
![Ridge Parity](solar_efficiency_webapp/plots/parity_ridge_manual.png)

### XGBoost (R² = 0.967)
![XGBoost Parity](solar_efficiency_webapp/plots/parity_xgb.png)

---

## 📦 Requirements

```
Flask>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
xgboost>=1.6.0
joblib>=1.1.0
```

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Amudhan Manimaran**
- 🌐 Portfolio: [amudhanmanimaran.github.io/Portfolio](https://amudhanmanimaran.github.io/Portfolio/)
- 💼 LinkedIn: [linkedin.com/in/amudhan-manimaran-3621bb32a](https://www.linkedin.com/in/amudhan-manimaran-3621bb32a)
- 🐙 GitHub: [github.com/AmudhanManimaran](https://github.com/AmudhanManimaran)
