🚘 Auto MPG Prediction – Machine Learning Project

Developed by Amir Mohammadi, a data‑driven professional showcasing end‑to‑end machine learning workflow — covering data preprocessing,
model training, hyperparameter tuning, and deployment with a Tkinter‑based graphical interface featuring a custom red theme (#B22222).
---

 📊 Dataset Overview

- Source: `fuel_effecienvy.xlsx`
- Sheet: `Sheet1`
- Used Columns:  
  `['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']`

 Data Cleaning
- `'horsepower'` column contained `'?'` → replaced with `NaN`.
- Converted to numeric via `pd.to_numeric(errors="coerce")`.
- Missing values imputed using **median** (due to right‑skewed distribution).

---

 🔍 Exploratory Analysis
Visual diagnostics (commented out in code):
- Histogram and BoxPlot for all features  
- Correlation Heatmap  
- ScatterPlot against target (`mpg`)

Observations:
- **Negative impact (reduces MPG):** weight, displacement, cylinders  
- **Positive impact (improves MPG):** newer model year, European/Japanese origin, higher acceleration  

---

🧠 Modeling

Train/Test Split
- `train_test_split(test_size=0.3, random_state=42)`

 Data Scaling
- Standardized features: `StandardScaler` for Linear & SVR  
- Min–Max scaling: `MinMaxScaler` for KNN  

---

 🤖 Models Implemented

| Model | Description | Metrics |
|--------|--------------|----------|
| **Linear Regression** | Baseline model with standard scaling | Predicts MPG linearly |
| **KNN Regressor (n=7, p=1, weights='distance')** | Distance‑weighted neighbor regression | Best among non‑linear distance‑based models |
| **SVR (C=10, degree=2, gamma='scale', kernel='rbf')** | Optimized Support Vector Regression | Final best model (R² ≈ **0.91**) |

All models are evaluated on test set using:
```python
mean_absolute_error, mean_squared_error, r2_score
