import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import joblib

df = pd.read_excel("data/fuel_effecienvy.xlsx", sheet_name="Sheet1", usecols=['mpg', 'cylinders', 'displacement',
                                                                         'horsepower', 'weight', 'acceleration',
                                                                         'model year', 'origin'])
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
print(df.head(10).to_string())
# print(df.columns)
# print(df.dtypes)
# print(df["cylinders"].unique())

df['horsepower'].replace('?', np.nan, inplace=True)
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower'].fillna(df['horsepower'].median(), inplace=True)

# ++++++++++  Histogram  +++++++++++++
# plt.figure(figsize=(12, 8))
# for i, col in enumerate(features, 1):
#     plt.subplot(2, 4, i)
#     sns.histplot(df[col], kde=True, color='steelblue')
#     plt.title(col)
# plt.tight_layout()
# plt.show()

# ++++++++++ Correlation+++++++++++++
# corr_matrix = df.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Heatmap", fontsize=14)
# plt.show()
# سه عامل اصلی که بیشترین اثر منفی بر کارایی سوخت دارن:وزن، حجم موتور، تعداد سیلندر.
# سه عامل مثبت که کارایی رو زیاد می‌کنن:سال ساخت، مبدأ ژاپنی/اروپایی، و شتاب بهتر.

# ++++++++++ Boxplot+++++++++++++
# plt.figure(figsize=(12, 8))
# for i, col in enumerate(features, 1):
#     plt.subplot(2, 4, i)
#     sns.boxplot(x=df[col], color='steelblue')
#     plt.title(col, fontsize=11)
#     plt.xlabel('')
# plt.tight_layout()
# plt.show()

# ++++++++++ ScatterPlot+++++++++++++
# plt.figure(figsize=(12, 8))
# for i, col in enumerate(features, 1):
#     plt.subplot(2, 4, i)
#     sns.scatterplot(x=df[col],y='mpg' , data= df)
#     plt.title(col, fontsize=11)
#     plt.xlabel('')
# plt.tight_layout()
# plt.show()

X = df[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']]
y = df['mpg']

# =====  Train/Test =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# =============Normalization By Standard Scaler==============
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# =====  Linear Regression Model =====
lin_model = LinearRegression()
lin_model.fit(X_train_std, y_train)
y_pred_lin = lin_model.predict(X_test_std)

# =====  SVR Model =====
svr_model = SVR(C=10, degree=2, gamma='scale', kernel='rbf')
svr_model.fit(X_train_std, y_train)
y_pred_svr = svr_model.predict(X_test_std)

# =====   KNN Regressor Model =====
minmax_scaler = MinMaxScaler()
X_train_mm = minmax_scaler.fit_transform(X_train)
X_test_mm = minmax_scaler.transform(X_test)

knn_model = KNeighborsRegressor(n_neighbors=7, p=1, weights='distance'
                                )
knn_model.fit(X_train_mm, y_train)
y_pred_knn = knn_model.predict(X_test_mm)


# ============ Metric Calculations ============
def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n--- {model_name} ---")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")


evaluate(y_test, y_pred_lin, "Linear Regression")
evaluate(y_test, y_pred_knn, "KNN Regressor (n=5)")
evaluate(y_test, y_pred_svr, "SVR")



#=========== Optimizing SVR + KNN ===========================

#=========== Creating Pipeline for normalizing + SVR ===================
# svr_pipeline = make_pipeline(StandardScaler(), SVR())
#=========== Defining Grid Parameter===================
# param_grid_svr = {
#     'svr__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'svr__C': [0.1, 1, 10, 100],
#     'svr__gamma': ['scale', 'auto'],
#     'svr__degree': [2, 3, 4]  # فقط برای kernel=poly اثر دارد
# }
# grid_svr = GridSearchCV(
#     estimator=svr_pipeline,
#     param_grid=param_grid_svr,
#     scoring='r2',
#     cv=5,
#     n_jobs=-1
# )
# grid_svr.fit(X_train, y_train)
#
#=========== SVR Best Results ===================
# print("Best SVR Parameters:", grid_svr.best_params_)
# print("Best SVR R² Score:", grid_svr.best_score_)
#
#=========== SVR Optimised Model===================
# best_svr = grid_svr.best_estimator_
# y_pred_svr = best_svr.predict(X_test)


#=========== Defining Basic Model===================
# knn = KNeighborsRegressor()
#
# param_grid_knn = {
#     'n_neighbors': [3, 5, 7, 9, 11, 13],
#     'weights': ['uniform', 'distance'],
#     'p': [1, 2]  # p=1 → Manhattan distance, p=2 → Euclidean distance
# }
#
#=========== Defining GridSearchCV===================
# grid_knn = GridSearchCV(
#     estimator=knn,
#     param_grid=param_grid_knn,
#     scoring='r2',
#     cv=5,
#     n_jobs=-1
# )
# grid_knn.fit(X_train, y_train)
# print("Best KNN Parameters:", grid_knn.best_params_)
# print("Best KNN R² Score:", grid_knn.best_score_)
#
# best_knn = grid_knn.best_estimator_
# y_pred_knn = best_knn.predict(X_test)


# =====================creating Models==============
# joblib.dump(lin_model, "linear_regression_model.joblib")
# joblib.dump(knn_model, "knn_model.joblib")
# joblib.dump(svr_model, "svr_model.joblib")
# joblib.dump(scaler, "scaler.joblib")