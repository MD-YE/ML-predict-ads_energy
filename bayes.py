import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import sys

# 1. Load dataset
data = sys.argv[1]
df = pd.read_excel(data, header=0)

# 2. Features & target
X_Eads = df.iloc[:, :9]
y_Eads = df["Eads"]

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_Eads, y_Eads, test_size=0.15, random_state=42)

# 4. Define parameter grid for BayesianRidge
param_grid = {
    'alpha_1': [1e-7, 1e-6, 1e-5, 1e-4],   # 先验分布 Gamma(α1, α2)
    'alpha_2': [1e-7, 1e-6, 1e-5, 1e-4],
    'lambda_1': [1e-7, 1e-6, 1e-5, 1e-4],  # 回归系数先验
    'lambda_2': [1e-7, 1e-6, 1e-5, 1e-4]
}

# 5. Grid search with cross-validation
bayes = BayesianRidge()
grid = GridSearchCV(bayes, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid.fit(X_train, y_train)

print("\nBest BayesianRidge params:")
print(grid.best_params_)
print("Best CV R²:", grid.best_score_)

# 6. Refit best model
best_bayes = grid.best_estimator_
best_bayes.fit(X_train, y_train)

# 7. Train/test performance
y_train_pred = best_bayes.predict(X_train)
y_test_pred = best_bayes.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n=== Performance Metrics ===")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE:  {test_rmse:.4f}")
print(f"Train R²:   {train_r2:.4f}")
print(f"Test R²:    {test_r2:.4f}")

# 8. Cross-validation evaluation
cv_r2_scores = cross_val_score(best_bayes, X_Eads, y_Eads, cv=5, scoring='r2')
cv_rmse_scores = -cross_val_score(best_bayes, X_Eads, y_Eads, cv=5, scoring='neg_root_mean_squared_error')

print("\n=== Cross-Validation Metrics ===")
print("Mean CV R²:", np.mean(cv_r2_scores))
print("Mean CV RMSE:", np.mean(cv_rmse_scores))

# 9. Pearson correlation
r_value, p_value = pearsonr(y_test, y_test_pred)
print("\nPearson r:", r_value)
print("p-value:", p_value)
