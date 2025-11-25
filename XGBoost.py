import numpy as np
import pandas as pd
import sys
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from xgboost import XGBRegressor

# =====================
# 1. 数据加载
# =====================
data = sys.argv[1]
df = pd.read_excel(data, header=0)

# 特征与目标
X_Eads = df.iloc[:, :9]
y_Eads = df[["Eads"]]   # 保持DataFrame结构便于后续扩展

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X_Eads, y_Eads, test_size=0.15, random_state=42
)

# =====================
# 2. 超参数网格搜索优化
# =====================
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.9, 1.0],
    "reg_alpha": [0.0, 0.1, 0.5],
    "reg_lambda": [1.0, 2.0, 5.0],
}

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n=== Best XGBoost Parameters ===")
print(grid_search.best_params_)
print("Best CV R²:", grid_search.best_score_)

# 最优模型
best_xgb = grid_search.best_estimator_

# =====================
# 3. 模型训练与性能评估
# =====================
best_xgb.fit(X_train, y_train)

# 训练集 & 测试集预测
y_train_pred = best_xgb.predict(X_train)
y_test_pred = best_xgb.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n=== Train/Test Performance ===")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE:  {test_rmse:.4f}")
print(f"Train R²:   {train_r2:.4f}")
print(f"Test R²:    {test_r2:.4f}")

# =====================
# 4. 交叉验证
# =====================
cv_r2_scores = cross_val_score(best_xgb, X_Eads, y_Eads, cv=5, scoring='r2')
cv_rmse_scores = -cross_val_score(best_xgb, X_Eads, y_Eads, cv=5, scoring='neg_root_mean_squared_error')

print("\n=== Cross-Validation Results ===")
print(f"Mean CV R²: {np.mean(cv_r2_scores):.4f}")
print(f"Mean CV RMSE: {np.mean(cv_rmse_scores):.4f}")

# =====================
# 5. Pearson 相关系数
# =====================
r_value, p_value = pearsonr(y_test.values.ravel(), y_test_pred.ravel())
print("\n=== Pearson Correlation ===")
print(f"Pearson r: {r_value:.4f}")
print(f"p-value: {p_value:.4e}")
