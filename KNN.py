import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import sys

# =============================
# 1. 数据读取
# =============================
data = sys.argv[1]
df = pd.read_excel(data, header=0)

X = df.iloc[:, :9]
y = df["Eads"]

# =============================
# 2. 数据拆分
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# =============================
# 3. KNN（9维原始特征）
# =============================
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)

# 预测
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# 交叉验证（5折）
cv_r2_scores = cross_val_score(knn, X, y, cv=5, scoring='r2')
cv_rmse_scores = -cross_val_score(knn, X, y, cv=5, scoring='neg_root_mean_squared_error')

# 计算 Pearson 相关系数
r_value, p_value = pearsonr(y_test, y_test_pred)

print("====== KNN (9 features) ======")
print("Mean CV R²:", np.mean(cv_r2_scores))
print("Mean CV RMSE:", np.mean(cv_rmse_scores))
print("Pearson r:", r_value)
print("p-value:", p_value)
print()

# =============================
# 4. KNN_PCA（降到2维）
# =============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.15, random_state=42
)

knn_pca = KNeighborsRegressor(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train_pca)

# 预测
y_train_pca_pred = knn_pca.predict(X_train_pca)
y_test_pca_pred = knn_pca.predict(X_test_pca)

# 交叉验证
cv_r2_scores_pca = cross_val_score(knn_pca, X_pca, y, cv=5, scoring='r2')
cv_rmse_scores_pca = -cross_val_score(knn_pca, X_pca, y, cv=5, scoring='neg_root_mean_squared_error')

# Pearson 相关系数
r_value_pca, p_value_pca = pearsonr(y_test_pca, y_test_pca_pred)

print("====== KNN_PCA (2 features) ======")
print("Mean CV R²:", np.mean(cv_r2_scores_pca))
print("Mean CV RMSE:", np.mean(cv_rmse_scores_pca))
print("Pearson r:", r_value_pca)
print("p-value:", p_value_pca)
