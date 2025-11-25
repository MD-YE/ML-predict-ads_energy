import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import sys

# 1. 读取数据
data = sys.argv[1]  # 例如 "Dataset_Eads_H_scaled.xlsx"
df = pd.read_excel(data, header=0)

X = df.iloc[:, :9]   # 9个特征
y = df["Eads"]       # 回归目标

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# ===============================
# 3. SVR with RBF kernel
svr_rbf = SVR(kernel="rbf")
svr_rbf.fit(X_train, y_train)

y_train_pred_rbf = svr_rbf.predict(X_train)
y_test_pred_rbf = svr_rbf.predict(X_test)

train_r2_rbf = r2_score(y_train, y_train_pred_rbf)
test_r2_rbf = r2_score(y_test, y_test_pred_rbf)
train_rmse_rbf = np.sqrt(mean_squared_error(y_train, y_train_pred_rbf))
test_rmse_rbf = np.sqrt(mean_squared_error(y_test, y_test_pred_rbf))

cv_r2_rbf = cross_val_score(svr_rbf, X, y, cv=5, scoring="r2")
cv_rmse_rbf = -cross_val_score(svr_rbf, X, y, cv=5, scoring="neg_root_mean_squared_error")

r_value, p_value = pearsonr(y_test, y_test_pred_rbf)

print("=== SVR with RBF kernel ===")
print("Mean CV R²:", np.mean(cv_r2_rbf))
print("Mean CV RMSE:", np.mean(cv_rmse_rbf))
print("Pearson r:", r_value)
print("p-value:", p_value)


# ===============================
# 4. SVR with Linear kernel
svr_linear = SVR(kernel="linear")
svr_linear.fit(X_train, y_train)

y_train_pred_linear = svr_linear.predict(X_train)
y_test_pred_linear = svr_linear.predict(X_test)

train_r2_linear = r2_score(y_train, y_train_pred_linear)
test_r2_linear = r2_score(y_test, y_test_pred_linear)
train_rmse_linear = np.sqrt(mean_squared_error(y_train, y_train_pred_linear))
test_rmse_linear = np.sqrt(mean_squared_error(y_test, y_test_pred_linear))

cv_r2_linear = cross_val_score(svr_linear, X, y, cv=5, scoring="r2")
cv_rmse_linear = -cross_val_score(svr_linear, X, y, cv=5, scoring="neg_root_mean_squared_error")

r_value, p_value =  pearsonr(y_test, y_test_pred_linear)

print("=== SVR with Linear kernel ===")
print("Mean CV R²:", np.mean(cv_r2_linear))
print("Mean CV RMSE:", np.mean(cv_rmse_linear))
print("Pearson r:", r_value)
print("p-value:", p_value)
