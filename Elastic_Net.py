import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics, model_selection, preprocessing
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.metrics import median_absolute_error
import pickle
import sys
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, Ridge
from scipy.stats import pearsonr
#A random_state of 42 was used to train the models published in the original publication.

"""
正则化保留了所有特征，线性回归器对模型的特征回归任务表现不佳，弹性网的α值很小，正则化强度很弱；模型L2（即Ridge）占比高，用弹性网/岭回归的线性回归器预测性能较差。
"""

# Load datasets
data = sys.argv[1]
df = pd.read_excel(data, header=0)

X_Eads = df.iloc[:,:9]
# Visualization of feature vector space in dataset 
X_Eads.describe()
# Hyperparameter optimization for the model predicting the surface activity (Eads)
# Target
y_Eads = df["Eads"]
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X_Eads, y_Eads, test_size=0.15, random_state=42) 
# Dataset splitting in train and test set

X_train_scaled = X_trainset.values
X_test_scaled = X_testset.values
# scaler = StandardScaler() 稍微有提升
# X_train_scaled = scaler.fit_transform(X_trainset.values)  
# X_test_scaled = scaler.transform(X_testset.values)   

param_grid = {
    'alpha': np.logspace(-3, 0, 10),  # 0.001 ~ 1
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}

enet_grid = GridSearchCV(ElasticNet(max_iter=10000), param_grid, cv=5, scoring='r2')
enet_grid.fit(X_train_scaled, y_trainset)

print("Best params:", enet_grid.best_params_)
print("Best CV R2:", enet_grid.best_score_)

best_alpha = enet_grid.best_params_['alpha']
best_l1_ratio = enet_grid.best_params_['l1_ratio']

enet_best = ElasticNet(alpha=best_alpha, l1_ratio=best_l1_ratio, max_iter=10000)
enet_best.fit(X_train_scaled, y_trainset)
y_pred = enet_best.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_testset, y_pred))

enet_r2_scores = cross_val_score(enet_best, X_Eads, y_Eads, cv=5, scoring='r2') 
enet_rmse_scores = -cross_val_score(enet_best, X_Eads, y_Eads, cv=5, scoring='neg_root_mean_squared_error') 
print("ENET CV RMSE:", np.mean(enet_rmse_scores))
print("ENET CV R²:", np.mean(enet_r2_scores))


plt.figure(figsize = (5,5))
plt.scatter(y_trainset, enet_best.predict(X_trainset))
plt.scatter(y_testset, enet_best.predict(X_testset)) 
plt.plot([-2, 2], [-2, 2], "k--")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()
 
feature_names = X_trainset.columns
print(f"Original number of features: {len(feature_names)}")
print("non-zero coefficients:", np.sum(enet_best.coef_.ravel() != 0))


import numpy as np
array_alpha = np.logspace(-2, 1) #不用线性linear的α值，是希望range尺度大点变化明显
array_alpha # 0.01到10的等比数列
rmse = [] #rmse表现偏差，R2表现关系
for alpha in array_alpha:
    ridge = Ridge(alpha = alpha)
    rmse_scores = -cross_val_score(ridge, X_Eads, y_Eads, scoring ="neg_root_mean_squared_error") #交叉验证
    #mean_rmse = np.mean(rmse_scores)
    rmse.append(np.mean(rmse_scores))
    #print(f"alpha = {alpha:.3f}, mean RMSE = {mean_rmse:.4f}")
plt.plot(array_alpha,rmse)
#plt.xscale("log") #x轴改成对数，让α显示更明显，0.01到0.1时变化缓慢，后面变化增大
plt.show()
ridge_best = Ridge(alpha=1.6)
ridge_best.fit(X_trainset, y_trainset)
y_pred = ridge_best.predict(X_testset)

ridge_r2_scores = cross_val_score(ridge_best, X_Eads, y_Eads, cv=5, scoring='r2') 
ridge_rmse_scores = -cross_val_score(ridge_best, X_Eads, y_Eads, cv=5, scoring='neg_root_mean_squared_error') 
print("Ridge CV RMSE:", np.mean(ridge_rmse_scores))
print("Ridge CV R²:", np.mean(ridge_r2_scores))

r_value, p_value = pearsonr(y_testset, y_pred)
print("Pearson r:", r_value)
print("p-value:", p_value)

