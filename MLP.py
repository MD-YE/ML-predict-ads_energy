#####################################################################################################

#!/usr/bin/env python
# coding: utf-8

# Libraries
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
from sklearn.neural_network import MLPRegressor 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse, r2_score
#A random_state of 42 was used to train the models published in the original publication.

"""
GridSearchCV : solver & hidden_layer_sizes & alpha (Regularization)...
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
# Dataset splitting in train and test set
X_train, X_test, y_train, y_test = train_test_split(X_Eads, y_Eads, test_size=0.15, random_state=42)

nn = MLPRegressor(hidden_layer_sizes = (64, 64), max_iter=5000, random_state=42)
nn.fit(X_train, y_train)
y_test_pred = nn.predict(X_test)
#rmse_train = np.sqrt(mse(y_train, nn.predict(X_train)))
#rmse_test  = np.sqrt(mse(y_test, nn.predict(X_test)))

#r2_train = r2_score(y_train, nn.predict(X_train))
#r2_test  = r2_score(y_test, nn.predict(X_test))

#print("Train RMSE:", rmse_train)
#print("Test RMSE:", rmse_test)
#print("Train R²:", r2_train)
#print("Test R²:", r2_test)

cv_r2_scores = cross_val_score(nn, X_Eads, y_Eads, cv=5, scoring='r2')  
rmse_scores = -cross_val_score(nn,  X_Eads, y_Eads, scoring = "neg_root_mean_squared_error", cv=5)
#print("Cross-val R² scores:", cv_r2_scores)
print("Mean CV R²:", np.mean(cv_r2_scores))
print("Mean CV rmse:", np.mean(rmse_scores))

plt.figure(figsize = (5,5))
plt.scatter(y_train, nn.predict(X_train), c = "cyan", edgecolor = "k")
plt.scatter(y_test, nn.predict(X_test), c = "pink", edgecolor = "k")
            
plt.plot([-2.0, 2.0], [-2.0, 2.0], "k--")
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()
r_value, p_value = pearsonr(y_test, y_test_pred)
print("Pearson r:", r_value)
print("p-value:", p_value)
