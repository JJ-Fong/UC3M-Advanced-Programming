#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 02:40:00 2021

@author: javi.fong
"""
import numpy as np
import pandas as pd
import math 
import os
from sklearn import neighbors, metrics
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA

my_nia = 100466870
np.random.seed(my_nia)


os.chdir("/Users/javi.fong/Documents/MS DataScience/2 Cuatrimestre/Advance Programming/Advance Programming - Assignment 2")
train = pd.read_pickle("traintestdata_pickle/trainst1ns16.pkl")
test = pd.read_pickle("traintestdata_pickle/testst1ns16.pkl")

#Choose closest point variables
train_x = train.iloc[:,:300]
train_y = train.energy

test_x = test.iloc[:,:300]
test_y = test.energy

random_columns = np.random.choice(train_x.columns, math.ceil(len(train_x.columns) * 0.1))

#Insert NaN
for col_name in random_columns:
    random_rows = np.random.choice(train_x[col_name].index, math.ceil(len(train_x[col_name].index) * 0.1)) 
    for row_name in random_rows: 
        train_x.at[row_name,col_name] = np.nan
    test_random_rows = np.random.choice(test_x[col_name].index, math.ceil(len(test_x[col_name].index) * 0.1)) 
    for t_row_name in test_random_rows: 
        test_x.at[t_row_name,col_name] = np.nan


#Split Train set into Train_train (first 10y and last 2y)
train_train_x = train_x[:3650]
train_train_y = train_y[:3650]

train_validation_x = train_x[3650:]
train_validation_y = train_y[3650:]
    
#Imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer()

#Scalers
from sklearn.preprocessing import MinMaxScaler, RobustScaler
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(train_train_x)

robust_scaler = RobustScaler()
robust_scaler.fit(train_train_x)

#Model 
knn = neighbors.KNeighborsRegressor()

#Validation Train 
validation_indices = np.zeros(train_x.shape[0])
validation_indices[:3650] = -1
validation_partition = PredefinedSplit(validation_indices)
 

param_grid = {
    'Model__n_neighbors': list(range(1,20,1))
    , 'Impute__strategy': ['mean', 'median']
    }

minmax_pipeline = Pipeline([
    ('Impute', imputer),
    ('Scale', minmax_scaler),
    ('Model', knn)
    ])

min_max_search = GridSearchCV(
    minmax_pipeline 
    , param_grid
    , scoring='neg_mean_absolute_error'
    , cv = validation_partition
)

min_max_search.fit(train_x, train_y)

min_max_search.best_params_
min_max_search.best_score_

robust_pipeline = Pipeline([
    ('Impute', imputer),
    ('Scale', robust_scaler),
    ('Model', knn)
    ])

robust_search = GridSearchCV(
    robust_pipeline 
    , param_grid
    , scoring='neg_mean_absolute_error'
    , cv = validation_partition
)

robust_search.fit(train_x, train_y)

print("MixMax - Best Params:", min_max_search.best_params_, "Best Score", min_max_search.best_score_)
print("Robust - Best Params:", robust_search.best_params_, "Best Score", robust_search.best_score_)

#MixMax - Best Params: {'Impute__strategy': 'mean', 'Model__n_neighbors': 19} Best Score -2319305.878010094
#Robust - Best Params: {'Impute__strategy': 'median', 'Model__n_neighbors': 17} Best Score -3205493.7775987107

#Select K Best Feeatures
imputer =  SimpleImputer(strategy='mean')
scaler = MinMaxScaler()
knn = neighbors.KNeighborsRegressor(n_neighbors=19)

selector = SelectKBest()
pca = PCA() 


combined_feat = FeatureUnion([
    ("pca", pca),
    ("selector", selector)
    ])

model_pipe = Pipeline([
    ('imputer', imputer), 
    ('scaler', scaler), 
    ('features', combined_feat), 
    ('knn_reg', knn)
    ])
     
selector_grid = {
    'features__selector__k': list(range(5,300,10))
    , 'features__pca__n_components': list(range(2,5,1))
    }



model_search = GridSearchCV(
    model_pipe
    , selector_grid
    , scoring='neg_mean_absolute_error'
    , cv = validation_partition
    , verbose=1
)

model_search.fit(train_x, train_y) 

model_search.best_params_
model_search.best_score_


imputer =  SimpleImputer(strategy='mean')
scaler = MinMaxScaler()
selector = SelectKBest(k = 205)
pca = PCA(n_components=2) 
knn = neighbors.KNeighborsRegressor(n_neighbors=19)

combined_feat = FeatureUnion([
    ("pca", pca),
    ("selector", selector)
    ])

model_pipe = Pipeline([
    ('imputer', imputer), 
    ('scaler', scaler), 
    ('features', combined_feat), 
    ('knn_reg', knn)
    ])

model_pipe.fit(train_x, train_y)


#3
name, obj = model_pipe['features'].transformer_list[1]
train_x_new = train_x.iloc[:, obj.get_support(indices= True)]
kbest_vars = train_x_new.columns.tolist()

linear_corr = train_x.corrwith(train_y)
ordered_corr = linear_corr[linear_corr.abs().sort_values(ascending = False).index]
ordered_index = ordered_corr.index.tolist()

kbest_index = [] 
for vn in kbest_vars:
    kbest_index.append(ordered_index.index(vn))
    
import matplotlib.pyplot as plt 
plt.hist(
    kbest_index
    , color = 'blue'
    , bins = 30)
#4
y_hat = model_pipe.predict(test_x)
print(metrics.mean_absolute_error(test_y, y_hat))
