# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 15:24:08 2021

@author: mengt
"""
import os
os.getcwd()
os.chdir("/Users/javi.fong/Documents/MS DataScience/2 Cuatrimestre/Advance Programming/Advance Programming - Assignment 2")

import sklearn as sk
import numpy as np
from numpy.random import randint
import pandas as pd


### Make the results are reproducible using np.random.seed:

my_NIA = 100466870
np.random.seed(my_NIA)

## Read the solar dataset (in pickle format) into a Pandas dataframe 
train = pd.read_pickle('traintestdata_pickle/trainst1ns16.pkl')
test = pd.read_pickle('traintestdata_pickle/testst1ns16.pkl')

### Choose the clotest point variables

cp_train= train.iloc[:,:75]
cp_train_Y= train.energy

cp_test = test.iloc[:,:75]
cp_test_Y= test.energy

### The set of all data for final model
data_x = pd.concat([cp_train, cp_test])
data_y = pd.concat([cp_train_Y, cp_test_Y])



### 5
## a)
### Split the train set into Train_train and train_validation (First 10 years and the last 2 years)

train_train = cp_train.iloc[:3650,]
train_train_Y= cp_train_Y[:3650,]
train_validation = cp_train.iloc[3650:,]
train_validation_Y = cp_train_Y[3650:,]

### b)

### KNN, SVM and Regression Tree with default hyper-parameter

## Knn and SVM requiere scaling


### KNN with default parameter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import neighbors

scaler = StandardScaler()
knn = neighbors.KNeighborsRegressor()
classif_knn = Pipeline([
    ('standarization', scaler),
    ('knn', knn)
    ])
classif_knn.fit(train_train, train_train_Y)
y_hat_knn=classif_knn.predict(train_validation)
print("KNN with Default HP:" ,sk.metrics.mean_absolute_error(train_validation_Y, y_hat_knn)) 

## MAE=2588455.571506849


### SVM with default parameter

svm = sk.svm.SVC()
classif_svm = Pipeline([
    ('standarization', scaler),
    ('svm', svm)
    ])
classif_svm.fit(train_train, train_train_Y)
y_hat_svm=classif_svm.predict(train_validation)
print("SVM with Default HP:", sk.metrics.mean_absolute_error(train_validation_Y, y_hat_svm))


## MAE= 6920271.25479452



#### Regression Tree with default parameter
from sklearn import tree
classif_rt= tree.DecisionTreeRegressor()
classif_rt.fit(train_train, train_train_Y)
y_hat_rt=classif_rt.predict(train_validation)
print("Decission Tree with Default HP:",sk.metrics.mean_absolute_error(train_validation_Y, y_hat_rt))


## MAE= 3078653.835616438



### c) and d) KNN, SVM and Regression Tree with Hyper-parameter tunning

## hyper-parameter tuning with RandomizedSearch for Regression Trees and SVMs. Use GridSearch for KNN to optimize just the number of neighbors.


from sklearn.model_selection import PredefinedSplit


## Defining a fixed train/ validation grid-search
### -1 mean training and 0 mean validation

validation_indices = np.zeros(cp_train.shape[0])
validation_indices[:3650] = -1
tr_val_partition = PredefinedSplit(validation_indices)




from sklearn.model_selection import GridSearchCV, KFold
from sklearn import metrics

#Scaling Data 
scaler = StandardScaler()
scaler.fit(train_train)
scaled_all_train_x = scaler.transform(cp_train)
scaled_train_x = scaler.transform(train_train)
scaled_test_x = scaler.transform(cp_test)
scaled_validation_x = scaler.transform(train_validation)

## KNN with hyper parameter tunning: number of neighbors
## Search Space
param_grid_knn= {'n_neighbors': list(range(1,16,1))}
knn_grid= GridSearchCV(
    knn
    , param_grid_knn
    , scoring='neg_mean_absolute_error'
    , cv= tr_val_partition
    , verbose = 1
    )
knn_grid.fit(scaled_all_train_x, cp_train_Y) ## use scale data
print(f'Best hyper-parameters: {knn_grid.best_params_} and inner evaluation: {knn_grid.best_score_}')


knn_grid.best_score_
knn_grid.best_params_

y_test_pred_knn= knn_grid.predict(scaled_validation_x)
print(metrics.mean_absolute_error(train_validation_Y, y_test_pred_knn))

import pandas as pd
# Get all hyperparameters and their evaluation as a dataframe
cv_knn_results_df = pd.DataFrame(knn_grid.cv_results_).loc[:,['params', 'mean_test_score']]
# Sort by score, the best hyperparameters at the top
cv_knn_results_df = cv_knn_results_df.sort_values(by=['mean_test_score'], ascending = False)
cv_knn_results_df.iloc[:10,]



### SVM with hyperparameter tunning: 

from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.fixes import loguniform

## Search space
param_grid_svm = {'C': loguniform(1e0, 1e3),
 'gamma': loguniform(1e-4, 1e-3),
 'kernel': ['rbf','linear'],
 'class_weight':['balanced', None]}

budget = 20
svm_grid = RandomizedSearchCV(
    svm
    , param_grid_svm
    , scoring = 'neg_mean_absolute_error'
    , cv = tr_val_partition
    , n_iter = budget
    , n_jobs = 1
    , verbose = 1 
    )
svm_grid.fit(scaled_all_train_x, cp_train_Y) ### Use scale data
print(f'Best hyper-parameters: {svm_grid.best_params_} and inner evaluation: {svm_grid.best_score_}')


svm_grid.best_score_
svm_grid.best_params_

y_test_pred_svm= svm_grid.predict(scaled_validation_x)
print(metrics.mean_absolute_error(train_validation_Y, y_test_pred_svm))


##### Regression Tree with hyper-parameter tuning:

param_grid_rt = {
    "min_samples_split": list(range(2,16,2))
    , "max_depth": list(range(2,16,2))
    , "min_samples_leaf": list(range(2,16,2))
    , "max_leaf_nodes": list(range(2,16,2))
    }

budget= 20
rt_grid = RandomizedSearchCV(
    classif_rt
    , param_grid_rt
    , scoring = 'neg_mean_absolute_error'
    , cv=tr_val_partition
    , n_iter=budget
    , n_jobs=1
    , verbose =1 )
rt_grid.fit(cp_train, cp_train_Y)
print(f'Best hyper-parameters: {rt_grid.best_params_} and inner evaluation: {rt_grid.best_score_}')

rt_grid.best_score_
rt_grid.best_params_
y_test_pred_rt= rt_grid.predict(train_validation)
print(metrics.mean_absolute_error(train_validation_Y, y_test_pred_rt))
## e) 
## The best approach out of the six methods is the knn with hyper parameter tunning.


### 6) Model evaluation

## The model that was selected in model selection will be evaluated on the test set
prediction = knn_grid.predict(scaled_test_x)
print("MAE on Test Set:"
      , metrics.mean_absolute_error(prediction, cp_test_Y))

### 7) Final model
data_x = pd.concat([cp_train, cp_test])
data_y = pd.concat([cp_train_Y, cp_test_Y])
final_knn = neighbors.KNeighborsRegressor(n_neighbors=14)
data_x_scaled = scaler.transform(data_x)
final_knn.fit(data_x_scaled, data_y)
print("MAE on Full Data Set:",
      metrics.mean_absolute_error(data_y, final_knn.predict(data_x_scaled)))




