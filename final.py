# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 22:39:34 2019

@author: hasim
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from scipy.stats import randint
import random
import time

df = pd.read_csv("labeledpsd.csv")
#%%
df.drop(["Unnamed: 0"],axis=1,inplace=True)
df.drop(["ALL_MASK"],axis=1,inplace=True)
df.drop(["type"],axis=1,inplace=True)
df.drop(["pole"],axis=1,inplace=True)
#df.drop(["sira"],axis=1,inplace=True)
#%% change index to matlab index
df=df.set_index('sira')

#%%
sns.countplot(x="lf",data=df);
plt.show()
df.lf.value_counts()
#%%
df.insert(loc=0, column='label', value=df.lf)
#%%
dfo=df.copy()
group_object = df.groupby('label')
df = group_object.apply(lambda x: x.sample(group_object.size().min()))
#%%
sns.countplot(x="lf",data=df);
plt.show()
df.lf.value_counts()
#%%
#df.drop(["sira"],axis=1,inplace=True)
df.drop(["lf"],axis=1,inplace=True)
df.drop(["td"],axis=1,inplace=True)
df.drop(["ub"],axis=1,inplace=True)
df.drop(["rt"],axis=1,inplace=True)
df.drop(["br"],axis=1,inplace=True)
df.drop(["st"],axis=1,inplace=True)
df.drop(["oth"],axis=1,inplace=True)
df.drop(["faulty"],axis=1,inplace=True)
#%% run to downsample
#dfd=df.iloc[:, 1::10]
#df = pd.concat([df.label, dfd], axis=1)

#%%assign Class_att column as y attribute
target_names = ['False', 'True']
df=df.dropna()
y = df.label.to_frame()
y=y.astype('int')

#drop Class_att column, remain only numerical columns
new_data = df.drop(["label"],axis=1)

#Normalize values to fit between 0 and 1. 
X = new_data.copy()
#x = (new_data-np.min(new_data))/(np.max(new_data)-np.min(new_data)).values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state =0)

#%% Grid Search

rfc_grid=RandomForestClassifier(random_state=42)

param_grid = { 
    'max_depth': [None],
    'max_features': [100, 150, 250],
    'min_samples_leaf': [1, 2, 3, 5],
    'min_samples_split': [5, 10],
    'n_estimators': [100, 150, 200]
    #'criterion': ['gini', 'entropy']
}
#cvkfold = model_selection.StratifiedKFold(n_splits=5, shuffle=True)
CV_rfc = GridSearchCV(estimator=rfc_grid, 
                      param_grid=param_grid, 
                      cv = 5, 
                      n_jobs=-1, 
                      verbose=20)
#%%
CV_rfc.fit(X, y)
#%%
CV_rfc.best_params_
#%%
CV_rfc.best_score_
#%%Randomized Search

rfc_rand=RandomForestClassifier(random_state=42)

# param_grid = {"max_depth": [None],
#               "max_features": randint.rvs(100, 500, size=10).tolist(),
#               "min_samples_leaf": randint.rvs(1, 10, size=5).tolist(),
#               "min_samples_split": randint.rvs(2, 10, size=5).tolist(),
#               "n_estimators": randint.rvs(50, 250, size=5).tolist(),
#               "criterion": ["gini", "entropy"]
#               }

param_grid = {"max_depth": [5, 10],
              "max_features": random.sample(range(100, 500), 5),
              "min_samples_leaf": random.sample(range(1, 10), 4),
              "min_samples_split": random.sample(range(2, 10), 4),
              "n_estimators": random.sample(range(50, 250), 4),
              # "criterion": ["gini", "entropy"]
              }

RSCV_rfc = RandomizedSearchCV(param_distributions=param_grid, 
                              estimator = rfc_rand, 
                              scoring = "accuracy", 
                              cv = 5, 
                              n_jobs=-1, 
                              n_iter=100, 
                              verbose=20)
#%%
RSCV_rfc.fit(X, y)
#%%
RSCV_rfc.best_params_
#%%
RSCV_rfc.best_score_
#%% AdaBoost Classifier

ada = AdaBoostClassifier(random_state = 42)
params={'learning_rate': np.random.rand(1,5)[0,:].tolist(),# generate floats btw (0,1) returns list
        'n_estimators': random.sample(range(100, 500), 5),
            }

# scorer = make_scorer(f1_score)
ada_obj = RandomizedSearchCV(estimator=ada, 
                             param_distributions=params, 
                             scoring = "accuracy", 
                             cv = 5,
                             n_iter = 12,
                             n_jobs = -1,
                             verbose=20)
#%%
ada_obj.fit(X, y)
#%%
ada_obj.best_params_
#%%
ada_obj.best_score_
#%%
ada_clf = AdaBoostClassifier(learning_rate = 0.105,
                             n_estimators = 362,
                             random_state=42)
start_time=time.time()
ada_result = ada_clf.fit(X_train, y_train)
accuracy_ada = ada_clf.score(X_test, y_test)
print("AdaBoost accuracy is :", accuracy_ada)
print("\n--- %s seconds ---" % (time.time() - start_time))
#%% plot importances
ada_importances = ada_clf.feature_importances_
(pd.Series(ada_importances, index=X.columns)
   .head(5000)
   .plot(kind='bar'))
#%% AdaBoost CV

import warnings
warnings.filterwarnings("ignore")
from sklearn import model_selection

ada_cv = AdaBoostClassifier(learning_rate = 0.105,
                             n_estimators = 362,
                             random_state=42)
start_time=time.time()
kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=False)
results = model_selection.cross_val_score(ada_cv, X, y, cv=kfold)
#print("Accuracy: Final mean:%.3f%%, Final standard deviation:(%.3f%%)" % (results.mean()*100.0, results.std()*100.0))
print('Accuracies from each of the 5 folds using kfold:',results)
print("\n--- %s seconds ---" % (time.time() - start_time))

# %%Random Forest Classification

rf_model_initial = RandomForestClassifier(max_depth = None, max_features = 184, min_samples_leaf = 3, min_samples_split = 6, n_estimators = 101, random_state = 42)
start_time=time.time()
rf_model_initial.fit(X_train,y_train)
accuracy_rf = rf_model_initial.score(X_test,y_test)
print("\nRandom Forest accuracy is :",rf_model_initial.score(X_test,y_test))
print("\n--- %s seconds ---" % (time.time() - start_time))
y_pred = rf_model_initial.predict(X_test)

#%% Confusion Matrix for Random Forest Classification

cm_rf=confusion_matrix(y_test,rf_model_initial.predict(X_test))
f, ax = plt.subplots(figsize = (8,8))
sns.heatmap(cm_rf, annot = True, linewidths = 0.5, color = "red", fmt = ".0f", ax=ax)
plt.xlabel("y_predicted")
plt.ylabel("y_true")
plt.title("Loose Foundation - CM of RF Class. - N=150")
plt.show()
print(classification_report(y_test, y_pred, target_names=target_names))
#%% FALSE PREDICTED PSDs
y_pred = pd.DataFrame(data=y_pred)
y_test.equals(y_pred)
y_pred.columns=["prediction"]#rename column
y_test["sira"]=y_test.index
y_test=y_test.reset_index(drop=True)
dfc = pd.concat([y_pred, y_test], axis=1)
#%%compare
dfc['compare'] = np.where(dfc.prediction == dfc.label, 'True', 'False')  
#create new column in dfc to check if prices match
#%%get rows where the values do not match
dfr=dfc[(dfc.compare=='False')]
#%% FEATURE IMPORTANCES
###################################################################
importances = rf_model_initial.feature_importances_
(pd.Series(importances, index=X.columns)
   .head(5000)
   .plot(kind='bar'))
###################################################################

#%% Random Forest Classification CV
import warnings
warnings.filterwarnings("ignore")
from sklearn import model_selection

rf_model_cv = RandomForestClassifier(max_depth = None, max_features = 184, min_samples_leaf = 3, min_samples_split = 6, n_estimators = 101, random_state = 42)
start_time=time.time()
kfold = model_selection.StratifiedKFold(n_splits=5, shuffle=False)
results = model_selection.cross_val_score(rf_model_cv, X, y, cv=kfold)
print('Accuracies from each of the 5 folds using kfold:',results)
print("\n--- %s seconds ---" % (time.time() - start_time))

#%% Print out falsely predicted PSDs (in matlab index)
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax
#%%
render_mpl_table(dfr, header_columns=0, col_width=2.0)
#%%