# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 06:04:20 2020

@author: I.Azuma, K.Morita
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.svm import NuSVR
import scipy as sp

### fitting modules ###

"""
input : reference dataframe / mix dataframe / parameters

output : one result matrix

"""

# elasticnet
def fit_ElasticNet(ref,dat,alpha=1,l1_ratio=0.05,max_iter=100000,**kwargs):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=1e-5, random_state=None, fit_intercept=True)
    model.fit(ref,dat)
    print(model.score(ref,dat))
    res_mat = pd.DataFrame(model.coef_,index=dat.columns, columns=ref.columns)
    return res_mat


# NuSVR
def fit_NuSVR(ref,dat,nu=[0.25,0.5,0.75],max_iter=100000,**kwargs):
    tune_parameters = [{'kernel': ['linear'],
                        'nu':nu,
                        'gamma': ['auto'],
                        'C': [1]}]
    res_mat=[]
    ap = res_mat.append
    for i in range(len(dat.columns)):
        y = list(dat.iloc[:,i])
        gscv = GridSearchCV(NuSVR(max_iter=max_iter), tune_parameters, scoring="neg_mean_squared_error")
        gscv.fit(ref,y)
        model = gscv.best_estimator_
        ap(model.coef_[0])
    res_mat = pd.DataFrame(res_mat,index=dat.columns, columns=ref.columns)
    return res_mat   


# NNLS
def fit_NNLS(ref,dat,**kwargs):
    A = np.array(ref)
    res_mat = pd.DataFrame(index=dat.columns, columns=ref.columns)
    for i in range(len(list(dat.columns))):
        b = np.array(dat.iloc[:,i])
        res_mat.iloc[i,:] = sp.optimize.nnls(A,b)[0]
    return res_mat