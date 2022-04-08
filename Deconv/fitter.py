# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 06:04:20 2020

@author: I.Azuma, K.Morita
"""

import copy

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.svm import NuSVR
import scipy as sp

from _utils import utils

class Fitter():

    def __init__(self):
        self.df_mix=pd.DataFrame()
        self.df_ref=pd.DataFrame()
        self.__res=pd.DataFrame()
        self.__method=None
        self.__method_dict={'elasticnet':self._fit_ElasticNet,'NuSVR':self._fit_NuSVR,'NNLS':self._fit_NNLS}

    def set_method(self, method=""):
        """set fitting method"""
        self.__method = self.__method_dict.get(method, None)
        if self.__method is None:
            raise KeyError("!! Set appropriate method : {}!!".format(self.__method_dict))
        print("method: ",method)

    def set_data(self,df_mix,df_ref):
        """set data"""
        self.df_mix = df_mix
        self.df_ref = df_ref

    def fit(self, **kwargs):
        # pre-processing
        df_mix=copy.deepcopy(self.df_mix)
        df_ref=copy.deepcopy(self.df_ref)
        df_mix=self.

    ### fitting methods ###
    # elasticnet
    def _fit_ElasticNet(ref,dat,alpha=1,l1_ratio=0.05,max_iter=100000,**kwargs):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, tol=1e-5, random_state=None, fit_intercept=True)
        model.fit(ref,dat)
        print(model.score(ref,dat))
        res_mat = pd.DataFrame(model.coef_,index=dat.columns, columns=ref.columns)
        return res_mat

    # NuSVR
    def _fit_NuSVR(ref,dat,nu=[0.25,0.5,0.75],max_iter=100000,**kwargs):
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
    def _fit_NNLS(ref,dat,**kwargs):
        A = np.array(ref)
        res_mat = pd.DataFrame(index=dat.columns, columns=ref.columns)
        for i in range(len(list(dat.columns))):
            b = np.array(dat.iloc[:,i])
            res_mat.iloc[i,:] = sp.optimize.nnls(A,b)[0]
        return res_mat