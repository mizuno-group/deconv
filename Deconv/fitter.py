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
from combat.pycombat import pycombat

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

    def fit(self,
            number_of_repeats=1,
            alpha=1,l1_ratio=0.05,
            nu=[0.25,0.5,0.75],
            max_iter=100000,
            combat=False,nonpara=False):
        # pre-processing
        df_mix=copy.deepcopy(self.df_mix)
        df_ref=copy.deepcopy(self.df_ref)
        # celc median of same gene name
        df_mix=df_mix.groupby(level=0,axis=0).median()
        df_ref=df_ref.groupby(level=0,axis=0).median()
        # dropna
        df_mix=utils.drop_all_missing(df_mix)
        df_ref=utils.drop_all_missing(df_ref)
        # index intersection
        df_mix, df_ref = self.__intersection_index(df_mix, df_ref)
        # update data
        self.set_data(df_mix, df_ref)
        # fitting
        self.__fit()
        if combat:
            self.__combat_correction(nonpara=nonpara)
            self.__fit(number_of_repeats=number_of_repeats,alpha=alpha,l1_ratio=l1_ratio,nu=nu,max_iter=max_iter)       

    ### fitting methods ###
    def __fit(self,
              number_of_repeats=1,
              alpha=1,l1_ratio=0.05,
              nu=[0.25,0.5,0.75],
              max_iter=100000):
        try:
            for i in range(number_of_repeats):
                res_mat = self.method(self.__reference_data,self.__mix_data,alpha=alpha,l1_ratio=l1_ratio,nu=nu,max_iter=max_iter)
                # sum up
                if i == 0:
                    res = res_mat
                else:
                    res = res + res_mat
            res = res / number_of_repeats
        except:
            raise NotImplementedError
        self.__res=res       

    def combat_correction(self,nonpara=False):
        """combat correction between deconvolution result and sample transcriptome"""
        batch_list = [0]*len(self.__res.index) + [1]*len(self.__res.index)
        if len(batch_list)<20:
            print('*** sample size is small ***')
            print('combat correction is not recommended')
        mix_data_estimated = np.dot(np.array(self.__res),np.array(self.__reference_data.T))
        mix_data_estimated = pd.DataFrame(mix_data_estimated.T,index=list(self.__mix_data.index))
        mix_data_sum = pd.concat([self.__mix_data,mix_data_estimated],axis=1)
        mix_data_corrected = pycombat(mix_data_sum,batch_list,par_prior=not nonpara)
        mix_data_corrected = mix_data_corrected.iloc[:,:len(self.__res.index)]
        self.__mix_data = mix_data_corrected
        return        

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

    ### processing ###
    def __intersection_index(self,df,df2):
        ind1 = df.index.tolist()
        ind2 = df2.index.tolist()
        df.index = [i.upper() for i in ind1]
        df2.index = [i.upper() for i in ind2]
        ind = list(set(df.index) & set(df2.index))
        df = df.loc[ind,:]
        df2 = df2.loc[ind,:]
        return df,df2