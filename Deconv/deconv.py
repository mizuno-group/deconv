# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:55:45 2020

@author: I.Azuma, K.Morita
"""

import pandas as pd
import numpy as np
import os
from combat.pycombat import pycombat
import seaborn as sns
import matplotlib.pyplot as plt
import copy

import processing
import deg
import fitter
from _utils import utils

class Deconvolution():
    ### initialize ###
    def __init__(self):
        self.__mix_data=pd.DataFrame()
        self.__reference_data=pd.DataFrame()
        self.__res=[]
        self.__processing = utils
        self.method_dict={'elasticnet':fitter.fit_ElasticNet,'NuSVR':fitter.fit_NuSVR,'NNLS':fitter.fit_NNLS}
    
    ### main ###
    def set_data(self,dat=pd.DataFrame(),ref=pd.DataFrame()):
        if len(dat)>0:
            self.__set_mix_data(dat)
        if len(ref)>0:
            self.__set_reference_data(ref)
        if len(self.__reference_data)==0:
            self.__set_reference_data()
        
    def preprocessing_mix(self, df_ref=None, places:list=[],          
                          trimming=True, threshold=0.9, strategy="median", trim=1.0, batch=False, split="_",
                          combat=False, batch_lst=[[],[]], plot=False,
                          trans_method:str="", norm_method_list:list=[],):
        """preprocessing for mix data"""
        dat = processing.Processing()
        dat.set_data(self.__mix_data)
        if len(places)>0:
            dat.annotation(df_ref, places=places)
        if trimming:
            dat.trimming(threshold=threshold, strategy=strategy, trim=trim, batch=batch, split=split)
        if combat:
            dat.combat(batch_lst=batch_lst,plot=plot)
        if len(trans_method)>0:
            dat.transform(method=trans_method)
        if len(norm_method_list)>0:
            dat.normalize(methods=norm_method_list)
        self.__mix_data = dat.res

    def preprocessing_ref(self, df_ref=None, places:list=[],                           
                          trimming=True, threshold=0.9, strategy="median", trim=1.0, batch=False, split="_",
                          combat=False, batch_lst_lst=[[],[]], plot=False,
                          trans_method:str="", norm_method_list:list=[],):
        """preprocessing for reference data"""
        dat = processing.Processing()
        dat.set_data(self.__reference_data)
        if len(places)>0:
            dat.annotation(df_ref, places=places)
        if trimming:
            dat.trimming(threshold=threshold, strategy=strategy, trim=trim, batch=batch, split=split)
        if combat:
            dat.combat(batch_lst_lst=batch_lst_lst,plot=plot)
        if len(trans_method)>0:
            dat.transform(method=trans_method)
        if len(norm_method_list)>0:
            dat.normalize(methods=norm_method_list)
        self.__reference_data = dat.res

    def deg(self, method:str="ttest",number=150,limit_CV=1,limit_FC=1.5,log2=False):
        dat = deg.Deg()
        dat.set_method(method=method)
        dat.set_data(df_mix="",df_all="")
        dat.pre_processing(do_ann=False,ann_df=None,do_log2=True,do_quantile=True,do_trimming=False,do_drop=True)
        dat.narrow_intersec()
        dat.create_ref(sep="_",number=150,limit_CV=1,limit_FC=1.5,log2=False)
        return
    
    def fit(self):
        return

    ### in/out put control ###
    def __set_mix_data(self,dat=pd.DataFrame()):
        """set mixture sample transcriptome data"""
        self.__mix_data = dat

    def __set_reference_data(self,ref=pd.DataFrame()):
        """ set reference data with input or DCQ reference """
        if len(ref)==0:
            dirname = os.path.dirname(os.path.abspath('__file__'))
            file_ref = dirname+'/reference_files/ref_dcq.csv'
            ref = pd.read_csv(file_ref, index_col=0)
            print('DCQ reference is used')
        self.__reference_data = ref

    def get_res(self):
        return self.__res
    
    def get_data(self):
        return self.__mix_data, self.__reference_data

    ###  ###
    def __combat_correction(self,nonpara=False):
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


