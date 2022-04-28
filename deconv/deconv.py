# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 13:55:45 2020

@author: I.Azuma, K.Morita
"""

import copy
import os

import pandas as pd
import numpy as np
from combat.pycombat import pycombat

from . import processing
from . import deg
from . import fitter
from ._utils import plotter

class Deconvolution():
    ### initialize ###
    def __init__(self):
        self.__mix_data=pd.DataFrame()
        self.__reference_data=pd.DataFrame()
        self.__res=pd.DataFrame()
        self._pickup_genes_df=pd.DataFrame()
    
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

    def deg(self,method:str="ttest",intersection=False,
            sep:str="_",number=150,limit_CV=1,limit_FC=1.5,q_limit=0.05,
            log2=False,plot=False,prints=True):
        dat = deg.Deg()
        dat.set_method(method=method)
        dat.set_data(self.__mix_data,self.__reference_data)
        dat.create_ref(sep=sep,number=number,limit_CV=limit_CV,limit_FC=limit_FC,q_limit=q_limit,log2=log2,plot=plot,intersection=intersection,prints=prints)
        self.__reference_data = dat.final_ref
        self._pickup_genes_df = dat._pickup_genes_df
    
    def fit(self,method:str="",
            number_of_repeats=1,
            alpha=1,l1_ratio=0.05,
            nu=[0.25,0.5,0.75],
            max_iter=100000,
            combat=False,nonpara=False):
        dat=fitter.Fitter()
        dat.set_method(method=method)
        dat.set_data(self.__mix_data,self.__reference_data)
        dat.fit(number_of_repeats=number_of_repeats,alpha=alpha,l1_ratio=l1_ratio,nu=nu,max_iter=max_iter,combat=combat,nonpara=nonpara)
        self.__res=dat.res

    def plot_res(self,sort_index:list=[], control_names:list=["control, ctrl"], row_n:int=2,col_n:int=3,sep="_"):
        plotter.plot_immune_box(copy.deepcopy(self.__res),sort_index=sort_index,control_names=control_names,row_n=row_n,col_n=col_n,sep=sep)

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
