# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 20:53:01 2021

@author: I.Azuma, K.Morita
"""
import copy
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as st
import matplotlib.pyplot as plt
import statsmodels.stats.multitest as sm

from .deg_analyzer import ttest
from .deg_analyzer import multi_FC

from ._utils import utils

class Deg():
    ### init ###
    def __init__(self):
        self.df_mix=pd.DataFrame()
        self.df_ref=pd.DataFrame()
        self.final_ref=pd.DataFrame()
        self.__deg_class=None
        self.__method_dict={"ttest":ttest.Deg_ttest,"multiFC":multi_FC.Deg_Multi_FC}
    
    ### main ###
    def set_method(self,method="ttest"):
        """set deg definition method"""
        self.__deg_class = self.__method_dict.get(method, None)
        if self.__deg_class is None:
            raise KeyError("!! Set appropriate method : {}!!".format(self.__method_dict))
        print("method :",method)
        
    def set_data(self,df_mix,df_ref):
        """set data"""
        self.df_mix = df_mix
        self.df_ref = df_ref
            
    def create_ref(self,sep="_",number=200,limit_CV=1,limit_FC=1.5,q_limit=0.05,log2=False,plot=False):
        """
        create reference dataframe which contains signatures for each cell

        """
        dat=self.__deg_class()
        dat.set_data(self.df_mix,self.df_ref)
        dat.narrow_intersection()
        dat.create_ref(sep=sep,number=number,limit_CV=limit_CV,limit_FC=limit_FC,q_limit=q_limit,log2=log2,plot=plot)
        self.final_ref = dat.final_ref
