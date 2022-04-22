# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:27:52 2022

Normalization of transcriptome data

@author: I.Azuma, K.Morita
"""
import copy

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from . import utils

class Normalization():
    
    def __init__(self,methods:list=[]):
        self.method_list = ["raw","quantile","log2","col min-max","row min-max","global min-max","col z-score","row z-score"]
        self.method_dict =  {
                'raw':self.__raw,
                'quantile':self.__quantile,
                'log2':self.__log2,
                'col min-max':self.__col_minmax,
                'row min-max':self.__row_minmax,
                'global min-max':self.__global_minmax,
                'col z-score':self.__col_z,
                'row z-score':self.__row_z,
        }
        self.data = pd.DataFrame()
        self.original_col=list()
        self.original_row=list() 
        self.res = pd.DataFrame()
        self.methods=methods
        self.__processing = utils
    
    def set_data(self,df):
        """set target data (pandas dataframe)"""
        self.data=df
        self.original_col = df.columns.tolist()
        self.original_row = df.index.tolist()
            
    def perform_normalization(self):
        """perform normalization in the order registered in the methods"""
        if len(self.methods)==0:
            raise ValueError("!! Set method first !!")
        df_tmp = copy.deepcopy(self.data)
        for i,m in enumerate(self.methods):
            #print(i+1,":",m)
            df_tmp = self._do_norm(df_tmp,method=m)
        self.res=df_tmp

    def _do_norm(self,df,method:str=""):
        """conduct normalization"""
        if self.data is None:
            raise ValueError("!! Set data first !!")
        df_tmp = copy.deepcopy(df)
        converter = self.method_dict.get(method, None)
        if converter is not None:
            df_tmp = converter(df_tmp)
        else:
            raise KeyError("!! Set appropriate method : {}!!".format(self.method_list))
        # rename index, columns
        df_tmp = pd.DataFrame(df_tmp)
        df_tmp.index = self.original_row
        df_tmp.columns = self.original_col
        return df_tmp
            
    def __raw(self, df):
        return df

    def __quantile(self, df):
        return self.__processing.quantile(df)
    
    def __log2(self, df):
        return np.log2(df+1)

    def __col_minmax(self, df):
        """index (gene) wide min-max scaling"""
        scaler=MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(df))

    def __row_minmax(self, df):
        """columns (sample) wide min-max scaling"""
        scaler=MinMaxScaler()
        return pd.DataFrame(scaler.fit_transform(df.T).T)

    def __global_minmax(self, df):
        return self.__processing.global_minmax(df)

    def __col_z(self, df):
        """index (gene) wide z-score"""
        return self.__processing.standardz_sample(df)

    def __row_z(self, df):
        """columns (sample) wide z-score"""
        return self.__processing.standardz_sample(df.T).T

     