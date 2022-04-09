# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 14:36:50 2022

Transformation of transcriptome data

the initial input data : TPM
1. linear
2. log2
3. sqrt

@author: I.Azuma, K.Morita
"""
import copy

import numpy as np
import pandas as pd

class Transformation():

    def __init__(self,method=None):
        self.data = pd.DataFrame()
        self.res = pd.DataFrame()
        self.method_list = ["linear","log2","sqrt","double log2"]
        self.method_dict={
                'linear':self.__linear,
                'log2':self.__log2,
                'sqrt':self.__sqrt,
                'double log2':self.__double_log,
                }
        self.method=method
    
    def set_data(self,df):
        """set target data"""
        self.data = df
    
    def transform(self):
        """
        transform the target data with the selected method
        """
        if self.data is None:
            raise ValueError("!! Set data first !!")
        converter = self.method_dict.get(self.method, None)
        df = copy.deepcopy(self.data)
        if converter is not None:
            self.res = converter(df)
        else:
            raise KeyError("!! Set appropriate method : {}!!".format(self.method_list))
        
    def __linear(self, df):
        return df

    def __log2(self, df):
        return np.log2(df+1)

    def __sqrt(self, df):
        return np.sqrt(df)
    
    def __double_log(self, df):
        return np.log2(np.log2(df+1)+1)
