# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:00:52 2021

Modules for deg analyzer

@author: K.Morita, I.Azuma
"""

import copy

import pandas as pd
import numpy as np

def _intersection_index(df,df2):
    ind1 = df.index.tolist()
    ind2 = df2.index.tolist()
    df.index = [str(i).upper() for i in ind1]
    df2.index = [str(i).upper() for i in ind2]
    ind = list(set(df.index) & set(df2.index))
    df = df.loc[ind,:]
    df2 = df2.loc[ind,:]
    return df,df2

def _sepmaker(df=None,delimiter='.'):
    samples = list(df.columns)
    sample_unique = []
    seps=[]
    ap1 = sample_unique.append
    ap2 = seps.append
    for i in samples:
        if i.split(delimiter)[0] in sample_unique:
            number = sample_unique.index(i.split(delimiter)[0])
            ap2(number)
        else:
            ap1(i.split(delimiter)[0])
            ap2(len(sample_unique)-1)
    return seps, sample_unique

def _df_median(df,sep="_"):
    df_c = copy.deepcopy(df)
    df_c.columns=[i.split(sep)[0] for i in list(df_c.columns)]
    df_c = df_c.groupby(level=0,axis=1).median()
    return df_c

def _logFC(df_target,df_else):
    # calculate df_target / df_else logFC
    df_logFC = df_target.T.median() - df_else.T.median()
    df_logFC = pd.DataFrame(df_logFC)
    df_logFC.index = df_target.index
    df_logFC = df_logFC.replace(np.inf,np.nan)
    df_logFC = df_logFC.replace(-np.inf,np.nan)
    df_logFC = df_logFC.fillna(0)        
    return df_logFC

def _calc_CV(df_target):
    """
    CV : coefficient of variation
    """
    df_CV = np.std(df_target,axis=1) / np.mean(df_target,axis=1)
    df_CV = pd.DataFrame(df_CV)
    df_CV.index = df_target.index
    df_CV = df_CV.replace(np.inf,np.nan)
    df_CV = df_CV.replace(-np.inf,np.nan)
    df_CV = df_CV.fillna(0)
    return df_CV

def _get_res(pickup_genes):
    res=[i for i in pickup_genes if str(i)!='nan']
    res=list(set(res))
    return res