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

import fitter
from _utils import processing, plotter

class Deconvolution():
    ### initialize ###
    def __init__(self):
        self.__mix_data=[]
        self.__reference_data=[]
        self.__res=[]
        self.__processing = processing
        self.method_dict={'elasticnet':fitter.fit_ElasticNet,'NuSVR':fitter.fit_NuSVR,'NNLS':fitter.fit_NNLS}
    
    ### main ###
    def preprocessing():
        return

    def deg():
        return
    
    def fit():
        return


    def do_fit(self,file_dat='',file_ref='',method='elasticnet',prints=True,combat=False,nonpara=False,number_of_repeats=1,
               alpha=1,l1_ratio=0.05,nu=[0.25,0.5,0.75],max_iter=100000):
        
        # data input
        self.__set_mix_data(file_dat=file_dat)
        self.__set_reference_data(file_ref=file_ref)
        
        # data processing
        self.__mix_data=self.__calc_median_same_gene(self.__mix_data)
        self.__reference_data=self.__calc_median_same_gene(self.__reference_data)
        
        self.__mix_data = self.__processing.drop_all_missing(self.__mix_data)
        self.__reference_data = self.__processing.drop_all_missing(self.__reference_data)
        
        self.__gene_intersection(prints=prints)
        
        self.a, self.b = self.get_data()
        
        # fitting
        self.__fit(method=method,number_of_repeats=number_of_repeats,alpha=alpha,l1_ratio=l1_ratio,nu=nu,max_iter=max_iter,prints=prints)
        
        if combat:
            self.__combat_correction(nonpara=nonpara)
            self.__fit(method=method,number_of_repeats=number_of_repeats,alpha=alpha,l1_ratio=l1_ratio,nu=nu,max_iter=max_iter,prints=prints)
        else:
            pass
        return
    
    def set_data(self,mix_df,ref_df):
        mix_df.index = [t.upper() for t in mix_df.index]
        self.__mix_data = mix_df
        ref_df.index = [t.upper() for t in ref_df.index]
        self.__reference_data = ref_df
    
    def reflect_signature(self,signature:list):
        common = list(set(self.__mix_data.index)&set(self.__reference_data.index))
        print(len(common),"genes are common between mix and reference")
        sig_common = list(set(common)&set(signature))
        print(len(sig_common),"/",len(signature),"signatures are registered in the analysis target genes")
        
        final_target = self.__mix_data
        final_ref = self.__reference_data.loc[sig_common]
        final_ref = self.__df_median(final_ref,sep="_")
        #sns.clustermap(final_ref,col_cluster=False,z_score=0)
        sns.clustermap(final_ref,col_cluster=False)
        plt.show()
        
        self.__mix_data = final_target
        self.__reference_data = final_ref
    
    def do_simple_fit(self,method='elasticnet',prints=True,combat=False,nonpara=False,number_of_repeats=1,
               alpha=1,l1_ratio=0.05,nu=[0.25,0.5,0.75],max_iter=100000):
        """fitting using curated mix and reference data"""
        # fitting
        self.__gene_intersection(prints=prints)
        self.__fit(method=method,number_of_repeats=number_of_repeats,alpha=alpha,l1_ratio=l1_ratio,nu=nu,max_iter=max_iter,prints=prints)
    
    def summarize(self,row_n=2,col_n=3):
        """predict each immune cell population and plot them"""
        res = self.get_res()
        z_res = self.__processing.standardz_sample(res)
        plot4deconv.plot_immune_box(z_res,row_n=row_n,col_n=col_n)
        
    
    ### in/out put control ###
    def __set_mix_data(self,file_dat=''):
        try:
            self.__mix_data = pd.read_csv(file_dat,index_col=0)
        except:
            self.__mix_data = file_dat

    def __set_reference_data(self,file_ref=''):
        if len(file_ref)==0:
            dirname = os.path.dirname(os.path.abspath('__file__'))
            file_ref = dirname+'/reference_files/ref_dcq.csv'
            print('DCQ reference is used')
        try:
            self.__reference_data = pd.read_csv(file_ref,index_col=0)
        except:
            self.__reference_data = file_ref

    def get_res(self):
        return self.__res
    
    def get_data(self):
        return self.__mix_data, self.__reference_data


    ### processing method ### ここら辺はutilsに移行する
    def __calc_median_same_gene(self,df):
        df = df.dropna()
        df2 = pd.DataFrame()
        dup = df.index[df.index.duplicated(keep="first")]
        gene_list = pd.Series(dup).unique().tolist()
        if len(gene_list) != 0:
            for gene in gene_list:
                new = df.loc[gene].median()
                df2[gene] = new
            df = df.drop(gene_list)
            df = pd.concat([df,df2.T])
        return df
    
    def __gene_intersection(self,prints=True):
        # delete nan / inf
        ref = self.__reference_data.replace(np.inf,np.nan)
        ref = ref.replace(-np.inf,np.nan)
        ref = ref.dropna()
        
        dat = self.__mix_data.replace(np.inf,np.nan)
        dat = dat.replace(-np.inf,np.nan)
        dat = dat.dropna()
        
        # upper gene name
        ref.index = [i.upper() for i in list(ref.index)]
        dat.index = [i.upper() for i in list(dat.index)]
        
        # intersection
        marker = set(ref.index)
        genes = set(dat.index)
        marker = list(marker&genes)
        self.__reference_data = ref.loc[marker,:]
        self.__mix_data = dat.loc[marker,:]
        if prints:
            print("number of used genes = {}".format(len(marker)))
        return

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

    def __fit(self,method='elasticnet',number_of_repeats=1,alpha=1,l1_ratio=0.05,nu=[0.25,0.5,0.75],max_iter=100000, prints=True):
        try:
            for i in range(number_of_repeats):
                res_mat = self.method_dict[method](self.__reference_data,self.__mix_data,alpha=alpha,l1_ratio=l1_ratio,nu=nu,max_iter=max_iter)
                # sum up
                if i == 0:
                    res = res_mat
                else:
                    res = res + res_mat
            if prints:
                print('fitting method : {}'.format(method))
            res = res / number_of_repeats
        except:
            res=np.nan
            print('fitting error')
            print('confirm fitting method name')
        self.__res=res
        return
    
    def __df_median(self,df,sep="_"):
        df_c = copy.deepcopy(df)
        df_c.columns=[i.split(sep)[0] for i in list(df_c.columns)]
        df_c = df_c.groupby(level=0,axis=1).median()
        return df_c
