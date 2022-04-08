# -*- coding: utf-8 -*-
"""
2022/04/08

Modules for deg analyzer

@author: K.Morita
"""

import copy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Deg_abst():
    
    def __init__(self):
        self.df_mix=pd.DataFrame()
        self.df_ref=pd.DataFrame()
        self.df_target=pd.DataFrame()
        self.df_else=pd.DataFrame()
        self.df_logFC=pd.DataFrame()
        self.df_CV=pd.DataFrame()
        self.final_ref=pd.DataFrame()
        self.__pickup_genes=[]
        self.__pickup_genes_lst=[]
        self.__pickup_genes_df=pd.DataFrame()

    ### Main ###
    def set_data(self,df_mix,df_ref):
        """
        set data
        """
        self.df_mix = df_mix
        self.df_all = df_ref

    def get_res(self):
        self.__pickup_genes=[i for i in self.__pickup_genes if str(i)!='nan']
        self.__pickup_genes=list(set(self.__pickup_genes))
        return self.__pickup_genes

    def narrow_intersection(self):
        """take intersection genes"""
        self.df_mix, self.df_ref = self.__intersection_index(self.df_mix,self.df_ref)

    def create_ref(self):
        raise NotImplementedError

    def deg_extract(self):
        raise NotImplementedError

    def plot_deg_heatmap(self):
        """
        Overlook the DEGs definition condition with heatmap plotting
        """
        df = copy.deepcopy(self.df_ref)
        df.index = [t.upper() for t in df.index.tolist()]
        df.fillna(0)
        for i,sample in enumerate(self.samples):
            tmp_df = df.loc[[t for t in self.pickup_genes_df[sample] if str(t)!='nan']]
            sns.heatmap(tmp_df)
            plt.title(sample)
            plt.show()

    ### modules ###
    def __intersection_index(self,df,df2):
        ind1 = df.index.tolist()
        ind2 = df2.index.tolist()
        df.index = [i.upper() for i in ind1]
        df2.index = [i.upper() for i in ind2]
        ind = list(set(df.index) & set(df2.index))
        df = df.loc[ind,:]
        df2 = df2.loc[ind,:]
        return df,df2
    
    def __sepmaker(self,df=None,delimiter='.'):
        if df is None:
            df = self.df_all
        else:
            pass
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

    def __df_median(self,df,sep="_"):
        df_c = copy.deepcopy(df)
        df_c.columns=[i.split(sep)[0] for i in list(df_c.columns)]
        df_c = df_c.groupby(level=0,axis=1).median()
        return df_c

    def __logFC(self):
        # calculate df_target / df_else logFC
        df_logFC = self.df_target.T.median() - self.df_else.T.median()
        df_logFC = pd.DataFrame(df_logFC)
        self.df_logFC = df_logFC
    
    def __calc_CV(self):
        """
        CV : coefficient of variation
        """
        df_CV = np.std(self.df_target.T) / np.mean(self.df_target.T)
        df_CV.index = self.df_target.index
        df_CV = df_CV.replace(np.inf,np.nan)
        df_CV = df_CV.replace(-np.inf,np.nan)
        df_CV = df_CV.dropna()
        self.df_CV=pd.DataFrame(df_CV)
    
    def __selection(self,df_FC,df_CV,number=50,limit_CV=0.1,limit_FC=1.5):
        df_CV=self.df_CV
        df_FC=df_FC.sort_values(0,ascending=False)
        genes=df_FC.index.tolist()
        pickup_genes=[]
        ap = pickup_genes.append
        i=0
        while len(pickup_genes)<number:
            if len(genes)<i+1:
                pickup_genes = pickup_genes+[np.nan]*number
                print('not enough genes picked up')
            elif df_CV.iloc[i,0] < limit_CV and df_FC.iloc[i,0] > limit_FC:
                ap(genes[i])
            i+=1
        else:
            self.__pickup_genes = self.__pickup_genes + pickup_genes
            return pickup_genes
    