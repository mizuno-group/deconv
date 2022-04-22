# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 11:16:33 2022

@author: I.Azuma, K.Morita
"""

import pandas as pd
import copy
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from . import utils

class Deg_Multi_FC():
    def __init__(self):
        self.df_mix=pd.DataFrame()
        self.df_ref=pd.DataFrame()
        self.df_target=pd.DataFrame()
        self.df_else=pd.DataFrame()
        self.df_logFC=pd.DataFrame()
        self.df_CV=pd.DataFrame()
        self.final_ref=pd.DataFrame()
        self.seps=[]
        self._pickup_genes=[]
        self._pickup_genes_lst=[]
        self._pickup_genes_df=pd.DataFrame()
        self.min_FC=pd.DataFrame()
        
    ### main ###
    def set_data(self,df_mix,df_ref):
        """
        set data
        """
        self.df_mix = df_mix
        self.df_ref = df_ref

    def deg_extraction(self,sep_ind:str="_",number:int=150,limit_CV:float=0.1,limit_FC:float=1.5,prints=False):
        """
        Define DEGs between the target and other one for the cells that make up the REFERENCE.
        e.g. B cell vs CD4, B cell vs CD8, ...

        Parameters
        ----------
        sep_ind : str
            Assume the situation that the columns name is like "CellName_GSEXXX_n". The default is "_".
        number : int
            Number of top genes considered as DEGs. The default is 150.
        limit_CV : float
            Coefficient of Variation threshold. The default is 0.3.
        limit_FC : TYPE, float
            Minimum threshold for logFC detection. The default is 1.5.

        """
        df_c = copy.deepcopy(self.df_ref)
        immunes = [t.split(sep_ind)[0] for t in df_c.columns.tolist()]
        df_c.columns = immunes
        self.min_FC = pd.DataFrame()
        self.pickup_genes_list = []
        for c in sorted(list(set(immunes))):
            self.df_target = df_c[c]
            self.tmp_summary = pd.DataFrame()
            for o in sorted(list(set(immunes))):
                if o == c:
                    pass
                else:
                    self.df_else = df_c[o]
                    df_logFC = utils._logFC(self.df_target,self.df_else)
                    df_logFC.columns = [o]
                    self.tmp_summary = pd.concat([self.tmp_summary,df_logFC],axis=1)
            tmp_min = self.tmp_summary.T.min()
            self.df_minFC = pd.DataFrame(tmp_min)
            self.df_CV = utils._calc_CV(self.df_target)
            pickup_genes = self._selection(self.df_minFC,self.df_CV,number=number,limit_CV=limit_CV,limit_FC=limit_FC)
            self.pickup_genes_list.append(pickup_genes)
            self.min_FC = pd.concat([self.min_FC,tmp_min],axis=1)
        self.min_FC.columns = sorted(list(set(immunes)))
        self.pickup_genes_df=pd.DataFrame(self.pickup_genes_list).T.dropna(how="all")
        self.pickup_genes_df.columns = sorted(list(set(immunes)))
        #curate = [[i for i in t if str(i)!='nan'] for t in self.pickup_genes_list]
        #self.deg_dic = dict(zip(list(set(immunes)),curate))
    
    def create_ref(self,sep="_",number=200,limit_CV=1,limit_FC=1.5,log2=False,plot=False,prints=False,**kwargs):
        """
        create reference dataframe which contains signatures for each cell

        """
        ref_inter_df = copy.deepcopy(self.df_ref)
        if log2:
            self.df_ref = copy.deepcopy(np.log2(self.df_ref+1))
        # DEG extraction
        self.deg_extraction(sep_ind=sep,number=number,limit_CV=limit_CV,limit_FC=limit_FC,prints=prints)
        signature = utils._get_res(self._pickup_genes) # union of each reference cell's signatures
        sig_ref = ref_inter_df.loc[signature]
        final_ref = utils._df_median(sig_ref,sep=sep)
        if plot:
            print("signature genes :",len(signature))
            sns.clustermap(final_ref,col_cluster=False,z_score=0)
            plt.show()
        self.final_ref = final_ref

    def narrow_intersection(self):
        """take intersection genes"""
        self.df_mix, self.df_ref = utils._intersection_index(self.df_mix,self.df_ref)
        
    def _selection(self,df_FC,df_CV,number=50,limit_CV=0.1,limit_FC=1.5,prints=False):
        df_CV=self.df_CV
        df_FC=df_FC.sort_values(0,ascending=False)
        genes=df_FC.index.tolist()
        pickup_genes=[]
        ap = pickup_genes.append
        i=0
        while len(pickup_genes)<number:
            if len(genes)<i+1:
                pickup_genes = pickup_genes+[np.nan]*number
                if prints:
                    print('not enough genes picked up')
            elif df_CV.iloc[i,0] < limit_CV and df_FC.iloc[i,0] > limit_FC:
                ap(genes[i])
            i+=1
        else:
            self._pickup_genes = self._pickup_genes + pickup_genes
            return pickup_genes

    def plot_deg_heatmap(self,df_ref):
        """
        Overlook the DEGs definition condition with heatmap plotting
        """
        df = copy.deepcopy(df_ref)
        df.index = [t.upper() for t in df.index.tolist()]
        df.fillna(0)
        for i,sample in enumerate(self.samples):
            tmp_df = df.loc[[t for t in self.pickup_genes_df[sample] if str(t)!='nan']]
            sns.heatmap(tmp_df)
            plt.title(sample)
            plt.show()