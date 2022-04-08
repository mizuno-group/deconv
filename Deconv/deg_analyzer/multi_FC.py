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

from deg_analyzer import abst

class Deg_Multi_FC(abst.Deg_abst):
    def __init__(self):
        super().__init__()
        self.min_FC=pd.DataFrame()
        
    ### main ###
    def deg_extraction(self,sep_ind:str="_",number:int=150,limit_CV:float=0.1,limit_FC:float=1.5):
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
                    self.__logFC()
                    df_logFC = self.df_logFC
                    df_logFC.columns = [o]
                    self.tmp_summary = pd.concat([self.tmp_summary,df_logFC],axis=1)
            tmp_min = self.tmp_summary.T.min()
            self.df_minFC = pd.DataFrame(tmp_min)
            self.__calc_CV()
            pickup_genes = self.__selection(self.minFC,self.df_CV,number=number,limit_CV=limit_CV,limit_FC=limit_FC)
            self.pickup_genes_list.append(pickup_genes)
            self.min_FC = pd.concat([self.min_FC,tmp_min],axis=1)
        self.min_FC.columns = sorted(list(set(immunes)))
        self.pickup_genes_df=pd.DataFrame(self.pickup_genes_list).T.dropna(how="all")
        self.pickup_genes_df.columns = sorted(list(set(immunes)))
        #curate = [[i for i in t if str(i)!='nan'] for t in self.pickup_genes_list]
        #self.deg_dic = dict(zip(list(set(immunes)),curate))
    
    def create_ref(self,sep="_",number=200,limit_CV=1,limit_FC=1.5,log2=False,plot=False):
        """
        create reference dataframe which contains signatures for each cell

        """
        ref_inter_df = copy.deepcopy(self.df_all)
        df2 = copy.deepcopy(self.df_all)
        if log2:
            df2 = np.log2(df2+1)
        self.df_all = df2
        # DEG extraction
        self.deg_extraction(sep_ind=sep,number=number,limit_CV=limit_CV,limit_FC=limit_FC)
        signature = self.get_res() # union of each reference cell's signatures
        sig_ref = ref_inter_df.loc[signature]
        final_ref = self.__df_median(sig_ref,sep=sep)
        if plot:
            print("signature genes :",len(signature))
            sns.clustermap(final_ref,col_cluster=False,z_score=0)
            plt.show()
        self.final_ref = final_ref

