# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 12:42:26 2022

Determine DEGs to compare target cell (Monocyte) to others (B,CD4,CD8,...) with ttest

@author: I.Azuma
"""
import pandas as pd
import copy
from scipy import stats as st
import statsmodels.stats.multitest as sm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random

from . import abst

class Deg_ttest(abst.Deg_abst):
    def __init__(self):
        super().__init__()
        self.seps=[]
        
    ### main ###
    def deg_extraction(self,sep_ind="_",number=150,q_limit=0.05,limit_CV=0.1,base="logFC"):
        df_c = copy.deepcopy(self.df_ref)
        cluster, self.samples = super()._sepmaker(df=df_c,delimiter=sep_ind)
        self.__make_seplist(sep=cluster) # prepare self.seps
        self._pickup_genes = []
        self.pickup_genes_list = []
        ap = self.pickup_genes_list.append
        for i,sep in enumerate(self.seps):
            self.__df_separate(sep)
            self.__DEG_extraction_qval(q_limit=q_limit)
            self.df_logFC = super()._logFC(self.df_target,self.df_else)
            self.df_CV = super()._calc_CV(self.df_target)
            pickup_genes = self._selection(self.df_logFC, self.df_CV,number=number,limit_CV=limit_CV,base=base)
            ap(pickup_genes)
        self._pickup_genes_df=pd.DataFrame(self.pickup_genes_list).T
        self.pickup_genes_dic = dict(zip(self.samples,self.pickup_genes_list))
    
    def create_ref(self,sep="_",number=200,limit_CV=1,q_limit=0.05,log2=False,base="logFC",plot=False,**kwargs):
        """
        create reference dataframe which contains signatures for each cell
        """
        ref_inter_df = copy.deepcopy(self.df_ref)
        if log2:
            self.df_ref = copy.deepcopy(np.log2(self.df_ref+1))
        # DEG extraction
        self.deg_extraction(sep_ind=sep,number=number,q_limit=q_limit,limit_CV=limit_CV,base=base)
        signature = super().get_res(self._pickup_genes) # union of each reference cell's signatures
        sig_ref = ref_inter_df.loc[signature,:]
        final_ref = super()._df_median(sig_ref,sep=sep)
        if plot:
            print("signature genes :",len(signature))
            sns.clustermap(final_ref,col_cluster=False,z_score=0)
            plt.show()
        self.final_ref = final_ref
    
    '''
    def create_random_ref(self,sep="_",seed=123,high_cut=100.0,low_cut=0.0,do_plot=False):
        """
        create randomized reference which is the same size to the correct reference used in deconvolution
        """
        if self.final_ref is None:
            raise ValueError("!! Conduct create_ref() at fist !!")
        n = len(self.final_ref)
        tmp_df = copy.deepcopy(self.df_ref)
        s_max = pd.DataFrame(tmp_df.T.max()).sort_values(0)
        selected = s_max[(s_max[0]>low_cut)&(s_max[0]<high_cut)]
        if len(selected)<n:
            print(len(selected),"<",n)
            raise ValueError("!! Threshold setting is too strict !!")
        random.seed(seed)
        random_sig = random.sample(selected.index.tolist(),n)
        ref_inter_df = copy.deepcopy(self.df_ref)
        random_tmp = ref_inter_df.loc[random_sig]
        random_ref = self.__df_median(random_tmp,sep=sep)
        if do_plot:
            try:
                sns.clustermap(random_ref,col_cluster=False,z_score=0)
            except:
                sns.clustermap(random_ref,col_cluster=False)
                print("without Z-score due to the inf values error")
            plt.show()
        else:
            pass
        self.random_ref = random_ref
    '''
    
    ### processing ###
    def narrow_intersection(self):
        """take intersection genes"""
        self.df_mix, self.df_ref = super()._intersection_index(self.df_mix,self.df_ref)

    def __make_seplist(self,sep=[0,0,0,1,1,1,2,2,2]):
        seps = [[0 if v!=i else 1 for v in sep] for i in list(range(max(sep)+1))]
        self.seps=seps
    
    def __df_separate(self,sep):
        df = copy.deepcopy(self.df_ref)
        df.columns=[str(i) for i in sep]
        self.df_target=df.loc[:,df.columns.str.contains('1')]
        self.df_else=df.loc[:,df.columns.str.contains('0')]
    
    def __DEG_extraction_qval(self,q_limit=0.1,**kwargs):
        p_vals = [st.ttest_ind(self.df_target.iloc[i,:],self.df_else.iloc[i,:],equal_var=False)[1] for i in range(len(self.df_target.index))]
        p_vals = [float(str(i).replace('nan','1')) for i in p_vals]
        q_vals = sm.multipletests(p_vals, alpha=0.1, method='fdr_bh')[1]
        self.df_qval = pd.DataFrame({"q_value":q_vals},index=self.df_target.index.tolist()).sort_values("q_value",ascending=True)
        TF=[True if i<q_limit else False for i in list(q_vals)]
        print("extracted genes number = {}".format(TF.count(True)))
        self.df_target=self.df_target.loc[TF,:]
        self.df_else=self.df_else.loc[TF,:]
        return
    
    def _selection(self,df_FC_in,df_CV_in,number=50,limit_CV=0.1,limit_FC=1.5,base="logFC"):
        df_FC=copy.deepcopy(df_FC_in)
        df_CV=copy.deepcopy(df_CV_in)
        df_qval=self.df_qval.loc[df_FC.index.tolist()].sort_values("q_value")
        if base=="logFC":
            df_FC=df_FC.sort_values(by=0,ascending=False)
            genes=df_FC.index.tolist()
        elif base=="qvalue":
            genes=df_qval.index.tolist()
        df_FC = df_FC.loc[genes,:]
        df_CV = df_CV.loc[genes,:]
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
            self._pickup_genes = self._pickup_genes + pickup_genes
            return pickup_genes
    
