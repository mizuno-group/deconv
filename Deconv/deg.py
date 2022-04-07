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

from DEG_analyzer import deg_ttest as dt
from DEG_analyzer import deg_multi_FC as dmf

import processing

class Deg():
    ### init ###
    def __init__(self):
        self.df_mix=pd.DataFrame()
        self.df_all=pd.DataFrame()
        self.df_target=pd.DataFrame()
        self.df_else=pd.DataFrame()
        self.df_logFC=pd.DataFrame()
        self.df_CV=pd.DataFrame()
        self.final_ref=pd.DataFrame()
        self.seps=[]
        self.__pickup_genes=[]
        self.__pickup_genes_df=pd.DataFrame()
        self.__method_dict={"ttest":dt.Deg_ttest(),"multiFC":dmf. Deg_Multi_FC()}
        self.__processing=processing
    
    ### main ###
    def set_method(self,method="ttest"):
        self.__deg_class = self.__method_dict[method]
        print("method :",method)
        
    def set_data(self,df_mix,df_all):
        """
        set directly
        """
        self.df_mix = df_mix
        self.df_all = df_all
        
        print(self.df_mix.shape)
        print(self.df_all.shape)
        
    
    def pre_processing(self,do_ann=False,ann_df=None,do_log2=True,do_quantile=True,do_trimming=False,do_drop=True):
        if do_ann:
            self.df_all = self.__processing.annotation(self.df_all,ann_df)
        else:
            pass
        if do_log2:
            df_c = copy.deepcopy(self.df_all)
            self.df_all = self.__processing.log2(df_c)
            print("log2 conversion")
        else:
            pass
        if do_quantile:
            df_c = copy.deepcopy(self.df_all)
            self.df_all = self.__processing.quantile(df_c)
            print("quantile normalization")
        else:
            pass
        if do_trimming:
            df_c = copy.deepcopy(self.df_all)
            raw_batch = [t.split("_")[0] for t in df_c.columns.tolist()]
            batch_list = pd.Series(raw_batch).astype("category").cat.codes.tolist()
            self.df_all = self.__processing.array_imputer(df_c,threshold=0.9,strategy="median",trim=1.0,batch=True,lst_batch=batch_list, trim_red=False)
            print("trimming")
        else:
            pass
        if do_drop:
            df_c = copy.deepcopy(self.df_all)
            replace = df_c.replace(0,np.nan)
            drop = replace.dropna(how="all")
            drop = drop.fillna(0)
            self.df_all = drop
            print("drop nan")
    
    def narrow_intersec(self):
        """
        df_mix : deconvolution target data (for which we want to know the population of each immune cell)
        df_all : reference data for estimate the population of the immune cells (e.g. LM6)
        
        Note that mix_data is already processed (trimmed) in general (log2 --> trim+impute --> batch norm --> QN).
        This is because the robustness of the analysis is reduced if the number of genes to be analyzed is not narrowed down to a certain extent.
        """   
        # trimming
        mix_data = copy.deepcopy(self.df_mix)
        reference_data = copy.deepcopy(self.df_all)
        
        self.df_mix, self.df_all = self.__intersection_index(mix_data,reference_data) # update
        print("narrowd gene number :",len(self.df_all))
        
    
    def create_ref(self,**kwargs):
        """
        create reference dataframe which contains signatures for each cell

        """
        mix_inter_df = copy.deepcopy(self.df_mix)
        ref_inter_df = copy.deepcopy(self.df_all)
        
        self.__deg_class.set_data(df_mix=mix_inter_df,df_all=ref_inter_df)
        self.__deg_class.create_ref(**kwargs)
        self.final_ref = self.__deg_class.final_ref
        
    
    def def_signature(self,sep="_",number=200,limit_CV=1,q_limit=0.05,log2=False):
        """
        define signature (marker) genes in the reference data
        """
        df2 = copy.deepcopy(self.df_all)
        if log2:
            df2 = np.log2(df2+1)
        else:
            pass
        cluster, a = self.sepmaker(df=df2,delimiter=sep)
        print(cluster,a)
        self.prepare(df2,sep=cluster)
        
        self.DEG_extraction(method='ttest',number=number,q_limit=q_limit,limit_CV=limit_CV)
        res = self.get_res()
        return res
    
    ### in/out put ###
    def get_res(self):
        self.__pickup_genes=[i for i in self.__pickup_genes if str(i)!='nan']
        self.__pickup_genes=list(set(self.__pickup_genes))
        return self.__pickup_genes
    
    def get_res_df(self):
        return self.__pickup_genes_df
    
        
    ### DEG method ###
    def __intersection_index(self,df,df2):
        ind1 = df.index.tolist()
        ind2 = df2.index.tolist()
        df.index = [i.upper() for i in ind1]
        df2.index = [i.upper() for i in ind2]
        ind = list(set(df.index) & set(df2.index))
        df = df.loc[ind,:]
        df2 = df2.loc[ind,:]
        return df,df2
    
    def __DEG_extraction_qval(self,q_limit=0.1,**kwargs):
        p_vals = [st.ttest_ind(self.df_target.iloc[i,:],self.df_else.iloc[i,:],equal_var=False)[1] for i in range(len(self.df_target.index))]
        p_vals = [float(str(i).replace('nan','1')) for i in p_vals]
        q_vals = sm.multipletests(p_vals, alpha=0.1, method='fdr_bh')[1]
        TF=[True if i<q_limit else False for i in list(q_vals)]
        print("extracted genes number = {}".format(TF.count(True)))
        self.df_target=self.df_target.loc[TF,:]
        self.df_else=self.df_else.loc[TF,:]
        return
        
    ### processing ###
    def sepmaker(self,df=None,delimiter='.'):
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

    def __df_separate(self,sep):
        df = copy.deepcopy(self.df_all)
        df.columns=[str(i) for i in sep]
        self.df_target=df.loc[:,df.columns.str.contains('1')]
        self.df_else=df.loc[:,df.columns.str.contains('0')]
        
    def __make_seplist(self,sep=[0,0,0,1,1,1,2,2,2]):
        res = [[0 if v!=i else 1 for v in sep] for i in list(range(max(sep)+1))]
        self.seps=res  

    def __logFC(self):
        # calculate df_target / df_else logFC
        df_logFC = self.df_target.T.median() - self.df_else.T.median()
        df_logFC = pd.DataFrame(df_logFC)
        self.df_logFC = df_logFC
        """
        # for not logged dataframe
        df_FC=self.df_target.T.median() / self.df_else.T.median()
        df_logFC = np.log(df_FC)
        df_logFC = df_logFC.replace(np.inf,np.nan)
        df_logFC = df_logFC.replace(-np.inf,np.nan)
        df_logFC = df_logFC.dropna()
        df_logFC = pd.DataFrame(df_logFC)
        self.df_logFC = df_logFC
        """
    
    def __calc_deviation(self):
        df_CV = pd.DataFrame(index=self.df_target.index)
        df_CV.loc[:,'CV'] = st.variation(self.df_target.T)
        df_CV = df_CV.replace(np.inf,np.nan)
        df_CV = df_CV.replace(-np.inf,np.nan)
        df_CV = df_CV.dropna()
        self.df_CV=df_CV
    
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
        
    def __intersection(self):
        lis1 = list(self.df_logFC.index)
        lis2 = list(self.df_CV.index)
        self.df_logFC = self.df_logFC.loc[list(set(lis1)&set(lis2)),:]
        self.df_CV = self.df_CV.loc[list(set(lis1)&set(lis2)),:]

    def __same_gene_median(self):
        df = copy.deepcopy(self.df_all)
        df2 = pd.DataFrame()
        dup = df.index[df.index.duplicated(keep="first")]
        gene_list = pd.Series(dup).unique().tolist()
        if len(gene_list) != 0:
            for gene in gene_list:
                new = df.loc[gene].median()
                df2[gene] = new
        df = df.drop(gene_list)
        df = pd.concat([df,df2.T])
        self.df_all = df

    def __selection(self,number=50,limit_CV=0.1):
        self.__intersection()
        df_logFC=self.df_logFC.sort_values(by=0,ascending=False)
        df_CV=self.df_CV.loc[list(df_logFC.index),:]
        genes=list(df_logFC.index)
        pickup_genes=[]
        ap = pickup_genes.append
        i=0
        while len(pickup_genes)<number:
            if len(genes)<i+1:
                pickup_genes = pickup_genes+[np.nan]*number
                print('not enough genes picked up')
            elif df_CV.iloc[i,0] < limit_CV and df_logFC.iloc[i,0] > 1:
                ap(genes[i])
            i+=1
        else:
            self.__pickup_genes = self.__pickup_genes + pickup_genes
            return pickup_genes

    def __selection_loop(self,loop_range=[50,200],limit_CV=0.1):
        self.__intersection()
        df_logFC=self.df_logFC.sort_values(by=0,ascending=False)
        df_CV=self.df_CV.loc[list(df_logFC.index),:]
        genes=list(df_logFC.index)
        
        pickup_genes_list = []
        ap_list=pickup_genes_list.append
        for v in range(loop_range[0],loop_range[1]+1):
            pickup_genes=[]
            ap = pickup_genes.append
            i=0
            while len(pickup_genes)<v:
                if len(genes)<i+1:
                    pickup_genes = pickup_genes+[np.nan]*v
                elif df_CV.iloc[i,0] < limit_CV and df_logFC.iloc[i,0] > 1:
                    ap(genes[i])
                i+=1
            else:
                ap_list(pickup_genes)
        return pickup_genes_list
    
    ### legacy ###
    def prepare(self,df,sep=[0,0,0,1,1,1,2,2,2]):
        """
        df : dataframe or file path
        sep : sample separation by conditions
        
        """
        self.set_df_all(df)
        self.__make_seplist(sep=sep)
        self.__same_gene_median()
    
    def set_df_all(self,df_all):
        """
        set directly
        """
        self.df_all = df_all
        
        print(self.df_all.shape)
    
    def condition_number(self,loop_range=[50,200],method='ttest',q_limit=0.1,limit_CV=0.3,columns_sep='.'):        
        res_list = []
        ap = res_list.append
        for i,sep in enumerate(self.seps):
            print(i)
            self.__df_separate(sep)
            self.__method_dict[method](q_limit=q_limit)
            self.__logFC()
            self.__calc_CV()
            pickup_genes_list = self.__selection_loop(loop_range=loop_range,limit_CV=limit_CV)
            ap(pickup_genes_list)
        
        condition_numbers = []
        ap = condition_numbers.append
        best_number=0
        for l in range(len(range(loop_range[0],loop_range[1]+1))):
            pickup_genes = []
            for v in res_list:
                pickup_genes = pickup_genes + v[l]
            
            pickup_genes = [i for i in pickup_genes if str(i)!='nan']
            pickup_genes = list(set(pickup_genes))
            
            df_res = self.df_all.loc[pickup_genes,:]
            df_res.columns=[i.split(columns_sep)[0] for i in list(df_res.columns)]
            df_res = df_res.groupby(level=0,axis=1).median()
            condition_number = np.linalg.cond(df_res)
            if len(condition_numbers) == 0:
                best_number=range(loop_range[0],loop_range[1]+1)[l]
            else:    
                if min(condition_numbers) > condition_number:
                    best_number=range(loop_range[0],loop_range[1]+1)[l]
            ap(condition_number)
        print('pickup number = {}'.format(best_number))
        self.DEG_extraction(method=method,number=best_number,q_limit=q_limit,limit_CV=limit_CV)
        return condition_numbers
    
    