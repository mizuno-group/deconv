# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 21:00:52 2021

@author: K.Morita, I.Azuma
"""

from ctypes import util
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import copy
from scipy.stats import rankdata
from sklearn.utils import resample
from tqdm import tqdm
from combat.pycombat import pycombat
import matplotlib.pyplot as plt

from _utils import utils, normalization, transformation

class Processing():

    def __init__(self):
        self.data=pd.DataFrame()
        self.res=pd.DataFrame()

    def annotation(self, ref_df, places:list=[0,1]):
        """
        annotate row IDs to gene names

        Parameters
        ----------
        ref_df : two rows of dataframe. e.g. ["Gene stable ID","MGI symbol"]
        places : list of positions of target rows in the ref_df

        """
        self.res = utils.annotation(self.res, ref_df, places=places)

    def trimming(self,threshold=0.9,strategy="median",trim=1.0,batch=False,split="_"):
        if batch:
            raw_batch = [t.split(split)[0] for t in self.data.columns.tolist()]
            batch_list = pd.Series(raw_batch).astype("category").cat.codes.tolist()
            self.res = utils.array_imputer(np.log2(self.res)+1,threshold=threshold,strategy=strategy,trim=trim,batch=True,lst_batch=batch_list,trim_red=False)
        else:
            self.res = utils.array_imputer(np.log2(self.res)+1,threshold=threshold,strategy=strategy,trim=trim,batch=False)

    def combat(self, batch_lst_lst=[[],[]],plot=False):
        self.res = utils.multi_batch_norm(self.res,batch_lst_lst=batch_lst_lst,do_plots=plot)

    def normalize(self, methods:list=[]):
        """normalize the target data with the selected methods"""
        dat = normalization.Normalization(methods=methods)
        dat.set_data(self.res)
        dat.perform_normalization()
        self.res = dat.res

    def transform(self, method:str=""):
        """transform the target data with the selected method"""
        dat = transformation.Transformation(method=method)
        dat.set_data(self.res)
        dat.transform()
        self.res = dat.res

    def set_data(self, df):
        self.data = df
        self.res = df
