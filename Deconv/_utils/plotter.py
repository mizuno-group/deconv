# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 10:25:03 2022

Plotting modules for deconvolution

@author: I.Azuma, K.Morita
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_hist_loop(df,topn=5,bins=200,title="Test"):
    """plot histgram with for-loop"""
    for i in range(topn):
        plt.hist(df.iloc[:,i],alpha=0.6,bins=bins)
    plt.title(title)
    plt.show()
    

def plot_box(melt_df):
    """
    plot box plot and point plot of dataframe
    melt_df :
        variable  value
        XXX       0.04
        XXX       0.03
         .         .
         .         .
         .         .
        YYY       0.05
        YYY       0.02
    """
    sns.set()
    sns.set_style('whitegrid')
    sns.set_palette('Set3')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sns.boxplot(x='variable', y='value', data=melt_df, showfliers=False, ax=ax)
    sns.stripplot(x='variable', y='value', data=melt_df, jitter=True, color='black', ax=ax)
    plt.show()


def plot_immune_box(df,sort_index:list=[], control_names:list=["control, ctrl"], row_n:int=2,col_n:int=3, sep="_"):
    """plot box plot and point plot of dataframe"""
    immunes = df.columns.tolist()
    df.index = [i.split(sep)[0] for i in df.index]
    if len(sort_index)==0:
        sort_index=sorted(list(set(df.index.tolist())))
    df = df.loc[sort_index,:]
    fig = plt.figure(figsize=(5*col_n,5*row_n))
    for i,immune in enumerate(immunes):
        df_melt = pd.melt(df[[immune]].T)
        final_melt = pd.DataFrame()
        for t in sort_index:
            final_melt = pd.concat([final_melt,df_melt[df_melt["variable"]==t]])
        my_pal = {val: "yellow" if val in control_names else "lightgreen" for val in final_melt.variable.unique()}
        ax = fig.add_subplot(row_n,col_n,i+1)
        sns.boxplot(x='variable', y='value', data=final_melt, width=0.6, showfliers=False, notch=False, ax=ax, boxprops=dict(alpha=0.8), palette=my_pal)
        sns.stripplot(x='variable', y='value', data=final_melt, jitter=True, color='black', ax=ax)
        ax.set_title(immune)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().yaxis.set_ticks_position('left')
        plt.gca().xaxis.set_ticks_position('bottom')
    plt.grid(False)
    plt.tight_layout()
    plt.show()
