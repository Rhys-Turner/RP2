# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 10:39:13 2023

@author: Rhys Turner
"""

##Kmeans cluster_lables 

import pandas as pd 
import os
import numpy as np 
from tslearn.utils import to_time_series_dataset
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.clustering import TimeSeriesKMeans

def read_csv_folder(folder_path):
    dfs = {}
    for filename in os.listdir(folder_path):       # int through the
        if filename.endswith(".csv"):

            csv_path = os.path.join(folder_path, filename) 
            df = pd.read_csv(csv_path)
            df_name = os.path.splitext(filename)[0]

            dfs[df_name] = df   # df file name as the key

    return dfs


## creating the folder path 
#folder_path = "C:/Users/Rhys Turner/OneDrive/Project_2_clean/-30_to_30_glu"
folder_path = "C:/Users/Rhys Turner/OneDrive/Project_2_clean/-1000_to_1000_glu"


## folder path is the path to the -30 to 30 or the -1000 to 1000 dataset. 
li = read_csv_folder(folder_path)

## droping indexing columns  must drop the index for the -30 dataset
def drop_columns(dataframes):
    for key, df in dataframes.items():
        df.drop(df.columns[0:1], axis=1, inplace=True)

#drop_columns(li)

## removed NaN values 
def rm_nans(dict_of_data_frames):
    for key, df in dict_of_data_frames.items():
        df = df.dropna()   # Remove NaN values from the data frame
        dict_of_data_frames[key] = df
    return dict_of_data_frames

## removing NaN values of the biomasses
li = rm_nans(li)


## removed all single values from list 
def rm_single_rows(dict_of_dfs):
    keys_to_rm = []
    for key, df in dict_of_dfs.items():
        if len(df) == 1:
            keys_to_rm.append(key) ## append all single rows to list 
    for key in keys_to_rm:  ## remove all that are in list 
        del dict_of_dfs[key]

    return dict_of_dfs

li = rm_single_rows(li)


## splitting the list by flux
def split_dict_by_flux(dict_of_dfs):
    dict_n = {} 
    dict_p = {}

    for key, df in dict_of_dfs.items():
        
        filtered_dict_n = df[df["flux"].between(-1000, 0)]  ## filter dictionary splitting the flux
        filtered_dict_p = df[df['flux'].between(0, 1000)] 

        if not filtered_dict_n.empty:
            dict_n[key] = filtered_dict_n   ## negitive dict 

        if not filtered_dict_p.empty:
            dict_p[key] = filtered_dict_p ## positive dict 

    return dict_n, dict_p

neg_dict, pos_dict = split_dict_by_flux(li)

## must remove the single a second time after the split of the datafarmes 
neg_dict = rm_single_rows(neg_dict)
pos_dict = rm_single_rows(pos_dict)


## function to drop the flux values 
def drop_first_column(dict_of_dfs):
    for key, df in dict_of_dfs.items():
        dict_of_dfs[key] = df.iloc[:, 1:]  # Drops the first column

    return dict_of_dfs


neg_dict = drop_first_column(neg_dict)
pos_dict = drop_first_column(pos_dict)


## this function also changes the negirive flux values "bear in mind" -- 
def flip_neg_flux(dict_of_dfs):
    for key, df in dict_of_dfs.items():
        
        dict_of_dfs[key] = df.iloc[::-1] ## inverting the negfluxes 
        
    return dict_of_dfs

inv_neg_dict =  flip_neg_flux(neg_dict)

## joining the negitive and positive dictionaries together, importantly if the 2 dictionaries have the same 
## key then "_neg_flux" is added to key of the negitive dictionary entry 

def join_dicts(pos_dict, neg_dict):
    joined_dict = pos_dict.copy()

    for key, value in neg_dict.items():
        if key in joined_dict:  ## checking if key is in the positive dict 
            new_key = key + "_neg_flux"  ## adding neg_flux to key if already in pos dicts 
            joined_dict[new_key] = value
        else:
            joined_dict[key] = value

    return joined_dict

pos_neg_dict = join_dicts(pos_dict, inv_neg_dict)


## z-normalization process/ time series meanvarience

def apply_tsmv(dict_of_dfs):
    norm_dfs = {}
    for reaction, df in dict_of_dfs.items():
        
        tsd = to_time_series_dataset([df["Biomass"].values]) ## formats the dataset to the correct format for tslearn
        
        transformer = TimeSeriesScalerMeanVariance() ## same as a z-norm from scikitlearn
        norm_tsd = transformer.fit_transform(tsd)
        
        df["Biomass"] = norm_tsd[0].flatten() ## updates the bio column with normalised vals
        norm_dfs[reaction] = df ## stoes it back with original key (reaction)
        
    return norm_dfs

pos_neg_dict_znorm = apply_tsmv(pos_neg_dict)



def pad_dfs_nans(dict_of_dfs):
    max_len = max(len(df) for df in dict_of_dfs.values()) ## find the longest df in the dict 

    for x in dict_of_dfs:  #itr through keys  
        df = dict_of_dfs[x] 
        cur_len = len(df) ## find the length of that current df is 
        
        if cur_len < max_len:  ## if the current df is less than the max df 
            pad_len = max_len - cur_len   ## how much nan padding pad_len
            pad_df = pd.DataFrame(np.nan, index=np.arange(pad_len), columns=df.columns) ## extend with NaN rows making as long as max
            dict_of_dfs[x] = pd.concat([df, pad_df])

    return dict_of_dfs

ts_full  = pad_dfs_nans(pos_neg_dict_znorm)


def kmeans_clusters(n_clusters, data):
    
    list_of_dataframes = [df.values for df in data.values()]  ## make a list of numpy arrays
    
    combined_array = np.stack(list_of_dataframes) ## combins to a single numpy array
    keys = list(data.keys())        ## identifier key
    
    kmeans = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", n_init=10)        ## K-Means clustering algo 
    
    cluster_labels = kmeans.fit_predict(combined_array)      ## retrieveing cluster labales
      
    clusters = {}
    for i, label in enumerate(cluster_labels):      # Iter through the samples and assign them to respective clusters or makes a new one 
        if label not in clusters:
            clusters[label] = {'reactions': []}
        clusters[label]['reactions'].append(keys[i])

    return clusters    


## put in the number of clusters needed 

clusters = kmeans_clusters(2, ts_full)
## gives clusters as a dict of dicts for plotting 