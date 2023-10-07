# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 10:27:33 2023

@author: Rhys Turner
"""

### DBSCAN cluster labels 


import pandas as pd 
from scipy import stats
from dtaidistance import dtw 
from sklearn.cluster import DBSCAN
import os

def read_csv_folder(folder_path):
    dfs = {}
    for filename in os.listdir(folder_path):       # int through filess
        if filename.endswith(".csv"):

            csv_path = os.path.join(folder_path, filename) 
            df = pd.read_csv(csv_path)
            df_name = os.path.splitext(filename)[0]

            dfs[df_name] = df   # df file name as the key

    return dfs  ## store as dict of dfs 

## creating folder path

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


## removed all sinlge vales from list 
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


## must remove the single a second time afte the split of the datafarmes 
neg_dict = rm_single_rows(neg_dict)
pos_dict = rm_single_rows(pos_dict)


## function to drop the flux values 
def drop_first_column(dict_of_dfs):
    for key, df in dict_of_dfs.items():
        dict_of_dfs[key] = df.iloc[:, 1:]  # Drops the first column

    return dict_of_dfs

## dropping the flux values for the clustering algo
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
            new_key = key + "_neg_flux"  ## adding neg flux to key 
            joined_dict[new_key] = value
        else:
            joined_dict[key] = value

    return joined_dict


## join dicts
pos_neg_dict = join_dicts(pos_dict, inv_neg_dict)

## z-normalization however missing dictionaries where all the values are the same 
def z_norm_dfs(dict_of_dfs):
    norm_data = {}
    for key, df in dict_of_dfs.items():

        if not df.eq(df.iloc[0]).all().all():           # Check values in the data frame are the same

            norm_df = pd.DataFrame(stats.zscore(df), columns=df.columns, index=df.index)  # Apply z-normalization to the data frame
            
            norm_data[key] = norm_df                   
            
        else:
            norm_data[key] = df              ## skipping if all the values are equal

    return norm_data

z_norm_pos_neg_dict = z_norm_dfs(pos_neg_dict)


## converting the dataframes of the dictionaries to arrays for the Dtw distance matrix 
def dfs_to_arrays(dict_of_dfs):
    for key, df in dict_of_dfs.items():
        
        dict_of_dfs[key] = df.values            
                                            
    return dict_of_dfs

ts_full = dfs_to_arrays(z_norm_pos_neg_dict)        


## converting to a list of np.arrays for the distance matrix
def dtwdistance_matrix(dict_of_arrays):
    ds_arrays =list(dict_of_arrays.values())        ## list of values (arrays)
    ds_matrix = dtw.distance_matrix_fast(ds_arrays)  ## dtw distance matrix
    
    return ds_matrix

ds = dtwdistance_matrix(ts_full)



def DBscan_clustering_with_dtw(data_dict, ds_matrix, eps, MinPts):  ## dbscan algo 
    keys = list(data_dict.keys()) ## identifier key 

    dbscan = DBSCAN(eps=eps, min_samples = MinPts, metric='precomputed')   ## DBSCAN clustering algo 
    
    
    cluster_labels = dbscan.fit_predict(ds_matrix)  ## retrieveing cluster labales
    
    # Retrieve clusters and keys
    clusters = {}
    for i, label in enumerate(cluster_labels):  ## Iter through the samples and assign them to respective clusters
        if label not in clusters:
            clusters[label] = {'reactions': []}
        clusters[label]['reactions'].append(keys[i])

    return clusters


## change this depending on parametrs/dataset ect
clusters_dbscan = DBscan_clustering_with_dtw(ts_full, ds, 0.26, 14) ## put in the epsilon and MinPts 
## gives clusters for plotting in a dict of dicts 