#############################################################
# File: xai_utils.py
# Goal: Functions to extract information from counterfactuals
# Author: Aurora Ramirez (Univ. of Cordoba, Spain)
#############################################################

import pandas as pd
import numpy as np

def counterfactuals_stats(sample, df_counterfactuals):
    '''
    This function provides statistics of a data frame containing conterfactual generated by DiCE for one sample.
    It returns a data frame with the number of feature changes and min/max changes for each counterfactual.
    Arguments
    ---------
    sample: The original test sample
    df_counterfactuals: The data frame of counterfactuals returned by DiCE
    Return
    ---------
    A data frame with one row for each counterfactual and columns with the following values: number of features changed,
    minimum change, feature corresponding to the minimum change, maximum change, feature corresponding to the maximum change. 
    '''
    feature_names = df_counterfactuals.drop('target', axis=1).columns
    num_cfs = df_counterfactuals.shape[0]

    cfs_index = list()
    num_feature_changes = list()
    min_feature_changes = list()
    min_feature_names = list()
    max_feature_changes = list()
    max_feature_names = list()

    for i in range(0, num_cfs):
        cfs_index.append(i+1)
        counterfactual_features = df_counterfactuals.iloc[i].drop('target')
        dif_features = np.abs([sample - counterfactual_features])
        num_feature_changes.append(np.count_nonzero(dif_features))
        nonzero_difs = dif_features[dif_features>0]
        if len(nonzero_difs>0):
            min_feature_changes.append(np.min(nonzero_difs))
            min_feature_pos =  np.where(dif_features == min_feature_changes[i])[2][0]
            min_feature_names.append(feature_names[min_feature_pos])

            max_feature_changes.append(np.max(nonzero_difs))
            max_feature_pos = np.where(dif_features == max_feature_changes[i])[2][0]
            max_feature_names.append(feature_names[max_feature_pos])
    
        else:
            min_feature_changes.append(np.nan)
            min_feature_names.append(np.nan)
            max_feature_changes.append(np.nan)
            max_feature_names.append(np.nan)

    cfs_stats = pd.DataFrame({'counterfactual': cfs_index, 'num_changes': num_feature_changes, 'min_feat_change': min_feature_changes, 
                            'min_feat_name': min_feature_names, 'max_feat_change': max_feature_changes, 'max_feat_name': max_feature_names})
    return(cfs_stats)


def get_counterfactual_changes(sample, counterfactual, feature_names):    
    '''
    This function compares the feature values of a sample and one conterfactual to keep only those features that have changed.
    Arguments
    ---------
    sample: The original test sample
    counterfactual: The values of a counterfactual example
    feature_names: A list with the feature names
    Return
    ---------
    A data frame with one row for each feature that has changed and columns with the following values: feature name, value of
    the feature in the counterfactual, difference between the counterfactual and the original value.
    '''
    sample_values = np.asarray(sample)
    counterfactual_values = np.asarray(counterfactual)
    num_cols = len(counterfactual)
    feat_change_names = []
    feat_change_values = []
    feat_change_dif = []
    for i in range(0, num_cols):
        if sample_values[i] != counterfactual_values[i]:
            feat_change_names.append(feature_names[i])
            feat_change_values.append(counterfactual_values[i])
            feat_change_dif.append(counterfactual_values[i] - sample_values[i])
    if len(feat_change_values)>0:
        df_changes = pd.DataFrame(data={'feature': feat_change_names, 'value': feat_change_values, 'difference': feat_change_dif})
        return df_changes
    return None

def get_summary_changes(df_samples, df_counterfactuals, num_cfs):
    '''
    This function summarises the differences between a sample and a set of counterfactuals.
    Arguments
    ---------
    df_samples: The original test samples as data frame (e.g., as returned by DiCE)
    df_counterfactuals: The data frame of counterfactuals returned by DiCE
    num_cfs: Number of counterfactuals per sample
    Return
    ---------
    A data frame with one row for each feature in the original sample and the following columns: number of times the feature
    has been changed considering all counterfactuals, maximum difference between the original feature value and one counterfactual.
    '''
    num_samples = df_samples.shape[0]
    if 'target' in df_counterfactuals.columns:
        df_counterfactuals.drop('target', axis=1, inplace=True)
    feature_names = list(df_counterfactuals.columns)
    num_feature_changes = np.zeros(shape=len(feature_names))
    feature_max_dif = np.zeros(shape=len(feature_names))
    first_cf = 0
    last_cf = num_cfs
    for i in range(0, num_samples):
        df_sample = df_samples.iloc[i, :]
        df_sample_cfs = df_counterfactuals.iloc[first_cf:last_cf,:]
        for c in range(0, num_cfs):
            df_changes = get_counterfactual_changes(df_sample, df_sample_cfs.iloc[c,:], feature_names)
            if df_changes is not None:
                for j in range(0, df_changes.shape[0]):
                    feature_index = feature_names.index(df_changes['feature'][j])
                    feature_diff = df_changes['difference'][j]
                    num_feature_changes[feature_index] += 1
                    if np.abs(feature_diff) > np.abs(feature_max_dif[feature_index]):
                        feature_max_dif[feature_index] = feature_diff
        first_cf += num_cfs
        last_cf = first_cf + num_cfs
    
    df_summary = pd.DataFrame(data={'feature': feature_names, 'num_changes': num_feature_changes, 'max_difference': feature_max_dif})
    return df_summary