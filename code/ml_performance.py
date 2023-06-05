#######################################################
# File: ml_performance.py
# Goal: Functions to extract FP, FN, TP and TN
# Author: Aurora Ramirez (Univ. of Cordoba, Spain)
#######################################################

import pandas as pd

def get_fp(df_test, y_test, y_pred):
    '''
    This function returns a data frame with the false positives.
    Arguments
    ---------
    df_test: Feature values of samples in the test partition as data frame
    y_test: Target value of samples in the test partition as array
    y_pred: Predictions for samples in the test partition as array
    Return
    ---------
    A data frame with the test samples that were wrongly classified as positive.
    '''
    fp_samples = []
    for i in range(0, len(y_test)):
        if y_test.values[i] == 0 and y_pred[i] == 1:
            fp_samples.append(df_test.iloc[i])
    df_fp = pd.DataFrame(columns=df_test.columns, data=fp_samples)
    return df_fp

def get_fn(df_test, y_test, y_pred):
    '''
    This function returns a data frame with the false negatives.
    Arguments
    ---------
    df_test: Feature values of samples in the test partition as data frame
    y_test: Target value of samples in the test partition as array
    y_pred: Predictions for samples in the test partition as array
    Return
    ---------
    A data frame with the test samples that were wrongly classified as negative.
    '''
    fn_samples = []
    for i in range(0, len(y_test)):
        if y_test.values[i] == 1 and y_pred[i] == 0:
            fn_samples.append(df_test.iloc[i])
    df_fn = pd.DataFrame(columns=df_test.columns, data=fn_samples)
    return df_fn

def get_tp(df_test, y_test, y_pred):
    '''
    This function returns a data frame with the true positives.
    Arguments
    ---------
    df_test: Feature values of samples in the test partition as data frame
    y_test: Target value of samples in the test partition as array
    y_pred: Predictions for samples in the test partition as array
    Return
    ---------
    A data frame with the test samples that were correctly classified as positive.
    '''
    tp_samples = []
    for i in range(0, len(y_test)):
        if y_test.values[i] == 1 and y_pred[i] == 1:
            tp_samples.append(df_test.iloc[i])
    df_tp = pd.DataFrame(columns=df_test.columns, data=tp_samples)
    return df_tp

def get_tn(df_test, y_test, y_pred):
    '''
    This function returns a data frame with the true negatives.
    Arguments
    ---------
    df_test: Feature values of samples in the test partition as data frame
    y_test: Target value of samples in the test partition as array
    y_pred: Predictions for samples in the test partition as array
    Return
    ---------
    A data frame with the test samples that were correctly classified as negative.
    '''
    tn_samples = []
    for i in range(0, len(y_test)):
        if y_test.values[i] == 0 and y_pred[i] == 0:
            tn_samples.append(df_test.iloc[i])
    df_tn = pd.DataFrame(columns=df_test.columns, data=tn_samples)
    return df_tn
