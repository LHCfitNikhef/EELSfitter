#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from copy import copy
import scipy

def ewd(x, y, nbins):  
    """Apply Equal Width Discretization (EWD) to the training data to determine variances"""
    
    df_train = np.array(np.c_[x,y])
    xdata = np.array(copy(df_train[:,0]))
    xdata = np.squeeze(xdata)
    df_train = df_train[np.argsort(df_train[:,0])]
    cuts1, cuts2 = pd.cut(xdata, nbins, retbins=True)
    
    return df_train, cuts1, cuts2

def binned_statistics(x,y, nbins):
    """Find the mean, variance and number of counts within the bins described by ewd"""
    
    def CI_high(data, confidence=0.68):
        ## remove the lowest and highest 16% of the values
        
        a = 1.0 * np.array(data)
        n = len(a)
        b = np.sort(data)

        highest = np.int((1-(1-confidence)/2)*len(a))
        high_a = b[highest]
 
        return high_a
    
    def CI_low(data, confidence=0.68):
        ## remove the lowest and highest 16% of the values
        
        a = 1.0 * np.array(data)
        n = len(a)
        b = np.sort(data)
        lowest = np.int(((1-confidence)/2)*len(a))
        low_a = b[lowest]

        return low_a
    
    def get_mean(data):
        return np.mean(data)
    
    df_train, cuts1, cuts2 = ewd(x,y, nbins)
    mean, edges, binnum = scipy.stats.binned_statistic(df_train[:,0], df_train[:,1], statistic='mean', bins=cuts2)
    var, edges, binnum = scipy.stats.binned_statistic(df_train[:,0], df_train[:,1], statistic='std', bins=cuts2)
    count, edges, binnum = scipy.stats.binned_statistic(df_train[:,0], df_train[:,1], statistic='count', bins=cuts2)
    low, edges, binnum = scipy.stats.binned_statistic(df_train[:,0], df_train[:,1], statistic=CI_low, bins=cuts2)
    high, edges, binnum = scipy.stats.binned_statistic(df_train[:,0], df_train[:,1], statistic=CI_high, bins=cuts2)
    mean2, edges, binnum = scipy.stats.binned_statistic(df_train[:,0], df_train[:,1], statistic=get_mean, bins=cuts2)
    
    
    
    return mean, var, count, low, high, mean2

def get_median(x,y,nbins):
    df_train, cuts1, cuts2 = ewd(x,y, nbins)
    median, edges, binnum = scipy.stats.binned_statistic(df_train[:,0], df_train[:,1], statistic='median', bins=cuts2)
    return median

def vectorize_variance(x,y, nbins):
    """Apply the binned variances to the original training data"""
    
    df_train, cuts1, cuts2 = ewd(x,y, nbins)
    mean, std, count = binned_statistics(x,y, nbins)
    variance=[]
    m=0
    i=0
    while i<len(count):
        maximum = count[i]

        while m < maximum:
            variance.append(std[i])
            m+=1
        else:
            m=0
            i+=1
    return np.array(variance)

def vectorize_mean(x,y, nbins):
    
    df_train, cuts1, cuts2 = ewd(x,y, nbins)
    mean, std, count = binned_statistics(x,y,nbins)
    means=[]
    m=0
    i=0
    while i<len(count):
        maximum = count[i]

        while m < maximum:
            means.append(mean[i])
            m+=1
        else:
            m=0
            i+=1
    return np.array(means)

def get_mean_pseudodata(x,y, nbins):
    df_train, cuts1, cuts2 = ewd(x, y, nbins)
    mean, std, count = binned_statistics(x,y,nbins)
    meanvector = vectorize_mean(x,y,nbins)
    stdvector = vectorize_variance(x,y,nbins)
    return mean, std, meanvector, stdvector


   
    
"""Neural network functions: """
    
def custom_cost(y_true, y_pred):
    '''Chi square function'''
    return tf.reduce_mean(tf.square((y_true-y_pred)/sigma))

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def bootstrap():
    df_train_a, df_train_b = train_test_split(df_train, test_size=0.5)
    df_train_1, df_train_2 = train_test_split(df_train_a, test_size=0.5)
    df_train_3, df_train_4 = train_test_split(df_train_b, test_size=0.5)
    
    return df_train_1, df_train_2, df_train_3, df_train_4
    
    
def smooth(x,window_len=10,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    """
    
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]

    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    index = int(window_len/2)
    return y[(index-1):-(index)]

def gaussian(x, amp, cen, std):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    y = (amp) * np.exp(-(x-cen)**2 / (2*std**2))
    return y

    
def window(x,y, minval, maxval):
    """Function applies a window to arrow"""
    
    low = next(i for i, val in enumerate(x) if val > minval)
    treshold_min = str(low)
    treshold_min = int(treshold_min)
    up = next(i for i, val in enumerate(x) if val > maxval)
    treshold_max = str(up)
    treshold_max = int(treshold_max)
    x = x[treshold_min:treshold_max]
    y = y[treshold_min:treshold_max]
    
    return x,y
    

def residuals(prediction, y, std):
    res = np.divide((prediction - y), std)
    return res






