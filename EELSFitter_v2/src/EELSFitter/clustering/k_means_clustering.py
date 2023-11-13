#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math

#TODO: write docstrings and remove unused code

def dist(a, b, exp = 2):
    if b.size>1:
        return np.power(np.sum(np.power(a-b, exp), axis = 1), 1/exp)
    else:
        return np.power(np.power(a-b, exp), 1/exp)

def power(a, b, exp = 2):
    return np.power(a-b, exp)

def log(a, b):
    return np.log(a-b)

def exp(a, b):
    return np.exp(a-b)

def cost_clusters(values, clusters, r, cost_function = dist):
    cost = np.zeros(clusters.shape[0])
    for i in range(clusters.shape[0]):
        cluster = clusters[i]
        ri = r[i]
        cost[i] = np.sum(cost_function(values, cluster)*ri)
    return cost

def relocate_clusters(clusters, values, r):
    n_clusters = len(clusters)
    for i in range(n_clusters):
        clusters[i] = relocate_cluster(values[r[i].astype("bool")])
    return clusters

def relocate_cluster(values):
    #TODO: adjust to accept other costfunctions?
    if values.ndim > 1:
        return np.average(values, axis=0)
    else:
        return np.average(values)
    
def reassign_values(clusters, values, cost_function = dist):
    r = np.zeros((len(clusters), len(values)))
    for i in range(len(values)):
        value = values[i]
        ind = np.argmin(cost_function(clusters, value))
        r[ind, i] = 1        
    return r


def k_means(values, n_clusters=3, n_iterations=30, n_times=5):
    """
    The k means clustering algorithm. Unsupervised clustering into ``n_clusters`` clusters, by minimising
    the distance to the cluster means.

    Parameters
    ----------
    values : numpy.ndarray, shape=(M,)
        Array with values to be clustered.
    n_clusters : int, optional
        The desired number of clusters. The default is 3.
    n_iterations : int, optional
        number of iterations in the clustering algorithm. The default is 30.
    n_times : int, optional
        number of times the clustering algorithm is run. The final outcome that is returned is the one
        with the lowest overall distance. The default is 5.

    Returns
    -------
    min_clusters : numpy.ndarray, shape=(M,)
        values of the cluster means.
    min_r : numpy.ndarray, shape=(M,)
        array that assigns each entry in `values` array to one of the clusters

    """
    n_values = len(values)
    if values.ndim == 1:
        vpc = math.floor(n_values/n_clusters)
        halfvpc = math.floor(vpc/2)
        clusters = np.sort(values)[(np.linspace(0,n_clusters-1,n_clusters, dtype= int)*vpc+halfvpc)].astype("float")
    else:
        ind_cluster = np.floor(np.random.rand(n_clusters)*n_values).astype(int)
        clusters = values[ind_cluster].astype("float")
    
    
    cost_min = np.inf
    old_clusters = np.copy(clusters)
    for j in range(n_times):
        ind_cluster = np.floor(np.random.rand(n_clusters)*n_values).astype(int)
        clusters = values[ind_cluster].astype("float")
        for i in range(n_iterations):
            r = reassign_values(clusters, values)
            r_empty = np.argwhere(np.sum(r, axis = 1)==0)
            while len(r_empty) > 0:
                for r_inx in r_empty:
                    change = np.random.randint(0,len(values))
                    r[:, change] = 0
                    r[r_empty, change] = 1
                r_empty = np.argwhere(np.sum(r, axis = 0)==0)
            clusters = relocate_clusters(clusters, values, r)
            costs = np.sum(cost_clusters(values, clusters, r))
            if cost_min > costs:
                cost_min = costs
                min_clusters = clusters
                min_r = r
            if (old_clusters == clusters).all():
                break
            old_clusters = np.copy(clusters)
    return min_clusters, min_r
