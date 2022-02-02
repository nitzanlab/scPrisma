from numpy import  random
import math
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy as sp
import scipy.optimize
from sklearn.decomposition import PCA
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
import copy
from sklearn.feature_selection import VarianceThreshold

from spectrum_gen import *


def cell_normalization(V):
    V = normalize(V, axis=1, norm='l2')
    return V


def gene_normalization(V):
    V = normalize(V, axis=0, norm='l2')
    return V

def loss_alpha(alpha, *args):
    e2 = args[0]
    eigen_values = generate_eigenvalues_circulant(len(e2),alpha)
    eigen_values = np.sort(abs(eigen_values))[::-1]
    loss = np.linalg.norm(eigen_values-e2)
    return loss


def loss_alpha_linear(alpha, *args):
    e2 = args[0]
    eigen_values = get_pseudo_eigenvalues(alpha, len(e2), len(e2))
    eigen_values = np.array(eigen_values)
    eigen_values = np.sort(abs(eigen_values))[::-1]
    loss = np.linalg.norm(eigen_values-e2)
    return loss


def loss_alpha_p(alpha, *args):
    e2 = args[0]
    p= args[1]
    eigen_values = generate_eigenvalues_circulant(len(e2),alpha)
    eigen_values = np.sort(abs(eigen_values))[::-1]
    eigen_values = eigen_values[:p]
    e2 = e2[:p]
    loss = np.linalg.norm(eigen_values-e2)
    return loss

def get_alpha(ngenes=None, nchange=1, optimized_alpha=False,
              eigenvals=None):
    if optimized_alpha:
        alpha = optimize_alpha(eigenvals,loss_alpha_linear)
    else:
        alpha = get_alpha_theoretic(ngenes, nchange)
    return alpha

def optimize_alpha(e2, loss_alpha_func=loss_alpha):
    e2 = e2[e2 > 0.0000001]
    #starting_loss = loss_alpha_func(1,(e2))
    #print("starting loss: " + str(starting_loss))
    res = scipy.optimize.minimize_scalar(loss_alpha, bounds=(0, 0.99999999999), args = (e2) , method='bounded' )
    alpha = res.x
    #print(alpha)
    eigen_values = generate_eigenvalues_circulant(len(e2),alpha)
    eigen_values = np.sort(abs(eigen_values))[::-1]
    loss = np.linalg.norm(eigen_values-e2)
    #print("loss: " + str(loss))
    # plt.plot(r2, e2, marker)
    #print("entering loglog")
    #plt.loglog(r2, e2, color='green', label='Simulated eigenvalues')
    #plt.loglog(r2, eigen_values, color='blue', label='Theoretic eigenvalues')
    #plt.grid()
    #plt.ylabel("eigenvalue")
    #plt.xlabel("rank")
    #plt.legend()

    #plt.show()
    return alpha

def optimize_alpha_p(e2,p):
    e2 = e2[e2 > 0.000000000001]
    starting_loss = loss_alpha(1,(e2))
    #print("starting loss: " + str(starting_loss))
    res = scipy.optimize.minimize_scalar(loss_alpha_p, bounds=(0, 0.99999999999), args = (e2,p) , method='bounded' )
    alpha = res.x
    #print(alpha)
    eigen_values = generate_eigenvalues_circulant(len(e2),alpha)
    eigen_values = np.sort(abs(eigen_values))[::-1]
    loss = np.linalg.norm(eigen_values-e2)
    #print("loss: " + str(loss))
    r2 = np.array(range(len(e2))) + 1
    # plt.plot(r2, e2, marker)
    #print("entering loglog")
    #plt.loglog(r2, e2, color='green', label='Simulated eigenvalues')
    #plt.loglog(r2, eigen_values, color='blue', label='Theoretic eigenvalues')
    #plt.grid()
    #plt.title("Spectrum with p="+str(p))
    #plt.ylabel("eigenvalue")
    #plt.xlabel("rank")
    #plt.legend()

    plt.show()
    return alpha


def normalize_data_log(V):
    V = V.astype(np.float)
    V = V[~np.all(V == 0, axis=1)]
    V=V.T
    V = V[~np.all(V == 0, axis=1)]
    V=V.T
    V = np.round(np.log2(100000 * np.divide(V, np.sum(V, axis=0)) + 1.0000001), 2)
    return V


def identify_highly_variable_genes(expression, low_x=0.001, high_x=np.inf, low_y=10**-15, do_plot=True):
    """Identify the highly variable genes (follows the Seurat function).
    expression -- the DGE with cells as columns
    low_x      -- threshold for low cutoff of mean expression
    high_x     -- threshold for high cutoff of mean expression
    low_y      -- threshold for low cutoff of scaled dispersion."""

    mean_val = np.log(np.mean(np.exp(expression)-1, axis=1) + 1)
    gene_dispersion = np.log(np.var(np.exp(expression) - 1, axis=1) / np.mean(np.exp(expression) - 1))
    bins = np.arange(1, np.ceil(max(mean_val)), step=0.5)
    binned_data = np.digitize(mean_val, bins)
    # This should be written more efficiently
    gd_mean = np.array([])
    gd_std = np.array([])
    for bin in np.unique(binned_data):
        gd_mean = np.append(gd_mean, np.mean(gene_dispersion[binned_data == bin]))
        gd_std = np.append(gd_std, np.std(gene_dispersion[binned_data == bin]))
    gene_dispersion_scaled = (gene_dispersion - gd_mean[binned_data]) / (gd_std[binned_data] + 10e-8)
    genes = np.intersect1d(np.where(mean_val >= low_x), np.where(mean_val <= high_x))
    genes = np.intersect1d(genes, np.where(gene_dispersion_scaled >= low_y))

    if do_plot:
        col_genes = np.zeros(len(expression))
        col_genes[genes] = 1
        plt.figure()
        plt.scatter(mean_val, gene_dispersion_scaled, s=2, c=col_genes)
        plt.show()

    return genes

def highly_v(data , threshold_n=0.01):
    data = data.T
    y = np.var(data, axis=1)
    data = data[np.where(y >= threshold_n)]
    return data.T
