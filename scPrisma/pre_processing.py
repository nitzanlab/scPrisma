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
    '''
    Rows L2 normalizations
    :param V: gene expression matrix
    :return: normalized gene expression matrix
    '''
    V = normalize(V, axis=1, norm='l2')
    return V


def gene_normalization(V):
    '''
    Columns L2 normalizations
    :param V: gene expression matrix
    :return: normalized gene expression matrix
    '''
    V = normalize(V, axis=0, norm='l2')
    return V

def loss_alpha(alpha, *args):
    '''
    loss function for optimizing the alpha parameter (cyclic signal)
    :param alpha: current parameter for spectral calculation
    :param args: current spectrum
    :return: loss
    '''
    e2 = args[0]
    eigen_values = generate_eigenvalues_circulant(len(e2),alpha)
    eigen_values = np.sort(abs(eigen_values))[::-1]
    loss = np.linalg.norm(eigen_values-e2)
    return loss


def loss_alpha_linear(alpha, *args):
    '''
    loss function for optimizing the alpha parameter (linear signal)
    :param alpha: current parameter for spectral calculation
    :param args: current spectrum
    :return: loss
    '''
    e2 = args[0]
    eigen_values = get_pseudo_eigenvalues(alpha, len(e2), len(e2))
    eigen_values = np.array(eigen_values)
    eigen_values = np.sort(abs(eigen_values))[::-1]
    loss = np.linalg.norm(eigen_values-e2)
    return loss


def loss_alpha_p(alpha, *args):
    '''
    :param alpha:
    :param args:
    :return:
    '''
    e2 = args[0]
    p= args[1]
    eigen_values = generate_eigenvalues_circulant(len(e2),alpha)
    eigen_values = np.sort(abs(eigen_values))[::-1]
    eigen_values = eigen_values[:p]
    e2 = e2[:p]
    loss = np.linalg.norm(eigen_values-e2)
    return loss

def get_alpha(ngenes=None, nchange=1, optimized_alpha=True,
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
    r2 = np.array(range(len(e2))) + 1

    plt.show()
    return alpha



