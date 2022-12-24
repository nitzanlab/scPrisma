import numpy as np
import math
from numpy import  random
from numpy import linalg
from scipy import sparse
#from pre_processing import *
from numba import jit

@jit(nopython=True, parallel=True)
def generate_eigenvectors_circulant(n, alpha=0.9999):
    '''
    :param n: number of cells
    :param alpha: correlation between neighbors
    :return: eigen vectors and eigen values of circulant matrix
    '''
    eigen_vectors = []
    for i in range(n):
        v = np.zeros(n)
        for k in range(n):
            v[k] = math.sqrt(2 / n) * math.cos(math.pi * (((2 * (i) * (k)) / n) - 1 / 4))
        eigen_vectors.append(v)
    eigen_values = np.zeros(n)
    for i in range(n):
        for k in range(n):
            if k < n / 2:
                eigen_values[i] += (alpha ** k) * math.cos((2 * math.pi * i * k) / n)
            else:
                eigen_values[i] += (alpha ** (n-k)) * math.cos((2 * math.pi * i * k) / n)
    return np.array(eigen_vectors) , np.array(eigen_values)



def generate_spectral_matrix(n, alpha=0.99):
    '''
    :param n: number of cells
    :param alpha: correlation between neighbors
    :return: spectral matrix, each eigenvector is multiplied by the sqrt of the appropriate eigenvalue
    '''
    return generate_spectral_matrix_tmp(n,alpha)
    #return generate_spectral_matrix_inner(n,np.array(alpha))

@jit(nopython=True, parallel=True)
def generate_spectral_matrix_inner(n, alpha):
    '''
    :param n: number of cells
    :param alpha: correlation between neighbors
    :return: spectral matrix, each eigenvector is multiplied by the sqrt of the appropriate eigenvalue
    '''
    eigen_vectors = np.zeros((n,n))
    k = np.arange(n)
    for i in range(n):
        v = np.sqrt(2 / n) * np.cos(math.pi * (((2 * (i) * (k)) / n) - 1 / 4))
        eigen_vectors[i,:]=v
    eigen_values = np.zeros(n)
    k_1 = k[:int(n/2)]
    k_2 = k[int(n/2):]
    for i in range(n):
        eigen_values[i]=np.sum((alpha** k_1) * np.cos((2 * np.pi * i * k_1) / n))
        eigen_values[i]+=np.sum((alpha **  (n-k_2)) * np.cos((2 * np.pi * i * k_2) / n))
    for i in range(n):
        eigen_vectors[i]*=np.sqrt(eigen_values[i])
    return eigen_vectors

def generate_spectral_matrix_tmp(n, alpha=0.99):
    '''
    :param n: number of cells
    :param alpha: correlation between neighbors
    :return: spectral matrix, each eigenvector is multiplied by the sqrt of the appropriate eigenvalue
    '''
    eigen_vectors = []
    for i in range(n):
        v = np.zeros(n)
        for k in range(n):
            v[k] = math.sqrt(2 / n) * math.cos(math.pi * (((2 * (i) * (k)) / n) - 1 / 4))
        eigen_vectors.append(v)
    eigen_values = np.zeros(n)
    for i in range(n):
        for k in range(n):
            if k < n / 2:
                eigen_values[i] += (alpha ** k) * math.cos((2 * math.pi * i * k) / n)
            else:
                eigen_values[i] += (alpha ** (n-k)) * math.cos((2 * math.pi * i * k) / n)
    for i in range(n):
        eigen_vectors[i]*=np.sqrt(eigen_values[i])
    return np.array(eigen_vectors)

@jit(nopython=True, parallel=True)
def generate_eigenvalues_circulant(n, alpha=1):
    '''
    :param n: number of cells
    :param alpha: correlation between neighbors
    :return: eigenvalues of circulant matrix
    '''
    eigen_values = np.zeros(n)
    for i in range(n):
        for k in range(n):
            if k < n / 2:
                eigen_values[i] += (alpha ** k) * math.cos((2 * math.pi * i * k) / n)
            else:
                eigen_values[i] += (alpha ** (n-k)) * math.cos((2 * math.pi * i * k) / n)
    return eigen_values



@jit(nopython=True, parallel=True)
def get_psuedo_vecs(alpha, ncells):
    theta_lst = get_theta_lst(ncells)
    mat = np.zeros((ncells, ncells))
    for col, theta in enumerate(theta_lst):
        for row in range(1, ncells + 1):
            mat[row-1, col] = math.sin(row * theta) - alpha * math.sin(
                (row-1) * theta)

    return mat

@jit(nopython=True, parallel=True)
def get_pseudo_eigenvalues(alpha, n, m=1):
    theta_lst = [(j * math.pi) / (n + 1) for j in range(1, n+1)]
    vals = [rctp_func_short(theta, alpha) for theta in theta_lst]
    return vals

@jit(nopython=True, parallel=True)
def get_pseudo_eigenvalues_for_loss(n,alpha, m=1):
    theta_lst = [(j * math.pi) / (n + 1) for j in range(1, n+1)]
    vals = [rctp_func_short(theta, alpha) for theta in theta_lst]
    return vals

@jit(nopython=True, parallel=True)
def get_psuedo_data(ngenes, ncells, nchange=1):
    alpha = get_alpha(ngenes, nchange)
    psuedo_vecs = get_psuedo_vecs(alpha, ncells)
    psuedo_vals = get_pseudo_eigenvalues(alpha, ncells, ncells)
    psuedo_vecs = normalize(psuedo_vecs, axis=0, norm='l2')
    return psuedo_vecs, psuedo_vals

@jit(nopython=True, parallel=True)
def rctp_func(theta, alpha, m):
    return 1 + 2 * sum([alpha ** i * math.cos(i * theta)
                        for i in range(1, m + 1)], 0)

@jit(nopython=True, parallel=True)
def rctp_func_short(theta, alpha):
    return (1 - alpha ** 2) / \
           float(1 - 2 * alpha * math.cos(theta) + alpha ** 2)

@jit(nopython=True, parallel=True)
def get_theta_lst(ncells):
    return [(j * math.pi) / (ncells + 1) for j in range(1, ncells + 1)]

@jit(nopython=True, parallel=True)
def get_alpha(ngenes, nchange):
    return math.exp((-2 * nchange) / float(ngenes))







def get_psuedo_vecs(alpha, ncells):
    theta_lst = get_theta_lst(ncells)
    mat = np.zeros((ncells, ncells))
    for col, theta in enumerate(theta_lst):
        for row in range(1, ncells + 1):
            mat[row-1, col] = math.sin(row * theta) - alpha * math.sin(
                (row-1) * theta)

    return mat


def get_pseudo_eigenvalues(alpha, n, m=1):
    theta_lst = [(j * math.pi) / (n + 1) for j in range(1, n+1)]
    vals = [rctp_func_short(theta, alpha) for theta in theta_lst]
    return vals

def get_pseudo_eigenvalues_for_loss(n,alpha, m=1):
    theta_lst = [(j * math.pi) / (n + 1) for j in range(1, n+1)]
    vals = [rctp_func_short(theta, alpha) for theta in theta_lst]
    return vals


def get_psuedo_data(ncells, alpha, normalize_vectors=True):
    psuedo_vecs = get_psuedo_vecs(alpha, ncells)
    psuedo_vals = get_pseudo_eigenvalues(alpha, ncells, ncells)
    psuedo_vecs = normalize(psuedo_vecs, axis=0, norm='l2')
    if not normalize_vectors:
        return psuedo_vecs, psuedo_vals
    else:
        for i in range(len(psuedo_vals)):
            psuedo_vecs[:, i] *= np.sqrt(psuedo_vals[i])
        return psuedo_vecs


def rctp_func(theta, alpha, m):
    return 1 + 2 * sum([alpha ** i * math.cos(i * theta)
                        for i in range(1, m + 1)], 0)


def rctp_func_short(theta, alpha):
    return (1 - alpha ** 2) / \
           float(1 - 2 * alpha * math.cos(theta) + alpha ** 2)


def get_theta_lst(ncells):
    return [(j * math.pi) / (ncells + 1) for j in range(1, ncells + 1)]


def get_alpha_theoretic(ngenes, nchange):
    print(ngenes)
    print(nchange)
    return math.exp((-2 * nchange) / float(ngenes))



def get_numeric_eigen_values(alpha, n):
    values = [alpha ** i for i in range(n, -1, -1)]
    values = values + values[-2::-1]
    offset = list(range(-n, n+1))
    mat = sparse.diags(values, offset, shape=(n, n)).toarray()
    eig_vals, eig_vect = linalg.eig(mat)
    return eig_vect, eig_vals


def get_numeric_eigen_data(ncells, alpha, normalize_vectors=True):
    values = [alpha ** i for i in range(ncells, -1, -1)]
    values = values + values[-2::-1]
    offset = list(range(-ncells, ncells + 1))
    mat = sparse.diags(values, offset, shape=(ncells, ncells)).toarray()
    #mat = mat - mat.mean(axis=1, keepdims=True) #####
    eig_vals, eig_vecs = linalg.eig(mat)
    if not normalize_vectors:
        return eig_vecs, eig_vals
    else:
        for i in range(eig_vals.size):
            eig_vecs[:, i] *= np.sqrt(eig_vals[i])
        return eig_vecs


def get_linear_eig_data(ncells, alpha, method, normalize_vectors):
    if method == 'numeric':
        return get_numeric_eigen_data(ncells, alpha, normalize_vectors)
    elif method == 'pseudo':
        return get_pseudo_data(ncells, alpha, normalize_vectors)



def get_theoretic_eigen(cov, n_values=None):
    if n_values==None:
        n_values= cov.shape[0]
    eig_vals, eig_vecs = linalg.eigh(cov)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:,idx]
    eig_vals = eig_vals[:n_values]
    eig_vecs = eig_vecs[:,:n_values]
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)
    for i in range(eig_vals.size):
            eig_vecs[:, i] *= np.sqrt(eig_vals[i])
    return eig_vecs
