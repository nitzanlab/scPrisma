
from numpy import  random
import math
import numpy as np
import matplotlib.pyplot as plt
import copy



def tree_array(m,p,b):
    '''
    Simulate a tree like gene expression matrix
    :param m: number of mutations per bifurcation
    :param p: number of genes
    :param b: number of bifurcations
    :return: simulated tree like gene expression matrix
    '''
    num_genes=p
    num_mutations = m
    initial_profile = np.random.randint(0, 2, num_genes) * 2 - 1
    genes_mutated = np.random.choice(num_genes, num_mutations, replace=False)
    initial_profile[genes_mutated] = -(initial_profile[genes_mutated])
    cur_profiles_lin = np.copy(initial_profile)

    num_bifurcations = b
    for k in range(num_bifurcations):
        cur_profiles_lin = np.vstack((cur_profiles_lin, cur_profiles_lin))
        for j in range(cur_profiles_lin.shape[0]):
            genes_mutated = np.random.choice(num_genes, num_mutations, replace=False)
            cur_profiles_lin[j, genes_mutated] = -(cur_profiles_lin[j, genes_mutated])
    return cur_profiles_lin


def mutate_sample_all(v, nmut, skip, include_start=False):
    V = []
    ngenes = len(v)

    if include_start:
        V.append(copy.deepcopy(v))
    for k in range(nmut // skip):
        ix = np.random.choice(ngenes, skip)
        v[ix] = (v[ix] + 1) % 2
        V.append(copy.deepcopy(v))
    return V

def simulate_spatial_cyclic(ngenes, ncells, w):
    '''
    Simulated a cyclic signal in a gene expression
    :param ngenes: Number of genes
    :param ncells: Number of cells
    :param w: window length (percentage of cells that every gene is expressed in)
    :return: simulated cyclic signal
    '''
    leaves_arr = np.zeros((ncells, ngenes))
    #leaves_arr*=-1
    for i in range(ngenes):
        start = int(np.floor(random.rand() * ncells))
        if (start + int(np.floor(ncells * w))) >= ncells:
            leaves_arr[start:ncells, i] = 1
            leaves_arr[0:(start + int(np.floor(ncells * w))) % ncells, i] = 1
        else:
            leaves_arr[start:start + int(np.floor(ncells * w)), i] = 1

    return leaves_arr

def simulate_star_all_cells(ngenes, nmut, skip, depth, ix=[1, 2]):
    '''
    Simulated a star like  gene expression matrix
    :param ngenes: number of genes
    :param nmut: number of mutations per bifurcation
    :param skip: number of mutations between nearby cells
    :param depth: number of branches
    :param ix:
    :return: star like  gene expression matrix
    '''
    V = []
    leaves = []
    v0 = np.random.choice([0, 1], ngenes)
    V.append(mutate_sample_all(v0, nmut, skip, include_start=True))  # this changes v0
    leaves.append(V[0][0])

    for d in range(depth):
        new_leaves = []
        for v in leaves:
            v1 = copy.deepcopy(v)
            v2 = copy.deepcopy(v)
            V1 = mutate_sample_all(v1, nmut, skip, include_start=True)
            V2 = mutate_sample_all(v2, nmut, skip, include_start=True)
            V.append(V1)
            V.append(V2)
            new_leaves.append(V1[0])
            new_leaves.append(V2[0])
        leaves = new_leaves
    V = [a for row in V for a in row]
    return np.array(V)

def sample_all_cyclic(v, nmut, skip, include_start=False):
    V = []
    ngenes = len(v)
    if include_start:
        V.append(copy.deepcopy(v))
    for k in range(nmut // skip):
        ix = np.random.choice(ngenes, skip)
        v[ix] = (v[ix] ) *-1
        V.append(copy.deepcopy(v))
    first_cell= V[0]
    last_cell = V[-1]
    diff_list = []
    for i in range(len(V[0])):
        if first_cell[i] != last_cell[i]:
            diff_list.append(i)
    while len(diff_list) // skip !=0:
        ix = np.unique(np.random.choice(diff_list, skip))
        new_cell = copy.deepcopy(V[-1])
        new_cell[ix] = (V[-1][ix]) *-1
        V.append(new_cell)
        for j in ix:
            diff_list.remove(j)
    return V



def simulate_linear(ngenes, ncells, skip=1, noise=0,
                    other_genes=None):
    '''
    :param depth: number of branches
    :param ngenes: number of genes
    :param ncells: number of cells
    :param skip: number of mutations between nearby cells
    :param noise: variance of random noise
    :param other_genes:
    :return:
    '''
    data = np.full((ncells, ngenes), -1)
    data[0] = np.random.choice([-1, 1], ngenes, replace=True)
    for cell_idx in range(1, ncells):
        flip_vector = np.full((ngenes), 1)
        flip_idx = np.random.choice(ngenes, skip, replace=True)
        flip_vector[flip_idx] = -1
        data[cell_idx] = np.multiply(data[cell_idx - 1], flip_vector)
    noise = np.random.normal(0, noise, (ncells, ngenes))
    data = data.astype('float64', casting='safe')
    data += noise
    if other_genes is not None:
        data = np.concatenate((data, other_genes), axis=1)
    cov_est = np.cov(data)
    return data


def simulate_spatial_1d(ngenes, ncells, w,  ix=[1, 2]):
    #leaves_arr = np.zeros((ncells,ngenes))
    leaves_arr = np.zeros((int(ncells + 2*w*ncells),ngenes))
    for i in range(ngenes):
        # start = int(np.floor(random.rand() * (ncells-ncells*w)))
        #leaves_arr[start:start + int(np.floor(ncells*w)),i]= 1
        start = int(np.floor(random.rand() * (ncells +w*ncells)))
        leaves_arr[start:start + int(np.floor(ncells * w)), i] = 1
    return np.array(leaves_arr[int(ncells*w):int(ncells*w +ncells),:])

def simulate_linear_2(ngenes, nmut, skip):
    v0 = np.random.choice([0, 1], ngenes)
    V = mutate_sample_all(v0, nmut, skip)
    return np.array(V)

def simulate_square_cyclic(groups_num=4, ngenes_group=500, ngenes_cyclic=300 , nmut_group=8,nbif=8 , w=0.1):
    for i in range(groups_num):
        if i==0:
            A = tree_array(m=nmut_group,p=ngenes_group,b=nbif)
            np.random.shuffle(A)
        else:
            B = tree_array(m=nmut_group, p=ngenes_group, b=nbif)
            np.random.shuffle(B)
            A= np.concatenate([A,B], axis=0)

    C = simulate_spatial_cyclic(ngenes=ngenes_cyclic,ncells=A.shape[0],w=w)
    A = np.concatenate([A,C], axis=1)
    return A


def simulate_window_linear(ngenes, ncells, w):
    leaves_arr = np.zeros((ncells, ngenes))
    window_size = int(np.floor(ncells * w))
    for i in range(ngenes):
        start = int(np.floor(random.rand() * (ncells + window_size))) - window_size
        leaves_arr[max(start, 0):min(start+window_size, ncells), i] = 1
    return leaves_arr

def simulate_window_linear_2(ngenes, ncells, w):
    leaves_arr = np.zeros((ncells, ngenes))
    window_size = int(np.floor(ncells * w))
    for i in range(ngenes):
        start = int(np.floor(random.rand() * (ncells + window_size))) - window_size
        leaves_arr[max(start, 0):min(start+window_size, ncells), i] = 1
    return leaves_arr