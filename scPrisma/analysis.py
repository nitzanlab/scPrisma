import matplotlib.pyplot as plt
import numpy as np

from pre_processing import *
from algorithms import *
from spectrum_gen import *
from data_gen import *
from datasets import *
import copy
from scipy import stats
from sklearn.metrics import f1_score
import scanpy as sc
from scipy.signal import savgol_filter
from visualizations import *

def get_perm(n):
    perm = np.random.permutation(n)
    E = np.zeros((n,n))
    for i in range(n):
        E[i,perm[i]]=1
    return E , perm


def Perm_to_range(E):
    '''
    :param E: Permutation matrix
    :return: Permutation list
    '''
    order =[]
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if E[i,j]==1:
                order.append(j)
    return np.array(order)

def spearm(E,order):
    n=len(order)
    res=0
    e_range = E_to_range(E)
    for i in range(n):
        tmp = (stats.spearmanr(a=e_range, b=(order+i)%n))[0]
        if tmp>res:
            res=tmp
    order = order[::-1]
    for i in range(n):
        tmp = (stats.spearmanr(a=e_range, b=(order+i)%n))[0]
        if tmp>res:
            res=tmp
    return res

def analyze_corr_vs_noise():
    corr_a = np.load("corr_a.npy")
    noise_a = np.load("noise_c.npy")
    plt.plot(noise_a,corr_a)
    plt.xlabel("Noise variance")
    plt.ylabel("")
    plt.title("Spearman's Rank-Order Correlation as a function of Gaussian noise")
    plt.show()

def E_to_range(E):
    order =[]
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if E[i,j]==1:
                order.append(j)
    return np.array(order)



def calculate_avg_groups_layer(adata):
    avg_groups = np.zeros((8,adata.X.shape[1]))
    for i in range(8):
        tmp_adata = adata[adata.obs["layer"] == str(i)]
        for j in range(tmp_adata.X.shape[0]):
            avg_groups[j, :] += tmp_adata[j, :].X[0, :]
        avg_groups[j,:]/=tmp_adata.X.shape[0]
    return avg_groups


def analyze_diag(path):
    D = np.load(path)
    plot_diag(D)
    pass

def ga_to_labels(groups):
    for i ,group in enumerate(groups):
        if i==0:
            a = np.zeros(group)
        else:
            b = np.ones(group)*i
            a = np.concatenate((a, b), axis=None)
    return a

def f1_score_multi_class(E,groups_amount,labels):
    n=E.shape[0]
    res=0
    pred_labels = ga_to_labels(groups_amount)
    E_ranged = E_to_range(E)
    for i in range(n):
        y_pred = E_to_class(E_ranged,np.roll(pred_labels,i))
        tmp = f1_score(labels, y_pred, average='macro')
        if tmp>res:
            res=tmp
    pred_labels = pred_labels[::-1]
    for i in range(n):
        y_pred = E_to_class(E_ranged,np.roll(pred_labels,i))
        tmp = f1_score(labels, y_pred, average='macro')
        if tmp>res:
            res=tmp
    return res

def E_to_class(E_ranged,pred_labels):
    y_pred = np.zeros(len(E_ranged))
    for i in range(len(E_ranged)):
        y_pred[E_ranged[i]]=pred_labels[i]
    return y_pred


def p_of_genes(adata, genes_list):
    sum_genes = 0
    sum_all = 0
    for gene in genes_list:
        try:

            tmp_sum = np.sum(adata[:, gene].X)
            sum_genes += tmp_sum
        except:
            a = 1
    return sum_genes / np.sum(adata.X)

def plot_cell_cycle_by_phase(adata_filtered,adata_unfiltered):
    cyclic_by_phase = pd.read_csv("data/cyclic_by_phase.csv")
    df = cyclic_by_phase["G1.S"]
    G1S, G1S_F = score_list_of_genes(cyclic_by_phase=cyclic_by_phase, phase="G1.S", filtered=adata_filtered, unfiltered=adata_unfiltered)
    S, S_F = score_list_of_genes(cyclic_by_phase=cyclic_by_phase, phase="S", filtered=adata_filtered, unfiltered=adata_unfiltered)
    G2, G2_F = score_list_of_genes(cyclic_by_phase=cyclic_by_phase, phase="G2", filtered=adata_filtered, unfiltered=adata_unfiltered)
    G2M, G2M_F = score_list_of_genes(cyclic_by_phase=cyclic_by_phase, phase="G2.M", filtered=adata_filtered, unfiltered=adata_unfiltered)
    MG1, MG1_F = score_list_of_genes(cyclic_by_phase=cyclic_by_phase, phase="M.G1", filtered=adata_filtered, unfiltered=adata_unfiltered)
    ranged_pca_2d((adata_filtered.X), G1S_F / G1S_F.max(), title="G1S PCA filtered")
    ranged_pca_2d((adata_filtered.X), S_F / S_F.max(), title="S PCA filtered")
    ranged_pca_2d((adata_filtered.X), G2_F / G2_F.max(), title="G2 PCA filtered")
    ranged_pca_2d((adata_filtered.X), G2M_F / G2M_F.max(), title="G2M PCA filtered")
    ranged_pca_2d((adata_filtered.X), MG1_F / MG1_F.max(), title="MG1 PCA filtered")
    theta = (np.array(range(len(S))) * 2 * np.pi) / len(S)
    # theta = 2 * np.pi * range(len(S))
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(theta, G1S_F / (G1S_F.max()))
    ax.set_rmax(2)
    ax.set_rticks([0.5, 1])  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)

    ax.set_title("Normalized sum of genes related different phases", va='bottom')

    ax.plot(theta, S_F / (S_F.max()))
    ax.plot(theta, G2_F / (G2_F.max()))
    ax.plot(theta, G2M_F / (G2M_F.max()))
    ax.plot(theta, MG1_F / (MG1_F.max()))
    ax.legend(["G1.S", "S", "G2", "G2.M", "M.G1"])
    plt.show()

def calculate_avg_groups(adata,num_groups , groups_length):
    av_groups = np.zeros((num_groups,adata.X.shape[1]))
    for i in range(groups_length):
        for j in range(num_groups):
            av_groups[j,:] += adata[i +j*groups_length,:].X[0,:]
    for j in range(num_groups):
        av_groups[j, :] /=groups_length
    return av_groups



def read_list_of_genes():
    phases = ["G1.S","G2","M.G1","G2.M","S"]
    list_of_genes = []
    cyclic_by_phase = pd.read_csv("data/cyclic_by_phase.csv")
    for phase in phases:
        df = cyclic_by_phase[phase]
        list_a = df.values.tolist()
        for a in list_a:
            list_of_genes.append(a)
    return list_of_genes

def hela_gene_inference(adata, number_of_genes):
    no_cyclic_genes = copy.deepcopy(adata)
    only_cyclic_genes = copy.deepcopy(adata)
    list_of_genes = read_list_of_genes()
    list_of_genes = [x for x in list_of_genes if x in adata.var_names]
    list_of_genes = list(dict.fromkeys(list_of_genes))  # remove duplications
    list_of_non_cyclic_genes = []
    for i in adata.var_names:
        if i not in list_of_genes:
            list_of_non_cyclic_genes.append(i)
    only_cyclic_genes = only_cyclic_genes[:,list_of_genes]
    no_cyclic_genes = no_cyclic_genes[:,list_of_non_cyclic_genes]
    #print(no_cyclic_genes.shape)
    #print(only_cyclic_genes.shape)
    no_cyclic_genes  =(no_cyclic_genes.copy()).T
    only_cyclic_genes = (only_cyclic_genes.copy()).T
    tic = time.time()
    tic = int(tic)
    sc.pp.subsample(no_cyclic_genes, n_obs=number_of_genes , random_state=tic)
    sc.pp.subsample(only_cyclic_genes, n_obs=number_of_genes, random_state=tic)
    #print(no_cyclic_genes.shape)
    #print(only_cyclic_genes.shape)
    adata_classification = only_cyclic_genes.concatenate(no_cyclic_genes)
    adata_classification = adata_classification.T
    #print(adata_classification.shape)
    y_true = np.zeros(number_of_genes*2)
    y_true[number_of_genes:] = np.ones(number_of_genes)
    D = filter_cyclic_genes_line(adata_classification.X, regu=0, iterNum=15)
    #plot_diag(D)
    res = np.diagonal(D)
    #print(" AUC-ROC: " + str(calculate_roc_auc(res, y_true)))
    return calculate_roc_auc(res, y_true)

