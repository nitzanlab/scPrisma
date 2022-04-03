import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.signal
from Bio.Affy import CelFile
from analysis import *
from algorithms import *
from pre_processing import *
import pandas as pd
import scanpy as sc
from analysis import *
from evaluation import *
from scipy.stats import pearsonr
from sklearn.metrics import calinski_harabasz_score , davies_bouldin_score , silhouette_score
from scipy.signal import savgol_filter

def calculate_avg_groups_layer(adata):
    avg_groups = np.zeros((8,adata.X.shape[1]))
    for i in range(8):
        tmp_adata = adata[adata.obs["layer"] == i]
        for j in range(tmp_adata.X.shape[0]):
            avg_groups[i, :] += tmp_adata[j, :].X[0, :]
        avg_groups[i,:]/=tmp_adata.X.shape[0]
    return avg_groups


def read_cr_single_file(path,ZT="0" , n_obs=300):
    adata = sc.read_csv(path, delimiter='\t').T
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    sc.pp.subsample(adata,n_obs=n_obs)
    adata.obs['ZT'] = ZT
    return adata

def read_cr_single_file_layer(path,layer_path,ZT="0" , n_obs=300):
    adata = sc.read_csv(path, delimiter='\t').T
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.obs['ZT'] = ZT
    layers = cr_layer_read(layer_path)
    adata.obs['layer']=layers
    sc.pp.subsample(adata,n_obs=n_obs , random_state=123)
    return adata

def read_cr_single_file_layer_full(path,layer_path,ZT="0"):
    adata = sc.read_csv(path, delimiter='\t').T
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.obs['ZT'] = ZT
    layers = cr_layer_read(layer_path)
    adata.obs['layer']=layers
    return adata

def calculate_avg_groups(adata,num_groups , groups_length):
    av_groups = np.zeros((num_groups,adata.X.shape[1]))
    for i in range(groups_length):
        for j in range(num_groups):
            av_groups[j,:] += adata[i +j*groups_length,:].X[0,:]
    for j in range(num_groups):
        av_groups[j, :] /=groups_length
    return av_groups

def calculate_avg_groups_crit(adata,crit_list=[],criter=0):
    av_groups = np.zeros((len(crit_list),adata.X.shape[1]))
    for i,cluster in enumerate(crit_list):
        adata_tmp = adata[adata.obs[criter].isin([cluster])]
        for j in range(adata_tmp.X.shape[0]):
            av_groups[i,:] += adata_tmp[j,:].X[0,:]
        av_groups[i, :] /=(adata_tmp.X.shape[0])
    return av_groups




def cr_layer_read(path):
    df = pd.read_csv(path, delimiter=',' , header=None)
    position_matrx = df.to_numpy()
    layers_array = np.zeros(position_matrx.shape[0])
    for i in range(position_matrx.shape[0]):
        layers_array[i]=position_matrx[i,:].argmax()
    return layers_array


def e_to_range(E):
    order =[]
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if E[i,j]==1:
                order.append(j)
    return np.array(order)

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




def score_list_of_genes(cyclic_by_phase,phase , filtered,unfiltered):
    '''
    :param cyclic_by_phase: Pandas dataframe of labeled genes
    :param phase: The phase we want to analyze
    :param filtered: AnnData of filtered gene expression
    :param unfiltered: AnnData of unfiltered gene expression
    :return: sum - numpy array of normalized sum of unfiltered expression of genes related to the phase, sum_f - noprmalized filtered sum
    '''
    df = cyclic_by_phase[phase]
    list_of_genes=[]
    list_a = df.values.tolist()
    for a in list_a:
        list_of_genes.append(a)
    sum = filtered[:,0].X
    sum = np.array(sum)
    sum = sum*0
    sum_f = copy.deepcopy(sum)
    for i in list_of_genes:
        try:
            gene_ex_filtered = filtered[:,i].X
            gene_ex_filtered = np.array(gene_ex_filtered)
            sum_f+=gene_ex_filtered
            gene_ex_unfiltered = unfiltered[:,i].X
            gene_ex_unfiltered = np.array(gene_ex_unfiltered)
            sum+=gene_ex_unfiltered
        except:
            print("Gene does not exist")
    return sum , sum_f

def shuffle_adata(adata):
    '''
    Shuffle the rows(obs/cells) of adata
    :param adata: adata
    :return: shuffled adata
    '''
    perm = np.random.permutation(range(adata.X.shape[0]))
    return adata[perm,:]


def score_list_of_genes_single_adata(cyclic_by_phase,phase , adata):
    '''
    :param cyclic_by_phase: Pandas dataframe of labeled genes
    :param phase: The phase we want to analyze
    :param adata: AnnData of  gene expression
    :return: sum - numpy array of normalized sum of unfiltered expression of genes related to the phase, sum_f - noprmalized filtered sum
    '''
    df = cyclic_by_phase[phase]
    list_of_genes=[]
    list_a = df.values.tolist()
    for a in list_a:
        list_of_genes.append(a)
    sum = adata[:,0].X
    sum = np.array(sum)
    sum = sum*0
    for i in list_of_genes:
        try:
            gene_ex = adata[:,i].X
            gene_ex = np.array(gene_ex)
            sum+=gene_ex
        except:
            #print("Gene does not exist")
            continue
    return sum

def score_list_of_genes_single_adata_2(cyclic_by_phase,phase , adata):
    '''
    :param cyclic_by_phase: Pandas dataframe of labeled genes
    :param phase: The phase we want to analyze
    :param adata: AnnData of  gene expression
    :return: sum - numpy array of normalized sum of unfiltered expression of genes related to the phase, sum_f - noprmalized filtered sum
    '''
    df = cyclic_by_phase[phase]
    list_of_genes=[]
    list_a = df.values.tolist()
    for a in list_a:
        list_of_genes.append(a)
    sum = adata[:,0].X
    sum = np.array(sum)
    sum = sum*0
    for i in list_of_genes:
        try:
            gene_ex = adata[:,i].X
            gene_ex = np.array(gene_ex)
            sum+=gene_ex/gene_ex.max()
        except:
            #print("Gene does not exist")
            continue
    return sum

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



def draw_gene_ct(adata,gene,n_obs,title):
    i=gene
    fig, ax = plt.subplots()
    ax.plot(range(adata.X.shape[0]), adata[:, i].X, label=i)
    ax.plot(range(adata.X.shape[0]), savgol_filter(adata[:, i].X[:, 0], 25, 3), label=("Smoothed " + i))
    ax.legend()  # [i,("Smoothed " +i)]
    ax.set_title(i + " gene expression as a function of cell order- " + str(title))
    ax.set_xlabel("cell location at gene expression matrix")
    ax.set_ylabel("gene expression")
    sc.pl.pca_scatter(adata, color=i, title=("PCA of " + title + " painted by " + i))
    pass

def all_plots_liver(adata,title):
    avg_groups = calculate_avg_groups_layer(adata)
    visualize_distances(avg_groups,title="Distance between layers- " +  title)
    print("Starting norm: " + str(np.linalg.norm(adata.X)))
    values = [genes_score(adata, "cr/r_genes.csv"),genes_score(adata, "cr/z_genes.csv"),genes_score(adata, "cr/f_genes.csv")]
    #labels= ["Rhythmic genes","Zonation genes","Flat genes"]
    #plt.pie(values, labels=labels, autopct=make_autopct(values) , shadow=True)
    #plt.title("Sum of expression of genes by label- " + title)
    #plt.show()
    sc.tl.pca(adata)
    sc.pl.pca_scatter(adata, color='layer' , title=("PCA of " + title + " painted by layer"))
    sc.pl.pca_scatter(adata, color='ZT' , title=("PCA of " + title + " painted by ZT"))
    gene_list_r = [ 'clock', 'npas2', 'nr1d1', 'nr1d2', 'per1', 'per2', 'cry1', 'cry2', 'dbp', 'tef', 'hlf', 'elovl3', 'rora' ,'rorc']
    #gene_list_r = ['cry1','cry2','clock','n1d1','per1','per2','per3', 'rn18s']
    r_adata = sort_data_crit(adata=copy.deepcopy(adata.copy()),crit='ZT',crit_list=['0','6','12','18'])
    print("rhytmic genes")
    for i in gene_list_r:
        try:
            plt.plot(range(r_adata.X.shape[0]),r_adata[:,i].X , label=i)
            plt.plot(range(r_adata.X.shape[0]),savgol_filter(r_adata[:,i].X[:,0],25,3) , label=("Smoothed " +i))
            plt.legend()#[i,("Smoothed " +i)]
            plt.title(i +" expression as a function of cells ordered in cycle- "+str(title))
            plt.xlabel("cell location at gene expression matrix")
            plt.ylabel("gene expression")
            plt.show()
            sc.pl.pca_scatter(r_adata, color=i, title=("PCA of " + title + " painted by " + i))
        except:
            print("not found: "+str(i))
    print("zonation genes")
    gene_list_z = ['glul', 'ass1', 'asl', 'cyp2f2', 'cyp1a2', 'pck1', 'cyp2e1', 'cdh2', 'cdh1', 'cyp7a1', 'acly', 'alb', 'oat', 'aldob', 'cps1']
    for i in gene_list_z:
        try:
            plt.plot(range(r_adata.X.shape[0]),r_adata[:,i].X , label=i)
            plt.plot(range(r_adata.X.shape[0]),savgol_filter(r_adata[:,i].X[:,0],25,3) , label=("Smoothed " +i))
            plt.legend()#[i,("Smoothed " +i)]
            plt.title(i +" expression as a function of cells ordered in cycle- "+str(title))
            plt.xlabel("cell location at gene expression matrix")
            plt.ylabel("gene expression")
            plt.show()
            sc.pl.pca_scatter(r_adata, color=i, title=("PCA of " + title + " painted by " + i))
        except:
            print("not found: "+str(i))
    linear_adata =sort_data_linear(adata.copy())
    #print(linear_adata.obs['ZT'])
    for i in gene_list_r:
        try:
            plt.plot(range(adata.X.shape[0]),linear_adata[:,i].X , label=i)
            plt.plot(range(adata.X.shape[0]),savgol_filter(linear_adata[:,i].X[:,0],25,3) , label=("Smoothed " +i))
            plt.legend()#[i,("Smoothed " +i)]
            plt.title(i +" expression as a function of cells ordered by layer- "+str(title))
            plt.xlabel("cell location at gene expression matrix")
            plt.ylabel("gene expression")
            plt.show()
            sc.pl.pca_scatter(linear_adata, color=i, title=("PCA of " + title + " painted by " + i))
        except:
            print("not found: "+str(i))
    print("zonation genes")
    gene_list_z = ['glul', 'ass1', 'asl', 'cyp2f2', 'cyp1a2', 'pck1', 'cyp2e1', 'cdh2', 'cdh1', 'cyp7a1', 'acly', 'alb', 'oat', 'aldob', 'cps1']
    for i in gene_list_z:
        try:
            plt.plot(range(adata.X.shape[0]),linear_adata[:,i].X , label=i)
            plt.plot(range(adata.X.shape[0]),savgol_filter(linear_adata[:,i].X[:,0],25,3) , label=("Smoothed " +i))
            plt.legend()#[i,("Smoothed " +i)]
            plt.title(i +" expression as a function of cells ordered by layer- "+str(title))
            plt.xlabel("cell location at gene expression matrix")
            plt.ylabel("gene expression")
            plt.show()
            sc.pl.pca_scatter(linear_adata, color=i, title=("PCA of " + title + " painted by " + i))
        except:
            print("not found: "+str(i))

    #print("davies_bouldin_score: "+str(davies_bouldin_score(adata.X,labels)))
    #print("calinski_harabasz_score: "+str(calinski_harabasz_score(adata.X,labels)))
    #print("calinski_harabasz_score: "+str(silhouette_score(adata.X,labels)))

    pass

def read_liver_data(n_obs=250):
    adata = read_cr_single_file_layer("cr/GSM4308343_UMI_tab_ZT00A.txt", layer_path="cr/ZT00A_reco.txt", ZT="0",
                                      n_obs=n_obs)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata2 = read_cr_single_file_layer("cr/GSM4308346_UMI_tab_ZT06A.txt", layer_path="cr/ZT06A_reco.txt", ZT="6",
                                      n_obs=n_obs)
    adata2.obs_names_make_unique()
    adata2.var_names_make_unique()
    adata4 = read_cr_single_file_layer("cr/GSM4308348_UMI_tab_ZT12A.txt", layer_path="cr/ZT12A_reco.txt", ZT="12",
                                      n_obs=n_obs)
    adata4.var_names_make_unique()
    adata4.obs_names_make_unique()

    adata6 = read_cr_single_file_layer("cr/GSM4308351_UMI_tab_ZT18A.txt", layer_path="cr/ZT18A_reco.txt", ZT="18",
                                      n_obs=n_obs)
    adata6.var_names_make_unique()
    adata6.obs_names_make_unique()

    adata = adata.concatenate(adata2, adata4, adata6)
    return adata

def read_liver_data_2(n_obs=250):
    n_obs=int(n_obs/2)
    adata = read_cr_single_file_layer("cr/GSM4308343_UMI_tab_ZT00A.txt", layer_path="cr/ZT00A_reco.txt", ZT="0",
                                      n_obs=n_obs)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata1 = read_cr_single_file_layer("cr/GSM4308344_UMI_tab_ZT00B.txt", layer_path="cr/ZT00B_reco.txt", ZT="0",
                                      n_obs=n_obs)
    adata1.var_names_make_unique()
    adata1.obs_names_make_unique()
    adata=adata.concatenate(adata1)
    adata2 = read_cr_single_file_layer("cr/GSM4308346_UMI_tab_ZT06A.txt", layer_path="cr/ZT06A_reco.txt", ZT="6",
                                      n_obs=n_obs)
    adata2.obs_names_make_unique()
    adata2.var_names_make_unique()
    adata3 = read_cr_single_file_layer("cr/GSM4308347_UMI_tab_ZT06B.txt", layer_path="cr/ZT06B_reco.txt", ZT="6",
                                      n_obs=n_obs)
    adata3.obs_names_make_unique()
    adata3.var_names_make_unique()
    adata2=adata2.concatenate(adata3)

    adata4 = read_cr_single_file_layer("cr/GSM4308348_UMI_tab_ZT12A.txt", layer_path="cr/ZT12A_reco.txt", ZT="12", n_obs=n_obs)
    adata4.var_names_make_unique()
    adata4.obs_names_make_unique()
    adata5 = read_cr_single_file_layer("cr/GSM4308349_UMI_tab_ZT12B.txt", layer_path="cr/ZT12B_reco.txt", ZT="12",
                                      n_obs=n_obs)
    adata5.var_names_make_unique()
    adata5.obs_names_make_unique()
    adata4=adata4.concatenate(adata5)

    adata6 = read_cr_single_file_layer("cr/GSM4308351_UMI_tab_ZT18A.txt", layer_path="cr/ZT18A_reco.txt", ZT="18",
                                      n_obs=n_obs)
    adata6.var_names_make_unique()
    adata6.obs_names_make_unique()
    adata7 = read_cr_single_file_layer("cr/GSM4308352_UMI_tab_ZT18B.txt", layer_path="cr/ZT18B_reco.txt", ZT="18",
                                      n_obs=n_obs)
    adata7.var_names_make_unique()
    adata7.obs_names_make_unique()
    adata6=adata6.concatenate(adata7)

    adata = adata.concatenate(adata2, adata4, adata6)
    return adata

def read_liver_data_full(n_obs=6000):
    adata = read_cr_single_file_layer_full("cr/GSM4308343_UMI_tab_ZT00A.txt", layer_path="cr/ZT00A_reco.txt", ZT="0")
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata1 = read_cr_single_file_layer_full("cr/GSM4308344_UMI_tab_ZT00B.txt", layer_path="cr/ZT00B_reco.txt", ZT="0")
    adata1.var_names_make_unique()
    adata1.obs_names_make_unique()
    adata=adata.concatenate(adata1)
    adata2 = read_cr_single_file_layer_full("cr/GSM4308346_UMI_tab_ZT06A.txt", layer_path="cr/ZT06A_reco.txt", ZT="6")
    adata2.obs_names_make_unique()
    adata2.var_names_make_unique()
    adata3 = read_cr_single_file_layer_full("cr/GSM4308347_UMI_tab_ZT06B.txt", layer_path="cr/ZT06B_reco.txt", ZT="6")
    adata3.obs_names_make_unique()
    adata3.var_names_make_unique()
    adata2=adata2.concatenate(adata3)
    adata4 = read_cr_single_file_layer_full("cr/GSM4308348_UMI_tab_ZT12A.txt", layer_path="cr/ZT12A_reco.txt", ZT="12")
    adata4.var_names_make_unique()
    adata4.obs_names_make_unique()
    adata5 = read_cr_single_file_layer_full("cr/GSM4308349_UMI_tab_ZT12B.txt", layer_path="cr/ZT12B_reco.txt", ZT="12")
    adata5.var_names_make_unique()
    adata5.obs_names_make_unique()
    adata50 = read_cr_single_file_layer_full("cr/GSM4308350_UMI_tab_ZT12C.txt", layer_path="cr/ZT12C_reco.txt", ZT="12")
    adata50.var_names_make_unique()
    adata50.obs_names_make_unique()
    adata4=adata4.concatenate(adata5,adata50)
    adata6 = read_cr_single_file_layer_full("cr/GSM4308351_UMI_tab_ZT18A.txt", layer_path="cr/ZT18A_reco.txt", ZT="18")
    adata6.var_names_make_unique()
    adata6.obs_names_make_unique()
    adata7 = read_cr_single_file_layer_full("cr/GSM4308352_UMI_tab_ZT18B.txt", layer_path="cr/ZT18B_reco.txt", ZT="18")
    adata7.var_names_make_unique()
    adata7.obs_names_make_unique()
    adata6=adata6.concatenate(adata7)
    adata = adata.concatenate(adata2, adata4, adata6)
    sc.pp.subsample(adata, n_obs=n_obs, random_state=0)
    return adata



def read_file_ch(path, n_obs=500,fe="Positive"):
    '''
    :param path: Path to file
    :param n_obs: number of observations to subsample
    :param fe:  iron replete (Fe+) or iron deficient (Fe-)
    :return: adata object
    '''
    adata = sc.read_csv(path).T
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.obs["FE"] = fe
    sc.pp.subsample(adata,n_obs=n_obs)
    return adata


def E_to_range(E):
    order =[]
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if E[i,j]==1:
                order.append(j)
    return np.array(order)


def read_chlamydomonas_files(n_obs=500):
    '''
    :param n_obs: number of observations to subsample
    :return: iron deficient (Fe-) and iron replete (Fe+) adata objects
    '''
    adata_neg = read_file_ch("Chlamydomonas/GSM4770979_run1_CC5390_Fe_neg.csv",n_obs=n_obs,fe="Negative")
    adata_pos = read_file_ch("Chlamydomonas/GSM4770980_run1_CC5390_Fe_pos.csv",n_obs=n_obs,fe="Positive")
    return adata_neg , adata_pos

def read_scn_single_file(path,CT="0" , n_obs=300):
    adata = sc.read_csv(path).T
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.obs['CT'] = CT
    sc.pp.subsample(adata,n_obs=n_obs , random_state=123)
    return adata

def read_scn_single_file_no_ss(path,CT="0"):
    adata = sc.read_csv(path).T
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    adata.obs['CT'] = CT
    return adata

def all_plots_scn(adata,title ):
    sc.tl.pca(adata)
    sc.pl.pca_scatter(adata, color='CT' , title=("PCA of " + title + " painted by CT"))
    pass

def scn_single_cluster(adata,cluster):
    adata = adata[adata.obs['louvain'].isin([cluster])]
    return adata

def read_all_scn_no_obs():
    adata = read_scn_single_file_no_ss("SCN/GSM3290582_CT14.csv",  CT="14")
    adata1 = read_scn_single_file_no_ss("SCN/GSM3290583_CT18.csv",  CT="18",)
    adata2 = read_scn_single_file_no_ss("SCN/GSM3290584_CT22.csv",  CT="22")
    adata3 = read_scn_single_file_no_ss("SCN/GSM3290585_CT26.csv",  CT="26")
    adata4 = read_scn_single_file_no_ss("SCN/GSM3290586_CT30.csv",  CT="30")
    adata5 = read_scn_single_file_no_ss("SCN/GSM3290587_CT34.csv", CT="34")
    adata6 = read_scn_single_file_no_ss("SCN/GSM3290588_CT38.csv", CT="14",)
    adata7 = read_scn_single_file_no_ss("SCN/GSM3290589_CT42.csv", CT="18")
    adata8 = read_scn_single_file_no_ss("SCN/GSM3290590_CT46.csv", CT="22")
    adata9 = read_scn_single_file_no_ss("SCN/GSM3290591_CT50.csv", CT="26")
    adata10 = read_scn_single_file_no_ss("SCN/GSM3290592_CT54.csv", CT="30")
    adata11 = read_scn_single_file_no_ss("SCN/GSM3290593_CT58.csv", CT="34")
    adata = adata.concatenate(adata6)
    adata1 = adata1.concatenate(adata7)
    adata2 = adata2.concatenate(adata8)
    adata3 = adata3.concatenate(adata9)
    adata4 = adata4.concatenate(adata10)
    adata5 = adata5.concatenate(adata11)
    adata = adata.concatenate(adata1, adata2,adata3, adata4, adata5)
    return adata

def evaluate_single_scn_cluster(adata,cluster,type_genes,r_genes,gene_regu=0,en_regu=0,filter_regu=0):
    adata_tmp = (adata[adata.obs['louvain'].isin([cluster])])
    adata_tmp = sort_data_crit(adata=adata_tmp.copy(), crit='CT',
                             crit_list=['02', '06', '10', '14', '18', '22'])
    sc.pp.highly_variable_genes(adata_tmp, n_top_genes=3000)#min_mean=0.0125, max_mean=3, min_disp=0.5)
    for gene in r_genes: # Make sure that the rhytmic and type markers are not filtered out
        adata_tmp.var.highly_variable[gene]=True
    for gene in type_genes:
        adata_tmp.var.highly_variable[gene]=True
    adata_tmp = adata_tmp[:, adata_tmp.var.highly_variable]
    all_plots_scn(adata_tmp,title= cluster+ " - raw data, " )
    adata_tmp.write("SCN/" + cluster+"_raw" +".h5ad")
    D = filter_cyclic_genes(adata_tmp.X, regu=gene_regu, iterNum=200)
    D = np.identity(D.shape[0])-D
    adata_en = adata_tmp.copy()
    adata_en.X = (adata_en.X).dot(D)
    F = filter_full(adata_en.X, regu=en_regu, iterNum=50)
    adata_en.X = adata_en.X * F
    adata_en.write("SCN/" + cluster+"_en" +".h5ad")
    all_plots_scn(adata_en,title= cluster+ " - enhanced signal, " )
    F = filter_cyclic_full_line(adata_tmp.X, regu=filter_regu, iterNum=50)
    adata_tmp.X = adata_tmp.X * F
    adata_tmp.write("SCN/" + cluster+"_filtered" +".h5ad")
    all_plots_scn(adata_tmp,title= cluster+ " - filtered signal, " )
    pass

def read_all_scn():
    '''
    :return:
    '''
    adata = read_scn_single_file_no_ss("SCN/GSM3290582_CT14.csv",  CT="14")
    adata1 = read_scn_single_file_no_ss("SCN/GSM3290583_CT18.csv",  CT="18",)
    adata2 = read_scn_single_file_no_ss("SCN/GSM3290584_CT22.csv",  CT="22")
    adata3 = read_scn_single_file_no_ss("SCN/GSM3290585_CT26.csv",  CT="02")
    adata4 = read_scn_single_file_no_ss("SCN/GSM3290586_CT30.csv",  CT="06")
    adata5 = read_scn_single_file_no_ss("SCN/GSM3290587_CT34.csv", CT="10")
    adata6 = read_scn_single_file_no_ss("SCN/GSM3290588_CT38.csv", CT="14",)
    adata7 = read_scn_single_file_no_ss("SCN/GSM3290589_CT42.csv", CT="18")
    adata8 = read_scn_single_file_no_ss("SCN/GSM3290590_CT46.csv", CT="22")
    adata9 = read_scn_single_file_no_ss("SCN/GSM3290591_CT50.csv", CT="02")
    adata10 = read_scn_single_file_no_ss("SCN/GSM3290592_CT54.csv", CT="06")
    adata11 = read_scn_single_file_no_ss("SCN/GSM3290593_CT58.csv", CT="10")
    adata = adata.concatenate(adata6)
    adata1 = adata1.concatenate(adata7)
    adata2 = adata2.concatenate(adata8)
    adata3 = adata3.concatenate(adata9)
    adata4 = adata4.concatenate(adata10)
    adata5 = adata5.concatenate(adata11)
    adata = adata.concatenate(adata1, adata2,adata3, adata4, adata5)
    return adata

def read_all_scn_no_24(n_obs=250):
    adata = read_scn_single_file("SCN/GSM3290582_CT14.csv",  CT="14",
                                      n_obs=n_obs)
    adata1 = read_scn_single_file("SCN/GSM3290583_CT18.csv",  CT="18",
                                      n_obs=n_obs)
    adata2 = read_scn_single_file("SCN/GSM3290584_CT22.csv",  CT="22",
                                      n_obs=n_obs)
    adata3 = read_scn_single_file("SCN/GSM3290585_CT26.csv",  CT="26",
                                      n_obs=n_obs)
    adata4 = read_scn_single_file("SCN/GSM3290586_CT30.csv",  CT="30",
                                      n_obs=n_obs)
    adata5 = read_scn_single_file("SCN/GSM3290587_CT34.csv", CT="34",
                                  n_obs=n_obs)
    adata6 = read_scn_single_file("SCN/GSM3290588_CT38.csv", CT="14",
                                  n_obs=n_obs)
    adata7 = read_scn_single_file("SCN/GSM3290589_CT42.csv", CT="18",
                                  n_obs=n_obs)
    adata8 = read_scn_single_file("SCN/GSM3290590_CT46.csv", CT="22",
                                  n_obs=n_obs)
    adata9 = read_scn_single_file("SCN/GSM3290591_CT50.csv", CT="26",
                                  n_obs=n_obs)
    adata10 = read_scn_single_file("SCN/GSM3290592_CT54.csv", CT="30",
                                  n_obs=n_obs)
    adata11 = read_scn_single_file("SCN/GSM3290593_CT58.csv", CT="34",
                                  n_obs=n_obs)
    adata = adata.concatenate(adata6)
    adata1 = adata1.concatenate(adata7)
    adata2 = adata2.concatenate(adata8)
    adata3 = adata3.concatenate(adata9)
    adata4 = adata4.concatenate(adata10)
    adata5 = adata5.concatenate(adata11)
    adata = adata.concatenate(adata1, adata2,adata3, adata4, adata5)
    return adata


def chlam_genes(adata):
    tmp_hour = copy.deepcopy(adata[:, 0].X)
    tmp_hour *= 0
    df = pd.read_csv("Chlamydomonas/ch_genes.csv", header=None)
    new_header = df.iloc[1]
    df = df[2:]
    df.columns = new_header
    df = df.dropna()
    for i in range(48):
        if i % 2 == 0:
            labeled_genes = df.loc[df['phase'] == str(int(0.5 * i))]
        else:
            labeled_genes = df.loc[df['phase'] == str(0.5 * i)]
        for j in labeled_genes.values:
            try:
                gene_string = j[0] + ".v5.5"
                tmp_hour += adata[:, gene_string].X
            except:
                #print("gene was filtered out")
                a=1
        if i % 8 == 0:
            tmp_hour /= tmp_hour.max()
            ranged_pca_2d(adata.X, color=tmp_hour, title=(str(0.5 * i) + " filtered"))
            tmp_hour *= 0

    pass

def plot_diurnal_cycle_by_phase(adata, title = ""):

    phase_array = np.zeros((6,adata.X.shape[0]))
    df = pd.read_csv("Chlamydomonas/ch_genes.csv", header=None)
    new_header = df.iloc[1]
    df = df[2:]
    df.columns = new_header
    df = df.dropna()
    for i in range(6):
        for j in range(8):
            if  2 == 0:
                labeled_genes = df.loc[df['phase'] == str(int(0.5 * j  + i*4))]
            else:
                labeled_genes = df.loc[df['phase'] == str(0.5 * j  + i*4)]
            for k in labeled_genes.values:
                try:
                    gene_string = k[0] + ".v5.5"
                    a = adata[:, gene_string].X[0, :]
                    b = adata[:, gene_string].X[:, 0]
                    phase_array[i,:] += adata[:, gene_string].X[:, 0]
                except:
                    #print("gene was filtered out")
                    a=1
    for i in range(6):
        ranged_pca_2d((adata.X),scipy.signal.savgol_filter(phase_array[i,:]/phase_array[i,:].max(),window_length=35,polyorder=3), title=title + " phase: " +str(i*4) + " - " +str((i*4 +3.5)))
    pass




def score_single_type(path,cluster):
    adata = sc.read(filename=path)
    #sc.pp.filter_genes_dispersion(adata,n_top_genes=7000)
    adata = scn_single_cluster(adata, str(cluster))
    genes = sc.pp.highly_variable_genes(adata,n_top_genes=7000,inplace=False)
    genes_values = genes.loc[genes['highly_variable']==False]
    for i , r in genes_values.iterrows():
        adata[:,i].X*= 0
    labels_str = adata.obs["CT"]
    labels = np.zeros(adata.X.shape[0])

    for j, k in enumerate(labels_str):
        labels[j]=int(k)
    print(str(cluster))
    print("silhoutte score before: " + str(silhouette_score(adata.X, labels)))
    # a=1/0
    all_plots_scn(adata, title=("Raw data, cluster: " +str(cluster)), n_obs=1)
    #D = filter_non_cyclic_genes(adata.X, regu=0.1, iterNum=100)
    #adata.X = adata.X.dot(D)
    F = filter_full(adata.X, regu=25, iterNum=250)
    print(str(i))
    print("norm: " + str(np.linalg.norm(adata.X)))
    print("norm change: " + str(np.linalg.norm(adata.X - adata.X *F)))
    adata.X = adata.X * F
    labels_str = adata.obs["CT"]
    labels = np.zeros(adata.X.shape[0])
    adata.write(filename=("scn_" +str(cluster)+ "_filtered_data_250_25.h5ad"),compression='gzip')

    for j, k in enumerate(labels_str):
        labels[j]=int(k)
    print(str(i))
    print("silhoutte score after: " + str(silhouette_score(adata.X, labels)))

    all_plots_scn(adata, title=("Filtered data, cluster: " +str(cluster)), n_obs=1)
    pass




def paint_HeLa_by_phase(adata_filtered,adata_unfiltered):
    cyclic_by_phase = pd.read_csv("cyclic_by_phase.csv")
    df = cyclic_by_phase["G1.S"]
    list_of_genes=[]
    list_a = df.values.tolist()
    for a in list_a:
      list_of_genes.append(a)
    G1S , G1S_F = score_list_of_genes(cyclic_by_phase=cyclic_by_phase , phase="G1.S", filtered=adata_filtered,unfiltered=bdata)
    S , S_F = score_list_of_genes(cyclic_by_phase=cyclic_by_phase , phase="S", filtered=adata_filtered,unfiltered=bdata)
    G2 , G2_F = score_list_of_genes(cyclic_by_phase=cyclic_by_phase , phase="G2", filtered=adata_filtered,unfiltered=bdata)
    G2M , G2M_F = score_list_of_genes(cyclic_by_phase=cyclic_by_phase , phase="G2.M", filtered=adata_filtered,unfiltered=bdata)
    MG1 , MG1_F = score_list_of_genes(cyclic_by_phase=cyclic_by_phase , phase="M.G1", filtered=adata_filtered,unfiltered=bdata)
    plt.plot((range((adata_filtered.X).shape[0])),G1S_F)
    plt.plot((range((adata_filtered.X).shape[0])),S_F)
    plt.plot((range((adata_filtered.X).shape[0])),G2_F)
    plt.plot((range((adata_filtered.X).shape[0])),G2M_F)
    plt.plot((range((adata_filtered.X).shape[0])),MG1_F)
    plt.legend(["G1.S","S","G2","G2.M","M.G1"])
    plt.show()
    ranged_pca_2d((adata_filtered.X),G1S_F,title="G1S_F PCA filtered")
    ranged_pca_2d((adata_filtered.X),S_F, title="S_F PCA filtered")
    ranged_pca_2d((adata_filtered.X),G2_F,title="G2_F PCA filtered")
    ranged_pca_2d((adata_filtered.X),G2M_F,title="G2M_F PCA filtered")
    ranged_pca_2d((adata_filtered.X),MG1_F,title="MG1_F PCA filtered")
    plot_cell_cycle_by_phase(adata_filtered, adata_unfiltered)
    pass



def sort_data_linear(adata):
    adata = shuffle_adata(adata)
    layers = [[] for i in range(8)]
    obs = adata.obs
    for i, row in obs.iterrows():
        layer = int(row['layer'])
        layers[layer].append(i)
    order = sum(layers, [])
    sorted_data = adata[order,:]
    return sorted_data

def sort_data_crit(adata,crit,crit_list):
    adata = shuffle_adata(adata)
    layers = [[] for i in range(len(crit_list))]
    obs = adata.obs
    for i, row in obs.iterrows():
        layer = (row[crit])
        for j , item in enumerate(crit_list):
            if item==layer:
                layers[j].append(i)
    order = sum(layers, [])
    sorted_data = adata[order,:]
    return sorted_data


def all_plots_hela(adata,title):
    ranged_pca_2d(adata.X,color=range(adata.X.shape[0]),title=("HeLa cells PCA, painted by cell location in the matrix"))
    cyclic_by_phase = pd.read_csv("data/cyclic_by_phase.csv")
    G1S = score_list_of_genes_single_adata(cyclic_by_phase=cyclic_by_phase, phase="G1.S", adata=adata)
    S = score_list_of_genes_single_adata(cyclic_by_phase=cyclic_by_phase, phase="S", adata=adata)
    G2 = score_list_of_genes_single_adata(cyclic_by_phase=cyclic_by_phase, phase="G2", adata=adata)
    G2M = score_list_of_genes_single_adata(cyclic_by_phase=cyclic_by_phase, phase="G2.M", adata=adata)
    MG1 = score_list_of_genes_single_adata(cyclic_by_phase=cyclic_by_phase, phase="M.G1", adata=adata)
    ranged_pca_2d((adata.X), G1S / G1S.max(), title=("G1S PCA " +title))
    ranged_pca_2d((adata.X), S / S.max(), title=("S PCA filtered"+title))
    ranged_pca_2d((adata.X), G2 / G2.max(), title=("(G2 PCA filtered"+title))
    ranged_pca_2d((adata.X), G2M / G2M.max(), title=("G2M PCA filtered"+title))
    ranged_pca_2d((adata.X), MG1 / MG1.max(), title=("MG1 PCA filtered"+title))
    G1S= G1S[:,0]
    G2= G2[:,0]
    S= S[:,0]
    MG1= MG1[:,0]
    G2M= G2M[:,0]
    G1_len = len(savgol_filter((G1S / G1S.max()),7,5))
    theta = (np.array(range(G1_len)) * 2 * np.pi) / G1_len
    max_val = max(np.max(savgol_filter((S/ np.sum(S)),25,3)),
                  np.max(savgol_filter((G1S/ np.sum(G1S)),25,3)),
                  np.max(savgol_filter((G2/ np.sum(G2)),25,3)),
                  np.max(savgol_filter((G2M/ np.sum(G2M)),25,3)),
                  np.max(savgol_filter((MG1/ np.sum(MG1)),25,3)))
    max_val_int = int(max_val*1000) +1
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    #ax.plot(theta, savgol_filter((G1S / G1S.max()),25,3), color='blue',linewidth=1.5)
    ax.plot(theta, savgol_filter((G1S / np.sum(G1S)),25,3), color='blue',linewidth=1.5)
    ax.set_rmax(max_val_int/1000)
    ticks_array = np.array(range(max_val_int+1))/1000
    #ax.set_rticks([0.001, 0.002])  # Less radial ticks
    ax.set_rticks(ticks_array)  # Less radial ticks
    ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
    ax.grid(True)
    plt.tick_params(labelsize=16)

    ax.set_title(("Normalized sum of genes related different phases- " +str(title)), va='bottom')

    #ax.plot(theta, savgol_filter((S/ S.max()),25,3),color='orange' ,linewidth=1.5)
    #ax.plot(theta, savgol_filter((G2 / G2.max()),25,3), color='green',linewidth=1.5)
    #ax.plot(theta, savgol_filter((G2M / G2M.max()),25,3), color='red' ,linewidth=1.5)
    #ax.plot(theta, savgol_filter((MG1 / MG1.max()),25,3) , color='purple',linewidth=1.5)

    ax.plot(theta, savgol_filter((S/ np.sum(S)),25,3),color='orange' ,linewidth=1.5)
    ax.plot(theta, savgol_filter((G2 / np.sum(G2)),25,3), color='green',linewidth=1.5)
    ax.plot(theta, savgol_filter((G2M / np.sum(G2M)),25,3), color='red' ,linewidth=1.5)
    ax.plot(theta, savgol_filter((MG1 / np.sum(MG1)),25,3) , color='purple',linewidth=1.5)

    ax.legend(["G1.S", "S", "G2", "G2.M", "M.G1"], fontsize=15 , loc = 'center left', bbox_to_anchor = (1.2, 0.5))
    ax.scatter(circular_mean(theta,MG1 / np.sum(MG1))[0], max_val_int/1000,color='purple' ,  marker='*')#, color='r' , label='Mean')
    ax.scatter(circular_mean(theta,G2M / np.sum(G2M))[0], max_val_int/1000,color='red' ,  marker='*')#, color='r' , label='Mean')
    ax.scatter(circular_mean(theta,G2/np.sum(G2))[0], max_val_int/1000,color='green' ,  marker='*')#, color='r' , label='Mean')
    ax.scatter(circular_mean(theta,G1S / np.sum(G1S))[0], max_val_int/1000,color='blue', marker='*')#, color='r' , label='Mean')
    ax.scatter(circular_mean(theta,S/ np.sum(S))[0], max_val_int/1000,color='orange', marker='*')#, color='r' , label='Mean')
    plt.show()
    print("Circular mean and variance, G1S" + str(circular_mean(theta,G1S / np.sum(G1S))))
    print("Circular mean and variance, S" + str(circular_mean(theta,S/ np.sum(S))))
    print("Circular mean and variance, G2" + str(circular_mean(theta,G2/np.sum(G2))))
    print("Circular mean and variance, G2M" + str(circular_mean(theta,G2M / np.sum(G2M))))
    print("Circular mean and variance, MG1" + str(circular_mean(theta,MG1 / np.sum(MG1))))
    pass


def circular_mean(angles, weights=None):
    #https://stackoverflow.com/questions/52856232/scipy-circular-variance
    # https://en.wikipedia.org/wiki/Circular_mean
    if weights is None:
        weights = np.ones(len(angles))

    vectors = [[w * np.cos(a), w * np.sin(a)] for a, w in zip(angles, weights)]

    vector = np.sum(vectors, axis=0) / np.sum(weights)

    x, y = vector

    angle_mean = np.arctan2(y, x)
    angle_variance = 1. - np.linalg.norm(vector)  # x*2+y*2 = hypot(x,y)

    return angle_mean, angle_variance


