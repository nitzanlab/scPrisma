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


def auc_vs_noise():
    A = simulate_star_all_cells(ngenes=1000, nmut=7, skip=1, depth=2, ix=[1, 2])
    np.random.shuffle(A)
    print(A.shape)
    V = simulate_spatial_cyclic_z(ngenes=A.shape[1],ncells=A.shape[0], w =0.3)
    A[:,500:1000]=V[:,500:1000]
    y_true = np.zeros(1000)
    y_true[500:1000]=np.ones(500)
    noise_list =[]
    auc_list = []
    T = copy.deepcopy(A)
    for i in range(100):
        noise = np.random.normal(0,i*0.1,(A.shape))
        K = T + noise
        #E_sga  , E_rec_sga = sga_m_reorder_rows_matrix(K , iterNum=200 , batch_size=700)
        K = gene_normalization(K)
        D = filter_non_cyclic_genes(K, regu =0.1 , iterNum=20)
        res = np.diagonal(D)
        noisy=i*0.1
        print(noisy)
        print(" AUC-ROC: " + str(calculate_roc_auc(res,y_true)) )
        auc_list.append(calculate_roc_auc(res,y_true))
        noise_list.append(0.1*i)

    auc_a = np.array(auc_list)
    noise_a = np.array(noise_list)
    plt.plot(noise_a, auc_a)
    plt.show()

    with open('auc_a.npy', 'wb') as f:
        np.save(f,auc_a)
    with open('noise_a.npy', 'wb') as f:
        np.save(f,noise_a)
    pass

def calculate_avg_groups_layer(adata):
    avg_groups = np.zeros((8,adata.X.shape[1]))
    for i in range(8):
        tmp_adata = adata[adata.obs["layer"] == str(i)]
        for j in range(tmp_adata.X.shape[0]):
            avg_groups[j, :] += tmp_adata[j, :].X[0, :]
        avg_groups[j,:]/=tmp_adata.X.shape[0]
    return avg_groups

def corr_vs_noise():
    A = simulate_star_all_cells(ngenes=1000, nmut=7, skip=1, depth=2, ix=[1, 2])
    np.random.shuffle(A)
    n=A.shape[0]
    print(A.shape)
    V = simulate_spatial_cyclic_z(ngenes=A.shape[1],ncells=A.shape[0], w =0.3)
    A[:,500:1000]=V[:,500:1000]
    y_true = np.zeros(1000)
    y_true[500:1000]=np.ones(500)
    noise_list =[]
    corr_list = []
    T = copy.deepcopy(A)
    for i in range(40):
        print("Iteration number: " + str(i))
        noise = np.random.normal(0,i*0.1,(A.shape))
        K = T + noise
        E_sga  , E_rec_sga = sga_m_reorder_rows_matrix(K , iterNum=200 , batch_size=700)
        res = spearm(E_rec_sga,np.array(range(n)))
        print(" Corr: " + str(res))
        corr_list.append(res)
        noise_list.append(0.1*i)
    auc_a = np.array(corr_list)
    noise_a = np.array(noise_list)
    with open('corr_a.npy', 'wb') as f:
        np.save(f,auc_a)
    with open('noise_c.npy', 'wb') as f:
        np.save(f,noise_a)
    pass

def analyze_auc_vs_noise():
    auc_a = np.load("auc_a.npy")
    noise_a = np.load("noise_a.npy")
    plt.plot(noise_a,auc_a)
    plt.xlabel("Noise variance")
    plt.ylabel("AUC-ROC")
    plt.title("AUC-ROC as a function of Gaussian noise")
    plt.show()


def auc_vs_genesnum():
    A = simulate_star_all_cells(ngenes=1000, nmut=7, skip=1, depth=2, ix=[1, 2])
    np.random.shuffle(A)
    print(A.shape)
    V = simulate_spatial_cyclic_z(ngenes=A.shape[1],ncells=A.shape[0], w =0.3)
    genes_num_list =[]
    auc_list = []
    for i in range(1,99):
        T = copy.deepcopy(A)
        T[:,i*10:1000]=V[:,i*10:1000]
        y_true = np.zeros(1000)
        y_true[i*10:1000]=np.ones(1000-i*10)
        noise = np.random.normal(0,1,(A.shape))
        K = T + noise
        #E_sga  , E_rec_sga = sga_m_reorder_rows_matrix(K , iterNum=200 , batch_size=700)
        K = gene_normalization(K)
        D = filter_non_cyclic_genes(K, regu =0.1 , iterNum=20)
        res = np.diagonal(D)
        noisy=i*0.1
        print(noisy)
        print(" AUC-ROC: " + str(calculate_roc_auc(res,y_true)) )
        auc_list.append(calculate_roc_auc(res,y_true))
        genes_num_list.append(1000-i*10)

    auc_a = np.array(auc_list)
    genes_num_list = np.array(genes_num_list)
    plt.plot(genes_num_list, auc_a)
    plt.show()
    with open('auc_genes.npy', 'wb') as f:
        np.save(f,auc_a)
    with open('genes_num_list.npy', 'wb') as f:
        np.save(f,genes_num_list)
    pass

def analyze_auc_vs_genesnum():
    auc_a = np.load("auc_genes.npy")
    genes_num = np.load("genes_num_list.npy")
    plt.plot(genes_num,auc_a)
    plt.xlabel("Number of cyclic genes")
    plt.ylabel("AUC-ROC")
    plt.title("AUC-ROC as a function of number of cyclic genes")
    plt.show()


def auc_vs_genesnum_shuffled():
    A = simulate_star_all_cells(ngenes=1000, nmut=8, skip=1, depth=2, ix=[1, 2])
    np.random.shuffle(A)
    print(A.shape)
    V = simulate_spatial_cyclic_z(ngenes=A.shape[1],ncells=A.shape[0], w =0.3)
    np.random.shuffle(V)
    genes_num_list =[]
    auc_list = []
    for i in range(1,50):
        T = copy.deepcopy(A)
        T[:,i*20:1000]=V[:,i*20:1000]
        y_true = np.zeros(1000)
        y_true[i*20:1000]=np.ones(1000-i*20)
        noise = np.random.normal(0,1,(A.shape))
        K = T + noise
        E_sga  , E_rec_sga = sga_m_reorder_rows_matrix(K , iterNum=200 , batch_size=700)
        K = gene_normalization(K)
        D = filter_non_cyclic_genes(E_rec_sga.dot(K), regu =0.1 , iterNum=20)
        res = np.diagonal(D)
        noisy=i*0.1
        print(noisy)
        print(" AUC-ROC: " + str(calculate_roc_auc(res,y_true)) )
        plt.plot(range(len(res)),res)
        plt.show()
        auc_list.append(calculate_roc_auc(res,y_true))
        genes_num_list.append(1000-i*20)

    auc_a = np.array(auc_list)
    genes_num_list = np.array(genes_num_list)
    with open('auc_genes_s.npy', 'wb') as f:
        np.save(f,auc_a)
    with open('genes_num_list_s.npy', 'wb') as f:
        np.save(f,genes_num_list)
    pass

def analyze_auc_vs_genesnum_s():
    auc_a = np.load("auc_genes_s.npy")
    genes_num = np.load("genes_num_list_s.npy")
    plt.plot(genes_num,auc_a)
    plt.xlabel("Number of cyclic genes")
    plt.ylabel("AUC-ROC")
    plt.title("AUC-ROC as a function of number of cyclic genes")
    plt.show()


def analyze_cr():
    A = cr_read()
    E , E_rec = sga_m_reorder_rows_matrix(A, iterNum=300, batch_size=500)

    with open('E_rec_cr.npy', 'wb') as f:
        np.save(f,E_rec)
    with open('E_cr.npy', 'wb') as f:
        np.save(f,E)

    A = cr_read(file_list=["cr/r_genes.csv", "cr/zxr_genes.csv"])
    D = filter_non_cyclic_genes(E_rec.dot(A))

    with open('D.npy', 'wb') as f:
        np.save(f,D)
    return E_rec.dot(A)


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

def analyze_and_score_mESC():
    V = mESC_read(cc_genes=True , variance_t=2)
    E , E_rec = sga_m_reorder_rows_matrix(V, iterNum=300, batch_size=400)
    labels = np.zeros(96*3)
    #labels[0:96]=np.zeros(96)
    labels[96:96*2]=np.ones(96)
    labels[96*2:96*3]=np.ones(96)*2
    print(f1_score_multi_class(E_rec,[96,96,96],labels))
    with open('E_mESC_hv_genes.npy', 'wb') as f:
        np.save(f,E)
    pass

def analyze_and_score_cr():
    V = cr_read()
    #V = cell_normalization(V)
    #plt.imshow(V.dot(V.T))
    #plt.show()
    #a=1/0
    E , E_rec = sga_m_reorder_rows_matrix(V, iterNum=50, batch_size=800, lr=0.1)
    print(E.shape)
    with open('E_cr_cr_genes_1.npy', 'wb') as f:
        np.save(f,E)
    labels = np.zeros(1000)
    #labels[0:96]=np.zeros(96)
    labels[250:500]=np.ones(250)
    labels[500:750]=np.ones(250)*2
    labels[750:1000]=np.ones(250)*3
    #labels[500:600]=np.ones(100)
    #labels[600:700]=np.ones(100)*2
    #labels[700:800]=np.ones(100)*3
    print(f1_score_multi_class(E_rec,[250,250,250,250],labels))
    pass

def analyze_and_score_cr_hv():
    V = cr_read_hv()
    #V = cell_normalization(V)
    #plt.imshow(V.dot(V.T))
    #plt.show()
    #a=1/0
    print(V.shape)
    E , E_rec = sga_m_reorder_rows_matrix(V, iterNum=150, batch_size=8000 , lr=0.01)
    with open('E_cr_hv_genes_1.npy', 'wb') as f:
        np.save(f,E)
    labels = np.zeros(1000)
    #labels[0:96]=np.zeros(96)
    labels[250:500]=np.ones(250)
    labels[500:750]=np.ones(250)*2
    labels[750:1000]=np.ones(250)*3
    #labels[500:600]=np.ones(100)
    #labels[600:700]=np.ones(100)*2
    #labels[700:800]=np.ones(100)*3
    print(f1_score_multi_class(E_rec,[250,250,250,250],labels))
    pass


def cr_filtering_performence():
    V = cr_read_hv( cell_num =200)
    plt.imshow(V.dot(V.T))
    plt.show()
    labels = np.zeros(800)
    labels[200:400]=np.ones(200)
    labels[400:600]=np.ones(200)*2
    labels[600:800]=np.ones(200)*3
    painted_isomap_2D(V , labels)
    painted_lle_2D(V ,4,[0,200,400,600],[200,400,600,800] , group_label=["T=0","T=6","T=12","T=18"])
    print(np.linalg.norm(V))
    for i in range(1,10):
        print(i)
        F = filter_full(V , regu=0 , iterNum=i*4)
        painted_isomap_2D(V*F , labels)
        painted_lle_2D(V*F ,4,[0,200,400,600],[200,400,600,800] , group_label=["T=0","T=6","T=12","T=18"] , title=("LLE2D, iteration number= " +str(i*4)))
        print(np.linalg.norm(V-V*F))
    pass

def cr_filtering(variance_t=10):
    V = cr_read_hv( cell_num =200)
    plt.imshow(V.dot(V.T))
    plt.show()
    labels = np.zeros(800)
    labels[200:400]=np.ones(200)
    labels[400:600]=np.ones(200)*2
    labels[600:800]=np.ones(200)*3
    painted_isomap_2D(V , labels)
    painted_lle_2D(V ,4,[0,200,400,600],[200,400,600,800] , group_label=["T=0","T=6","T=12","T=18"])
    print(np.linalg.norm(V))
    for i in range(1,10):
        print(i)
        F = filter_cyclic_full(V , regu=0 , iterNum=i*4)
        painted_isomap_2D(V*F , labels)
        painted_lle_2D(V*F ,4,[0,200,400,600],[200,400,600,800] , group_label=["T=0","T=6","T=12","T=18"] , title=("LLE2D, iteration number= " +str(i*4)))
        print(np.linalg.norm(V-V*F))
    pass

def read_helaa():
    adata= sc.read_text("GSM4224315.txt")
    adata =adata.T
    sc.pp.filter_cells(adata, min_counts=1 )
    sc.pp.filter_genes(adata, min_counts=1 )
    #filter_result = sc.pp.highly_variable_genes(adata, n_top_genes=8000 , inplace=True , n_bins=50)
    #adata = adata[:, filter_result.gene_subset]     # subset the genes
    #adata.X = cell_normalization(adata.X)
    sc.pp.filter_genes_dispersion(adata,n_top_genes=8000)
    #sc.pp.recipe_zheng17(adata,n_top_genes=8000)
    E , E_rec = sga_m_reorder_rows_matrix(adata.X, iterNum=300,batch_size=6500)
    range1 = E_to_range(E_rec)
    adata = adata[range1,:]
    F = filter_full(adata.X,regu=5,iterNum=1)
    adata.X = adata.X * F
    gene_CCND3 = adata[:,'CCND3']
    gene_CCND3 = gene_CCND3.X
    gene_CCND3 = np.array(gene_CCND3)
    gene_CDK4 = adata[:,'CDK4']
    gene_CDK4 = gene_CDK4.X
    gene_CDK4 = np.array(gene_CDK4)
    gene_CDK2 = adata[:,'CDK2']
    gene_CDK2 = gene_CDK2.X
    gene_CDK2 = np.array(gene_CDK2)
    gene_CDK1 = adata[:,'CDK1']
    gene_CDK1 = gene_CDK1.X
    gene_CDK1= np.array(gene_CDK1)

    with open('adata_filtered.npy', 'wb') as f:
        np.save(f,adata.X)
    with open('E.npy', 'wb') as f:
        np.save(f,E)
    with open('gene_CCND3.npy', 'wb') as f:
        np.save(f,np.array(gene_CCND3))
    with open('gene_CDK4.npy', 'wb') as f:
        np.save(f,np.array(gene_CDK4))
    with open('gene_CDK1.npy', 'wb') as f:
        np.save(f,np.array(gene_CDK1))
    pass

def read_cr():
    ZT00A= sc.read_text("cr/ZT00A.txt")
    ZT00A = ZT00A.T
    ZT00A.var_names_make_unique()
    ZT06A= sc.read_text("cr/ZT06A.txt")
    ZT06A = ZT06A.T
    ZT06A.var_names_make_unique()
    ZT12A= sc.read_text("cr/ZT12A.txt")
    ZT12A = ZT12A.T
    ZT12A.var_names_make_unique()
    ZT18A= sc.read_text("cr/ZT18A.txt")
    ZT18A = ZT18A.T
    ZT18A.var_names_make_unique()
    outer = ZT00A.concatenate(ZT06A , join='outer')
    adata= sc.read_text("GSM4224315.txt")
    adata =adata.T
    sc.pp.filter_cells(adata, min_counts=1 )
    sc.pp.filter_genes(adata, min_counts=1 )
    #filter_result = sc.pp.highly_variable_genes(adata, n_top_genes=8000 , inplace=True , n_bins=50)
    #adata = adata[:, filter_result.gene_subset]     # subset the genes
    #adata.X = cell_normalization(adata.X)
    sc.pp.filter_genes_dispersion(adata,n_top_genes=8000)
    #sc.pp.recipe_zheng17(adata,n_top_genes=8000)
    E , E_rec = sga_m_reorder_rows_matrix(adata.X, iterNum=100,batch_size=7000)
    range1 = E_to_range(E_rec)
    adata = adata[range1,:]
    F = filter_full(adata.X,regu=5,iterNum=1)
    adata.X = adata.X * F
    gene_CCND3 = adata[:,'CCND3']
    gene_CCND3 = gene_CCND3.X
    gene_CCND3 = np.array(gene_CCND3)
    with open('adata_filtered.npy', 'wb') as f:
        np.save(f,adata.X)
    with open('E.npy', 'wb') as f:
        np.save(f,E)
    with open('gene_CCND3.npy', 'wb') as f:
        np.save(f,np.array(gene_CCND3))
    pass


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


def corr_vs_noise_snr_bins():
    n = 120
    p = 500
    V = simulate_spatial_cyclic_z(ngenes=p, ncells=n, w=0.3)
    Signal = np.linalg.norm(V)
    labels = np.zeros(120)
    groups = []
    for i in range(4):
        labels[30 * i:30 * (i + 1)] = np.ones(30) * i
        groups.append(30)
    snr_list = []
    score_list = []
    for i in range(40):
        print("Iteration number: " + str(i))
        noise = np.random.normal(0, i * 0.25, (V.shape))
        #noise = np.clip(noise,a_min=0,a_max=np.inf)
        noise_p = np.linalg.norm(noise)
        K = V + noise
        E_sga, E_rec_sga = sga_m_reorder_rows_matrix(K, iterNum=100, batch_size=400)
        res = f1_score_multi_class(E_rec_sga, groups, labels)
        print(" F1 score: " + str(res))
        score_list.append(res)
        SNR = Signal/noise_p
        snr_list.append(SNR)
    f1_score = np.array(score_list)
    noise_snr = np.array(snr_list)
    with open('score_a_bins.npy', 'wb') as f:
        np.save(f, f1_score)
    with open('noise_snr.npy', 'wb') as f:
        np.save(f, noise_snr)
    plt.plot(noise_snr, f1_score)
    plt.xlabel("SNR")
    plt.ylabel("f1 score")
    plt.title("F1 score of 4 bins as a function of Gaussian noise")
    plt.show()
    pass



def corr_vs_noise_bins():
    A = simulate_star_all_cells(ngenes=1000, nmut=7, skip=1, depth=3, ix=[1, 2])
    np.random.shuffle(A)
    print(A.shape)
    n=A.shape[0]
    p=1000
    #print(A.shape)
    V = simulate_spatial_cyclic_z(ngenes=p,ncells=n, w =0.3)
    A[:,500:1000]=V[:,500:1000]
    labels = np.zeros(120)
    groups=[]
    for i in range(4):
        labels[30*i:30*(i+1)]=np.ones(30)*i
        groups.append(30)
    noise_list =[]
    score_list = []
    T = copy.deepcopy(A)
    for i in range(40):
        print("Iteration number: " + str(i))
        noise = np.random.normal(0,i*0.1,(A.shape))
        K = T + noise
        E_sga  , E_rec_sga = sga_m_reorder_rows_matrix(K , iterNum=150 , batch_size=700)
        res = f1_score_multi_class(E_rec_sga,groups,labels)
        #res = spearm(E_rec_sga,np.array(range(n)))
        print(" F1 score: " + str(res))
        score_list.append(res)
        noise_list.append(0.1*i)
    f1_score = np.array(score_list)
    noise_a = np.array(noise_list)
    with open('score_a_bins.npy', 'wb') as f:
        np.save(f,f1_score)
    with open('noise_bins.npy', 'wb') as f:
        np.save(f,noise_a)
    plt.plot(noise_a,f1_score)
    plt.xlabel("Noise variance")
    plt.ylabel("")
    plt.title("F1 score of 4 bins as a function of Gaussian noise")
    plt.show()
    pass

def auc_vs_genesnum():
    A = simulate_star_all_cells(ngenes=1000, nmut=7, skip=1, depth=2, ix=[1, 2])
    np.random.shuffle(A)
    print(A.shape)
    V = simulate_spatial_cyclic_z(ngenes=A.shape[1],ncells=A.shape[0], w =0.3)
    genes_num_list =[]
    auc_list = []
    for i in range(1,99):
        T = copy.deepcopy(A)
        T[:,i*10:1000]=V[:,i*10:1000]
        y_true = np.zeros(1000)
        y_true[i*10:1000]=np.ones(1000-i*10)
        noise = np.random.normal(0,0.5,(A.shape))
        K = T + noise
        E_sga  , E_rec_sga = sga_m_reorder_rows_matrix(K , iterNum=200 , batch_size=700)
        K = E_sga.dot(K)
        D = filter_non_cyclic_genes(K, regu =3 , iterNum=20)
        res = np.diagonal(D)
        noisy=i*0.1
        print(noisy)
        print(" AUC-ROC: " + str(calculate_roc_auc(res,y_true)) )
        plt.plot(range(len(res)),res)
        plt.show()
        auc_list.append(calculate_roc_auc(res,y_true))
        genes_num_list.append(1000-i*10)

    auc_a = np.array(auc_list)
    genes_num_list = np.array(genes_num_list)
    with open('auc_genes.npy', 'wb') as f:
        np.save(f,auc_a)
    with open('genes_num_list.npy', 'wb') as f:
        np.save(f,genes_num_list)
    pass

def analyze_auc_vs_genesnum():
    auc_a = np.load("auc_genes.npy")
    genes_num = np.load("genes_num_list.npy")
    plt.plot(genes_num,auc_a)
    plt.xlabel("Number of cyclic genes")
    plt.ylabel("AUC-ROC")
    plt.title("AUC-ROC as a function of number of cyclic genes")
    plt.show()


def auc_vs_genesnum_shuffled():
    A = simulate_star_all_cells(ngenes=1000, nmut=8, skip=1, depth=2, ix=[1, 2])
    np.random.shuffle(A)
    print(A.shape)
    V = simulate_spatial_cyclic_z(ngenes=A.shape[1],ncells=A.shape[0], w =0.3)
    np.random.shuffle(V)
    genes_num_list =[]
    auc_list = []
    for i in range(1,50):
        T = copy.deepcopy(A)
        T[:,i*20:1000]=V[:,i*20:1000]
        y_true = np.zeros(1000)
        y_true[i*20:1000]=np.ones(1000-i*20)
        noise = np.random.normal(0,1,(A.shape))
        K = T + noise
        E_sga  , E_rec_sga = sga_m_reorder_rows_matrix(K , iterNum=200 , batch_size=700)
        K = gene_normalization(K)
        D = filter_non_cyclic_genes(E_rec_sga.dot(K), regu =0.1 , iterNum=20)
        res = np.diagonal(D)
        noisy=i*0.1
        print(noisy)
        print(" AUC-ROC: " + str(calculate_roc_auc(res,y_true)) )
        plt.plot(range(len(res)),res)
        plt.show()
        auc_list.append(calculate_roc_auc(res,y_true))
        genes_num_list.append(1000-i*20)

    auc_a = np.array(auc_list)
    genes_num_list = np.array(genes_num_list)
    with open('auc_genes_s.npy', 'wb') as f:
        np.save(f,auc_a)
    with open('genes_num_list_s.npy', 'wb') as f:
        np.save(f,genes_num_list)
    pass

def analyze_auc_vs_genesnum_s():
    auc_a = np.load("auc_genes_s.npy")
    genes_num = np.load("genes_num_list_s.npy")
    plt.plot(genes_num,auc_a)
    plt.xlabel("Number of cyclic genes")
    plt.ylabel("AUC-ROC")
    plt.title("AUC-ROC as a function of number of cyclic genes")
    plt.show()

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

def simulated_disentanglement():
    #V = simulate_linear_2(ngenes=500, nmut=1250, skip=5)
    V = simulate_spatial_1d(ngenes=500,ncells=250, w =0.2)
    np.random.shuffle(V)
    V[:,250:500]=np.zeros((V.shape[0],250))
    ranged_pca_2d(V,color=range(V.shape[0]),title="PCA of linear signal")
    A = simulate_spatial_cyclic_z(ngenes=500,ncells=250, w =0.2)
    A[:,0:250]=np.zeros((A.shape[0],250))
    ranged_pca_2d(A,color=range(A.shape[0]),title="PCA of cyclic signal")
    print("Cyclic starting norm: " +str(np.linalg.norm(A)))
    print("Linear starting norm: " +str(np.linalg.norm(V)))
    print(A.shape)
    cyclic_signal = copy.deepcopy(A)
    V[:,250:500]=A[:,250:500]
    noise = np.random.normal(0,0.25,V.shape)
    #noise = np.clip(noise,a_min=0 , a_max = np.inf)
    print("Noise starting norm: " +str(np.linalg.norm(noise)))
    V= V+ noise
    ranged_pca_2d(noise,color=range(noise.shape[0]),title="PCA of Gaussian noise")
    mse = np.inf
    for i in range(V.shape[0]):
        tmp_mse = np.linalg.norm(np.roll(V, i) - cyclic_signal)
        if tmp_mse < mse:
            mse = tmp_mse
    print("Starting MSE: " + str(mse))

    print("Total signal starting norm: " +str(np.linalg.norm(V)))
    #np.random.shuffle(V)
    ranged_pca_2d(V,color=range(V.shape[0]),title="PCA of simulated signal")
    #E , E_rec = sga_m_reorder_rows_matrix(V, iterNum=100, batch_size=400 , lr=0.1 , gama=0.9)
    #plt.imshow(E_rec)
    #plt.show()
    V_rec =V#E_rec.dot(V)
    plt.plot(range(V_rec.shape[0]), V_rec[:, 300])
    plt.title("Cyclic gene before filtering")
    plt.show()
    D = filter_non_cyclic_genes(V_rec,regu=5,iterNum=100)
    V_rec = V_rec.dot(D)
    plot_diag(D)
    pca_3d(V_rec)
    mse_list =  []
    iter_list = []
    for i in range(1):
        F = filter_full(V_rec,regu=20,iterNum=250)
        plt.plot(range(V_rec.shape[0]),V_rec[:,300])
        plt.show()
        V_rec = V_rec *F
        print("Filtered final norm: " +str(np.linalg.norm(V_rec)))
        mse = np.inf
        for i in range(V_rec.shape[0]):
            tmp_mse = np.linalg.norm(np.roll(V_rec,i)-cyclic_signal)
            if tmp_mse<mse:
                mse = tmp_mse
        print("MSE: " +str(mse))
        mse_list.append(mse)
        iter_list.append(i*5)
    print(mse_list)
    plt.plot(range(V_rec.shape[0]), V_rec[:, 300])
    plt.title("Cyclic gene after filtering")
    plt.show()
    iter_ar = np.array(iter_list)
    mse_ar = np.array(mse_list)
    with open('iter_ar.npy', 'wb') as f:
        np.save(f,iter_ar)
    with open('mse_Ar.npy', 'wb') as f:
        np.save(f,mse_ar)
    plt.plot(iter_ar,mse_ar)
    plt.title("MSE as a function of number filtering iterations")
    plt.show()
    pca_3d(V_rec)
    ranged_pca_2d(V_rec,color=range(V.shape[0]),title="PCA recunstructed and filtered cyclic signal")
    # D = filter_non_linear_genes(V)
    pass


def simulated_disentanglement_2():
    V = simulate_spatial_1d(ngenes=500,ncells=250, w =0.2)
    np.random.shuffle(V)
    V[:,250:500]=np.zeros((V.shape[0],250))
    ranged_pca_2d(V,color=range(V.shape[0]),title="PCA of linear signal")
    A = simulate_spatial_cyclic_z(ngenes=500,ncells=250, w =0.2)
    A[:,0:250]=np.zeros((A.shape[0],250))
    ranged_pca_2d(A,color=range(A.shape[0]),title="PCA of cyclic signal")
    print("Cyclic starting norm: " +str(np.linalg.norm(A)))
    print("Linear starting norm: " +str(np.linalg.norm(V)))
    print(A.shape)
    linear_signal = copy.deepcopy(V)
    V[:,250:500]=A[:,250:500]
    noise = np.random.normal(0,0.25,V.shape)
    #noise = np.clip(noise,a_min=0 , a_max = np.inf)
    print("Noise starting norm: " +str(np.linalg.norm(noise)))
    V= V+ noise
    ranged_pca_2d(noise,color=range(noise.shape[0]),title="PCA of Gaussian noise")
    mse = np.inf
    for i in range(V.shape[0]):
        tmp_mse = np.linalg.norm(np.roll(V, i) - linear_signal)
        if tmp_mse < mse:
            mse = tmp_mse
    print("Starting MSE: " + str(mse))

    print("Total signal starting norm: " +str(np.linalg.norm(V)))
    #np.random.shuffle(V)
    ranged_pca_2d(V,color=range(V.shape[0]),title="PCA of simulated signal")
    #E , E_rec = sga_m_reorder_rows_matrix(V, iterNum=100, batch_size=400 , lr=0.1 , gama=0.9)
    #plt.imshow(E_rec)
    #plt.show()
    V_rec =V#E_rec.dot(V)
    plt.plot(range(V_rec.shape[0]), V_rec[:, 300])
    plt.title("Cyclic gene before filtering")
    plt.show()
    D = filter_cyclic_genes(V_rec,regu=0.001,iterNum=100)
    V_rec = V_rec.dot(D)
    plot_diag(D)
    pca_3d(V_rec)
    mse_list =  []
    iter_list = []
    for i in range(1):
        F = filter_cyclic_full(V_rec,regu=20,iterNum=250)
        plt.plot(range(V_rec.shape[0]),V_rec[:,300])
        plt.show()
        V_rec = V_rec *F
        print("Filtered final norm: " +str(np.linalg.norm(V_rec)))
        mse = np.inf
        for i in range(V_rec.shape[0]):
            tmp_mse = np.linalg.norm(np.roll(V_rec,i)-linear_signal)
            if tmp_mse<mse:
                mse = tmp_mse
        print("MSE: " +str(mse))
        mse_list.append(mse)
        iter_list.append(i*5)
    print(mse_list)
    plt.plot(range(V_rec.shape[0]), V_rec[:, 300])
    plt.title("Cyclic gene after filtering")
    plt.show()
    iter_ar = np.array(iter_list)
    mse_ar = np.array(mse_list)
    with open('iter_ar.npy', 'wb') as f:
        np.save(f,iter_ar)
    with open('mse_Ar.npy', 'wb') as f:
        np.save(f,mse_ar)
    plt.plot(iter_ar,mse_ar)
    plt.title("MSE as a function of number filtering iterations")
    plt.show()
    pca_3d(V_rec)
    ranged_pca_2d(V_rec,color=range(V.shape[0]),title="PCA reconstructed and filtered linear signal")
    # D = filter_non_linear_genes(V)
    pass

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

def simulate_noisy_linear(true_ngenes, false_ngenes, ncells, w, noise_var,
                          return_only_data=False):
    V = simulate_window_linear(true_ngenes, ncells, w)
    V_zeros = np.zeros((ncells, false_ngenes))
    V = np.concatenate((V, V_zeros), axis=1)
    A = simulate_spatial_cyclic_z(ngenes=false_ngenes, ncells=ncells,
                                  w=w)

    A_zeros = np.zeros((ncells, true_ngenes))
    A = np.concatenate((A_zeros, A), axis=1)
    np.random.shuffle(A)
    # pca_3d(V)
    # pca_3d(A)
    total_data = A + V
    noise = np.random.normal(0, noise_var, total_data.shape)
    total_data = total_data + noise
    # pca_3d(total_data)
    print(f'linear norm: {np.linalg.norm(V)}')
    print(f'cyclic norm: {np.linalg.norm(A)}')
    print(f'noise norm: {np.linalg.norm(noise)}')
    print(f'total norm: {np.linalg.norm(total_data)}')
    if return_only_data:
        return total_data
    return V, A, noise, total_data


