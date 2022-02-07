import pandas as pd
from sklearn.manifold import MDS , LocallyLinearEmbedding

from analysis import *
from datasets import *
import numpy as np
import circle_fit as cf
import numpy as np
from sklearn.datasets import make_circles
from scipy import stats
from datasets import *

def genes_score(data,path):
    '''
    :param data: scanpy annData
    :param path: path for csv file of contains list of relevant genes
    :return: sum of expression of those genes in the data
    '''
    gene_csv = pd.read_csv(path)
    cyclic_genes = []
    for i in gene_csv.values:
        cyclic_genes.append(str(i[0]).lower())
    cell_cycle_genes = [x for x in cyclic_genes if x in data.var_names]
    cell_cycle_genes = list(set(cell_cycle_genes))

    score=0
    for i in cell_cycle_genes:
        try:
            a = data[:,i].X
            score+= np.sum(data[:,i].X[:,0])
        except:
            a=1
            #print("Gene does not exist")
    return score


def get_unit_circle(noise=.1):
    data = make_circles(noise=noise, factor=.05)
    idx = np.argwhere(data[1]==0)
    x = data[0][idx, 0]
    y = data[0][idx, 1]
    return x, y

def lsc_score_pca(data):
    pca = PCA(n_components=2 )
    X_r = pca.fit(data).transform(data)
    plt.scatter(X_r[:,0], X_r[:,1])
    plt.show()
    points =[]
    for i in range(data.shape[0]):
        points.append([X_r[i,0],X_r[i,1]])
    xc, yc, r, s = cf.hyper_fit(points)
    min_dist=np.inf
    max_dist = -np.inf
    for point in points:
        dist = np.sqrt(np.power(xc-point[0],2) + np.power(yc-point[1],2))
        if dist< min_dist:
            min_dist=dist
        if dist>max_dist:
            max_dist = dist
    return max_dist-min_dist

def lsc_score_lle(data):
    lle = LocallyLinearEmbedding(n_components=2 , n_neighbors=10 )
    X_r = lle.fit(data).transform(data)
    plt.scatter(X_r[:,0], X_r[:,1])
    plt.show()
    points =[]
    for i in range(data.shape[0]):
        points.append([X_r[i,0],X_r[i,1]])
    xc, yc, r, s = cf.hyper_fit(points)
    min_dist=np.inf
    max_dist = -np.inf
    for point in points:
        dist = np.sqrt(np.power(xc-point[0],2) + np.power(yc-point[1],2))
        if dist< min_dist:
            min_dist=dist
        if dist>max_dist:
            max_dist = dist
    return max_dist-min_dist

def lsc_score_tsne(data):
    tsne = TSNE(n_components=2)
    X_r = tsne.fit_transform(data)
    plt.scatter(X_r[:,0], X_r[:,1])
    plt.show()
    points =[]
    for i in range(data.shape[0]):
        points.append([X_r[i,0],X_r[i,1]])
    xc, yc, r, s = cf.hyper_fit(points)
    min_dist=np.inf
    max_dist = -np.inf
    for point in points:
        dist = np.sqrt(np.power(xc-point[0],2) + np.power(yc-point[1],2))
        if dist< min_dist:
            min_dist=dist
        if dist>max_dist:
            max_dist = dist
    return max_dist-min_dist

def mds_score_tsne(data):
    mds = MDS(n_components=2)
    X_r = mds.fit_transform(data)
    plt.scatter(X_r[:,0], X_r[:,1])
    plt.show()
    points =[]
    for i in range(data.shape[0]):
        points.append([X_r[i,0],X_r[i,1]])
    xc, yc, r, s = cf.hyper_fit(points)
    min_dist=np.inf
    max_dist = -np.inf
    for point in points:
        dist = np.sqrt(np.power(xc-point[0],2) + np.power(yc-point[1],2))
        if dist< min_dist:
            min_dist=dist
        if dist>max_dist:
            max_dist = dist
    return max_dist-min_dist


def sort_data_linear_2(adata):
    perm = np.random.permutation(range(adata.X.shape[0]))
    adata = adata[perm,:]
    layers = [[] for i in range(8)]
    obs = adata.obs
    for i, row in obs.iterrows():
        layer = int(row['layer'])
        layers[layer].append(i)
    order = sum(layers, [])
    sorted_data = adata[order,:]
    return sorted_data

def corr_rank(adata,gene,direction):
    adata_sorted = sort_data_linear_2(copy.deepcopy(adata.copy()))
    layer = adata_sorted.obs['layer']
    layer = layer.to_numpy()
    sorted_gene = np.array(adata_sorted[:,gene].X)
    sorted_gene = sorted_gene[:,0]
    if direction=='up':
        return stats.spearmanr(sorted_gene,layer)
    else:
        return stats.spearmanr(sorted_gene,np.flip(layer))
