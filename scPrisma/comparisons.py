from analysis import *
from algorithms import *
from datasets import *
import rpy2.robjects as ro
import rpy2.robjects.numpy2ri



def chl_genes_vectoir(adata,genes_path):
    '''
    Create diurnal cycle related genes vector for ccRemover
    :param adata: Scanpy annData
    :param genes_path: path for gene list file
    :return: binary vector that indicates if a gene is related to the diurnal cycle
    '''
    gene_list = []
    df = pd.read_csv(genes_path, header=None)
    new_header = df.iloc[1]
    df = df[2:]
    df.columns = new_header
    df = df.dropna()

    for i in range(48):
        if i % 2 == 0:
            phase_genes = df.loc[df['phase'] == str(int(0.5 * i))]
        else:
            phase_genes = df.loc[df['phase'] == str(0.5 * i)]
        for j in phase_genes.values:
            gene_string = j[0] + ".v5.5"
            gene_list.append(gene_string)
    chl_genes = [x for x in gene_list if x in adata.var_names]
    if_cc = np.zeros((adata.X.shape[1],1))
    for i, gene in enumerate(adata.var_names):
        if gene in chl_genes:
            if_cc[i,0]=1
    # Take only the values of the dataframe
    return if_cc

def prepare_data_for_ccremover(adata_neg,adata_pos, genes_path="Chlamydomonas/ch_genes.csv"):
    '''
    Prepare data for reunning ccRemover for removing the diurnal cycle, save data for R
    :param adata_neg: annData of (FE-) chlamydomonas
    :param adata_pos: annData of (FE+) chlamydomonas
    :param genes_path: path for gene list file
    '''
    if_cc_neg = chl_genes_vectoir(adata_neg,genes_path)
    B_neg =np.array(adata_neg.X)
    B_neg = B_neg.T
    rpy2.robjects.numpy2ri.activate()
    nr, nc = B_neg.shape
    Br = ro.r.matrix(B_neg, nrow=nr, ncol=nc)
    ro.r.assign("B_neg", Br)
    ro.r("save(B_neg, file='chl_neg.Rdata')")
    if_cc_neg = rpy2.robjects.IntVector(if_cc_neg)
    ro.r.assign("if_cc_neg", if_cc_neg)
    ro.r("save(if_cc_neg, file='if_cc_neg.Rdata')")

    if_cc_pos = chl_genes_vectoir(adata_pos,genes_path)
    B_pos =np.array(adata_pos.X)
    B_pos = B_pos.T
    rpy2.robjects.numpy2ri.activate()
    nr, nc = B_pos.shape
    Br = ro.r.matrix(B_pos, nrow=nr, ncol=nc)
    ro.r.assign("B_pos", Br)
    ro.r("save(B_pos, file='chl_pos.Rdata')")
    if_cc_pos = rpy2.robjects.IntVector(if_cc_pos)
    ro.r.assign("if_cc_pos", if_cc_pos)
    ro.r("save(if_cc_pos, file='if_cc_pos.Rdata')")
    pass


def read_chl_gene_list (genes_path="Chlamydomonas/ch_genes.csv"):
    gene_list = []
    df = pd.read_csv(genes_path, header=None)
    new_header = df.iloc[1]
    df = df[2:]
    df.columns = new_header
    df = df.dropna()

    for i in range(48):
        if i % 2 == 0:
            phase_genes = df.loc[df['phase'] == str(int(0.5 * i))]
        else:
            phase_genes = df.loc[df['phase'] == str(0.5 * i)]
        for j in phase_genes.values:
            gene_string = j[0] + ".v5.5"
            gene_list.append(gene_string)
    return gene_list


def seurat_chl(adata_path_pos,adata_path_neg,genes_path="Chlamydomonas/ch_genes.csv"):
    adata_pos = sc.read(adata_path_pos)
    adata_neg = sc.read(adata_path_neg)
    sc.pp.normalize_per_cell(adata_neg, counts_per_cell_after=1e4)
    sc.pp.normalize_per_cell(adata_pos, counts_per_cell_after=1e4)
    sc.pp.log1p(adata_neg)
    sc.pp.log1p(adata_pos)
    sc.pp.scale(adata_neg)
    sc.pp.scale(adata_pos)
    adata_unit = adata_neg.concatenate(adata_pos)
    sc.tl.pca(adata_unit)
    sc.pl.pca(adata_unit,color="FE")

    print("Old norm pos: " + str(np.linalg.norm(adata_pos.X)))
    print("Old norm neg: " + str(np.linalg.norm(adata_neg.X)))

    gene_list = []
    tmp_hour = copy.deepcopy(adata_pos[:, 0].X)
    tmp_hour *= 0
    df = pd.read_csv(genes_path, header=None)
    new_header = df.iloc[1]
    df = df[2:]
    df.columns = new_header
    df = df.dropna()

    for i in range(48):
        if i % 2 == 0:
            phase_genes = df.loc[df['phase'] == str(int(0.5 * i))]
        else:
            phase_genes = df.loc[df['phase'] == str(0.5 * i)]
        for j in phase_genes.values:
            gene_string = j[0] + ".v5.5"
            gene_list.append(gene_string)
    chl_genes = [x for x in gene_list if x in adata_pos.var_names]
    #s_genes = chl_genes[:int(len(chl_genes)/3)]
    #g2m_genes = chl_genes[int(len(chl_genes)/3):int(2*len(chl_genes)/3)]
    s_genes = chl_genes[:int(len(chl_genes)/2)]
    g2m_genes = chl_genes[int(len(chl_genes)/2):]
    sc.tl.score_genes_cell_cycle(adata_pos, s_genes=s_genes, g2m_genes=g2m_genes)
    sc.pp.regress_out(adata_pos, ['S_score', 'G2M_score'])
    sc.tl.score_genes_cell_cycle(adata_neg, s_genes=s_genes, g2m_genes=g2m_genes)
    sc.pp.regress_out(adata_neg, ['S_score', 'G2M_score'])

    adata_unit = adata_neg.concatenate(adata_pos)
    labels_str = adata_unit.obs["FE"]
    labels = np.zeros(adata_unit.X.shape[0])
    sc.tl.pca(adata_unit)
    sc.pl.pca(adata_unit,color="FE")

    for j, i in enumerate(labels_str):
        if i == "Negative":
            labels[j] = 0
        else:
            labels[j] = 1
    print("New norm pos: " + str(np.linalg.norm(adata_pos.X)))
    print("New norm neg: " + str(np.linalg.norm(adata_neg.X)))
    # print(f1_score(labels,kmeans.labels_))
    # print(1 - f1_score(labels,kmeans.labels_))
    print("silhoutte score : " + str(silhouette_score(adata_unit.X, labels)))
    print("davies_bouldin_score score : " + str(davies_bouldin_score(adata_unit.X, labels)))
    print("calinski_harabasz_score before : " + str(calinski_harabasz_score(adata_unit.X, labels)))
    pass

def evaluate_ccRemover_output(pos_path='dat2.csv', adata_pos_path='chl_pos_reordered.h5ad',neg_path='dat_neg.csv',adata_neg_path='chl_neg_reordered.h5ad'):
    pos_X  = pd.read_csv(pos_path)#,delimiter='\t')
    pos_X = pos_X.to_numpy()
    adata_pos = sc.read(adata_pos_path)
    adata_pos.X = pos_X.T
    neg_X  = pd.read_csv(neg_path)#,delimiter='\t')
    neg_X = neg_X.to_numpy()
    adata_neg = sc.read(adata_neg_path)
    adata_neg.X = neg_X.T
    adata_unit = adata_neg.concatenate(adata_pos)
    labels_str = adata_unit.obs["FE"]
    labels = labels_str
    sc.tl.pca(adata_unit)
    sc.pl.pca(adata_unit,color="FE")
    print("New norm pos: " + str(np.linalg.norm(adata_pos.X)))
    print("New norm neg: " + str(np.linalg.norm(adata_neg.X)))
    print("silhoutte score : " + str(silhouette_score(adata_unit.X, labels)))
    print("davies_bouldin_score score : " + str(davies_bouldin_score(adata_unit.X, labels)))
    print("calinski_harabasz_score before : " + str(calinski_harabasz_score(adata_unit.X, labels)))
    pass
