import copy

import numpy as np
from numpy import random
from scipy.optimize.linesearch import line_search

from scPrisma.pre_processing import *
from numba import jit

@jit(nopython=True, parallel=True)
def numba_diagonal(A):
    d = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        d[i]=A[i,i]
    return d

@jit(nopython=True, parallel=True)
def reconstruct_e(E_prob):
    '''
    Greedy algorithm to reconstruct permutation matrix from Bi-Stochastic matrix
    :param E_prob: Bi-Stochastic matrix
    :return: Permutation matrix
    '''
    res = []
    for i in range(E_prob.shape[0]):
        tmp = -1
        pointer = -1
        for j in range(E_prob.shape[1]):
            if E_prob[i, j] > tmp and not (j in res):
                tmp = E_prob[i, j]
                pointer = j
        res.append(pointer)
    res_array = np.zeros(E_prob.shape)
    for i, item in enumerate(res):
        res_array[i, item] = 1
    return res_array

def ge_to_spectral_matrix(A , optimize_alpha=True):
    '''
    :param A: Gene expression matrix
    :return: Theoretic spectral matrix
    '''
    n=A.shape[0]
    p = A.shape[1]
    min_np = min(n,p)
    if optimize_alpha:
        u, s, vh = np.linalg.svd(A)
        for i in range(min_np):
            s[i] *= s[i]
        alpha = optimize_alpha_p(s, 15)
    else:
        alpha = np.exp(-2/p)
    V = generate_spectral_matrix(n=n, alpha=alpha)
    V = V[1:, :]
    return V

def sga_reorder_rows_matrix(A, iterNum=300, batch_size=20):
    '''
    :param A: gene expression matrix
    :param iterNum: iteration number
    :param batch_size: batch size
    :return: permutation matrix
    '''
    A = cell_normalization(A)
    n= A.shape[0]
    V =ge_to_spectral_matrix(A)
    E = sga_matrix(A, E=np.ones((n, n)) / n, V=V.T, iterNum=iterNum, batch_size=batch_size)
    E_recon = reconstruct_e(E)
    return E, E_recon


@jit(nopython=True, parallel=True)
def sga_matrix(A, E, V, iterNum=400, batch_size=20, lr=0.1):
    '''
    :param A: gene expression matrix
    :param E: permutation matrix initial value
    :param V: Eigenvectors matrix multiple by sqrt of diagonal eigenvalues matrix
    :param iterNum: iteration number
    :param batch_size: batch size
    :param lr: learning rate
    :return: permutation matrix
    '''
    j = 0
    value = 0
    epsilon_t = lr
    while (j < iterNum):
        if j % 25 == 0:
            print("Iteration number: " + str(j) + " function value= " + str(value))
            print("Iteration number: " + str(j))
        epsilon_t *= 0.995
        A_tmp = A[:, np.random.randint(A.shape[1], size=batch_size)]
        value, grad = fAndG_matrix(A=A_tmp, E=E, V=V)
        E = E + epsilon_t * grad
        E = BBS(E)
        print("Iteration number: " + str(j) + " function value= " + str(value))
        # elapsed = time.time() - t
        # print("projection elpased:" + str(elapsed))
        j += 1
    return E


@jit(nopython=True, parallel=True)
def fAndG_matrix(A, E, V):
    '''
    Calculate the function value and the gradient of A matrix
    :param A: gene expression matrix
    :param E: permutation matrix
    :param V: spectral matrix
    :return: function value, gradient of E
    '''
    functionValue = np.trace((((((V.T).dot(E)).dot(A)).dot(A.T)).dot(E.T)).dot(V))
    gradient = (2 * ((((V).dot(V.T)).dot(E)).dot(A)).dot(A.T))

    return functionValue, gradient

@jit(nopython=True, parallel=True)
def G_matrix(A, E, VVT):
    '''
    Calculate the gradient of A matrix
    :param A: gene expression matrix
    :param E: permutation matrix
    :param V: spectral matrix
    :return: gradient of E
    '''
    gradient = (2 * (((VVT).dot(E)).dot(A)).dot(A.T))
    return gradient

@jit(nopython=True, parallel=True)
def sga_matrix_momentum(A, E, V, iterNum=400, batch_size=20, lr=0.1, gama=0.9 , verbose=True):
    '''
    :param A: gene expression matrix
    :param E: permutation matrix initial value
    :param V: Eigenvectors matrix multiple by sqrt of diagonal eigenvalues matrix
    :param iterNum: iteration number
    :param batch_size: batch size
    :param lr: learning rate
    :param gama: momentum parameter
    :return: permutation matrix
    '''
    j = 0
    value = 0
    VVT = (V).dot(V.T) #for runtime optimization
    epsilon_t = lr
    step = np.zeros(E.shape)
    while (j < iterNum ):
        if (j % 25 == 0) & verbose:
            value, grad = fAndG_matrix(A=A, E=E, V=V)
            print("Iteration number: ")
            print(j)
            print(" function value= ")
            print(value)
        A_tmp = A[:, np.random.randint(0,A.shape[1], batch_size)]
        grad = G_matrix(A=A_tmp, E=E, VVT=VVT)
        step = epsilon_t * grad + gama * step
        E = E + step
        E = BBS(E)
        j += 1
    return E

def sga_matrix_momentum_indicator(A, E, V,IN, iterNum=400, batch_size=20, lr=0.1, gama=0.9):
    '''
    :param A: gene expression matrix
    :param E: permutation matrix initial value
    :param V: Eigenvectors matrix multiple by sqrt of diagonal eigenvalues matrix
    :param iterNum: iteration number
    :param batch_size: batch size
    :param lr: learning rate
    :param gama: momentum parameter
    :return: permutation matrix
    '''
    j = 0
    value = 0
    epsilon_t = lr
    step = np.zeros(E.shape)
    E = E * IN
    E = BBS(E) * IN
    while (j < iterNum):
        if j % 25 == 0:
            print("Iteration number: " + str(j) + " function value= " + str(value))
        A_tmp = A[:, np.random.randint(A.shape[1], size=batch_size)]
        value, grad = fAndG_matrix(A=A_tmp, E=E, V=V)
        grad = grad
        step = epsilon_t * grad + gama * step
        E = E + step
        E = BBS(E) * IN
        j += 1
    return E

def sga_m_reorder_rows_matrix(A, iterNum=300, batch_size=None, gama=0.5, lr=0.1):
    '''
    Cyclic reorder rows using stochastic gradient ascent
    :param A: gene expression matrix
    :param iterNum:  iteration number
    :param batch_size: batch size
    :param gama: momentum parameter
    :return: permutation matrix
    '''
    if batch_size==None:
        batch_size= int((A.shape[0])*0.75)
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V = ge_to_spectral_matrix(A)
    E = sga_matrix_momentum(A, E=np.ones((n, n)) / n, V=V.T, iterNum=iterNum, batch_size=batch_size, gama=gama, lr=lr)
    E_recon = reconstruct_e(E)
    return E, E_recon

def reconstruction_cyclic(A, iterNum=300, batch_size=None, gama=0.5, lr=0.1 , verbose=True , final_loss=False):
    '''
    Cyclic reorder rows using stochastic gradient ascent
    :param A: gene expression matrix
    :param iterNum:  iteration number
    :param batch_size: batch size
    :param gama: momentum parameter
    :return: permutation matrix
    '''
    A = np.array(A).astype('float64')
    if batch_size==None:
        batch_size= int((A.shape[0])*0.75)
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V = ge_to_spectral_matrix(A)
    E = sga_matrix_momentum(A, E=np.ones((n, n)) / n, V=V.T, iterNum=iterNum, batch_size=batch_size, gama=gama, lr=lr , verbose=verbose)
    E_recon = reconstruct_e(E)
    if final_loss:
        value, grad = fAndG_matrix(A=((1/A.shape[0])*A), E=E_recon, V=V.T)
        return E, E_recon , value
    return E, E_recon

def filter_non_cyclic_genes(A, regu=0.1, lr=0.1, iterNum=500):
    '''
    :param A: gene expression matrix
    :param regu: regularization parameter
    :param iterNum: iteration number
    :param lr: learning rate
    :return: diagonal filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    U = ge_to_spectral_matrix(A)
    A = gene_normalization(A)
    D = gradient_ascent_filter_matrix(A, D=np.identity((p))/2, U=U.T, regu=regu, lr=lr, iterNum=iterNum)
    return D


def filter_cyclic_genes(A, regu=0.1, iterNum=500, lr=0.1):
    '''
    :param A: gene expression matrix
    :param regu: regularization parameter
    :param iterNum: iteration number
    :param lr: learning rate
    :return: diagonal filtering matrix
    '''
    A = np.array(A).astype('float64')
    V = cell_normalization(A)
    p = V.shape[1]
    U =ge_to_spectral_matrix(V)
    A = gene_normalization(A)
    D = gradient_ascent_filter_matrix(A, D=np.identity((p))/2, ascent=-1, U=U.T, regu=regu, iterNum=iterNum, lr=lr)
    return D

def filter_cyclic_genes_line(A, regu=0.1, iterNum=500, lr=0.1 , verbosity=25):
    '''
    :param A: gene expression matrix
    :param regu: regularization parameter
    :param iterNum: iteration number
    :param lr: learning rate
    :return: diagonal filtering matrix
    '''
    A = np.array(A).astype('float64')
    V = cell_normalization(A)
    p = V.shape[1]
    U =ge_to_spectral_matrix(V)
    A = gene_normalization(A)
    D = gradient_descent_filter_matrix_line(A, D=np.identity((p)), U=U.T, regu=regu , max_evals=iterNum,verbosity=verbosity)
    return D

def filter_linear_padded_genes_line(A, regu=0.1, iterNum=500, lr=0.1 , verbosity=25):
    '''
    :param A: gene expression matrix
    :param regu: regularization parameter
    :param iterNum: iteration number
    :param lr: learning rate
    :return: diagonal filtering matrix
    '''
    Padded_array = np.zeros((int(A.shape[0]/2),A.shape[1]))
    V = np.concatenate([A,Padded_array])
    V = cell_normalization(V)
    p = V.shape[1]
    U =ge_to_spectral_matrix(V)
    V = gene_normalization(V)
    D = gradient_ascent_filter_matrix(V, D=np.identity((p))/2, ascent=-1, U=U.T, regu=regu, iterNum=iterNum, lr=lr)
    return D

def filter_linear_genes_line(A, regu=0.1, iterNum=500, lr=0.1 , verbosity=25 , method='numeric'):
    '''
    :param A: gene expression matrix
    :param regu: regularization parameter
    :param iterNum: iteration number
    :param lr: learning rate
    :return: diagonal filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    p = A.shape[1]
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(optimized_alpha=True, eigenvals=real_eigenvalues)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    A = gene_normalization(A)
    D = gradient_descent_filter_matrix_line(A, D=np.identity((p)), U=eigenvectors[:, 1:], regu=regu , max_evals=iterNum,verbosity=verbosity)
    return D

def filter_non_cyclic_genes_line(A, regu=0.1, iterNum=500, lr=0.1, verbosity=25):
    '''
    :param A: gene expression matrix
    :param regu: regularization parameter
    :param iterNum: iteration number
    :param lr: learning rate
    :return: diagonal filtering matrix
    '''
    A = np.array(A).astype('float64')
    V = cell_normalization(A)
    p = V.shape[1]
    U =ge_to_spectral_matrix(V)
    A = gene_normalization(A)
    D = gradient_descent_filter_matrix_line(A, D=np.identity((p)), U=U.T, regu=regu , max_evals=iterNum , verbosity=verbosity)
    np.identity(D.shape[1])
    return (np.identity(D.shape[1]) - D)

@jit(nopython=True, parallel=True)
def gradient_ascent_filter_matrix(A, D, U, ascent=1, lr=0.1, regu=0.1, iterNum=400):
    '''
    :param A: gene expression matrix
    :param D: diagonal filter matrix (initial value)
    :param U: Eigenvectors matrix multiple by sqrt of diagonal eigenvalues matrix
    :param ascent: 1 - gradient ascent , -1 - gradient decent
    :param lr: learning rate
    :param regu: regularization parameter
    :param iterNum:  iteration number
    :return: diagonal filter matrix
    '''
    j = 0
    val = 0
    epsilon_t = lr
    ATUUTA = (2 * ((((A.T).dot(U)).dot(U.T)).dot(A)))#.dot(D)) - regu*np.sign(D)#((1 / t_0) * ((A.T).dot(A)).dot(D))
    while (j < iterNum):
        if j % 25 == 1:
            # value = (np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - (regu * np.linalg.norm((A).dot(D), 'fro')))
            val = np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - regu*np.linalg.norm(D,1)
            print("Iteration number: ")
            print(j)
            print( "function value= ")
            print(val)
            #plot_diag(D)
            #im = plt.imshow(D)
            #plt.colorbar()
            #plt.show()
        epsilon_t *= 0.995
        T = numba_diagonal(D)#.diagonal()
        grad = ATUUTA * T - regu * np.sign(D)
        D = D + ascent* epsilon_t * grad
        D = diag_projection(D)
        # print("Iteration number: " + str(j) + " grad value= " + str(grad))
        j += 1
    return D



#def loss_filter_genes(A,U,D,regu):
#    return np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - regu*np.linalg.norm(D,1)

@jit(nopython=True, parallel=True)
def loss_filter_genes(ATU,D,regu):
    D_diag = numba_diagonal(D)
    return np.trace((ATU.T * D_diag * D_diag).dot(ATU)) - regu*np.linalg.norm(D,1)

@jit(nopython=True, parallel=True)
def gradient_descent_filter_matrix_line(A, D, U, regu=0.1, gamma = 1e-04, max_evals = 250, verbosity = float('inf')):
    '''
    :param A: ene expression matrix
    :param D: diagonal filter matrix (initial value)
    :param U: Eigenvectors matrix multiple by sqrt of diagonal eigenvalues matrix
    :param ascent: 1 - gradient ascent , -1 - gradient decent
    :param lr: learning rate
    :param regu: regularization parameter
    :param iterNum:  iteration number
    :return: diagonal filter matrix
    '''
    ATUUTA = (2 * ((((A.T).dot(U)).dot(U.T)).dot(A)))
    w = D
    evals = 0
    ATU = (A.T).dot(U)
    loss = loss_filter_genes(ATU=ATU,D=w,regu=regu)
    w = diag_projection(w)

    grad = ATUUTA * w - regu * np.sign(w)
    G = numba_diagonal(grad)#.diagonal()
    grad = np.diag(G)
    alpha = 1 / np.linalg.norm(grad)
    while evals < max_evals and np.linalg.norm(grad) > 1e-07:
        evals += 1
        if evals % verbosity == 0:
            print((evals))
            print('th Iteration    Loss :: ')
            print((loss))
            print(' gradient :: ')
            print((np.linalg.norm(grad)))
        gTg  = np.linalg.norm(grad)
        gTg = gTg*gTg
        new_w = w - alpha * grad
        new_loss = loss_filter_genes(ATU, new_w, regu)
        new_w = diag_projection(new_w)
        new_grad = ATUUTA * new_w - regu * np.sign(new_w)
        G = numba_diagonal(new_grad)#.diagonal()
        new_grad = np.diag(G)
        while new_loss > loss - gamma * alpha * gTg:
            alpha = ((alpha ** 2) * gTg) / (2 * (new_loss + alpha * gTg - loss))
            new_w = w - alpha * grad
            new_loss = loss_filter_genes(ATU, new_w, regu)
            new_w = diag_projection(new_w)
            new_grad = ATUUTA * new_w - regu * np.sign(new_w)
            G = numba_diagonal(new_grad)#.diagonal()
            new_grad = np.diag(G)

        alpha = min(1, 2 * (loss - new_loss) / gTg)
        loss = new_loss
        grad = new_grad
        w = new_w
    return w

@jit(nopython=True, parallel=True)
def fAndG_filter_matrix(A, D, U, alpha):
    functionValue = (np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - (alpha * np.sum(np.abs(D))))
    gradient = ((2 * ((((A.T).dot(U)).dot(U.T)).dot(A)).dot(D)) - (alpha * np.sign(D)))
    return gradient, functionValue

@jit(nopython=True, parallel=True)
def fAndG_fixed_filter(A, D, U, regu):
    t_0 = np.linalg.norm((A).dot(D), 'fro')
    functionValue = np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - regu * t_0
    gradient = (2 * ((((A.T).dot(U)).dot(U.T)).dot(A)).dot(D)) - regu*np.sign(D)#((1 / t_0) * ((A.T).dot(A)).dot(D))

    return functionValue, gradient

@jit(nopython=True, parallel=True)
def gradient_ascent_filter(A, D, eigenvectors_list, eigenvalues_list, epsilon=0.1, regu=0.1, iterNum=400):
    '''
    :param A: Gene expression matrix
    :param D: diagonal filtering matrix
    :param eigenvectors_list:
    :param eigenvalues_list:
    :param epsilon:
    :param regu:
    :param iterNum:
    :return:
    '''
    # print(eigenvectors_list.shape)
    j = 0
    epsilon_t = epsilon
    while (j < iterNum):
        value = 0
        if j % 25 == 0:
            print("Iteration number: " + str(j))
        epsilon_t *= 0.995
        grad = np.zeros(D.shape)
        for i, v in enumerate(eigenvectors_list):
            tmp_value, tmp_grad = fAndG_regu(A=A, E=D, alpha=regu, x=v * eigenvalues_list[i])
            grad += tmp_grad
            value += tmp_value
        D = D + epsilon_t * grad
        D = diag_projection(D)
        print("Iteration number: " + str(j) + " function value= " + str(value))
        j += 1
    return D

@jit(nopython=True)
def diag_projection(D):
    T = numba_diagonal(D)#.diagonal()
    T = numba_vec_clip(T,len(T),0,1)
    return np.diag(T)


@jit(nopython=True, parallel=True)
def fAndG_regu(A, E, alpha, x):
    t_0 = (A.T).dot(x)
    t_1 = np.linalg.norm(E, 'fro')
    functionValue = ((x).dot((A).dot((E).dot((E.T).dot(t_0)))) - (alpha * t_1))
    gradient = ((2 * np.multiply.outer(t_0, ((x).dot(A)).dot(E))) - ((alpha / t_1) * E))
    return functionValue, gradient


def calculate_roc_auc(y_target, y_true):
    return roc_auc_score(y_true, y_target)


def filter_full(A, regu=0.1, iterNum=300):
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V =ge_to_spectral_matrix(A)
    #F = gradient_ascent_full(A, F=np.ones(A.shape), V=V.T, regu=regu, iterNum=iterNum)
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=V.T, regu=regu, iterNum=iterNum)
    return F

def enhancement_cyclic_per_gene(A, regu=0.1, iterNum=300):
    ''' Enhancement of cyclic signal
    :param A: Gene expression matrix (reordered according to cyclic ordering)
    :param regu: regularization coefficient
    :param iterNum: iteration number
    :return: filtering matrix
    '''
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V =ge_to_spectral_matrix(A)
    F = np.ones(A.shape)
    for i in range(p):
        F[:,i] = stochastic_gradient_ascent_full(A[:,i], F=np.ones(A.shape[0]), V=V.T, regu=regu, iterNum=iterNum)
    return F

def enhancement_cyclic(A, regu=0.1, iterNum=300 , verbosity=25):
    ''' Enhancement of cyclic signal
    :param A: Gene expression matrix (reordered according to cyclic ordering)
    :param regu: regularization coefficient
    :param iterNum: iteration number
    :return: filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    V =ge_to_spectral_matrix(A)
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=V.T, regu=regu, iterNum=iterNum , verbosity=verbosity)
    return F

def filtering_cyclic(A, regu=0.1, iterNum=300, verbosity = 25 , error=10e-7, optimize_alpha=True, line_search=True):
    ''' Filtering of cyclic signal
    :param A: Gene expression matrix (reordered according to cyclic ordering)
    :param regu: regularization coefficient
    :param iterNum: iteration number
    :return: filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A , optimize_alpha=optimize_alpha)
    print("starting filtering")
    if line_search:
        F = gradient_descent_full_line(A, F=np.ones(A.shape), V=V.T, regu=regu, max_evals=iterNum,verbosity=verbosity , error=error)
    else:
        F = gradient_descent_full(A,np.ones(A.shape),V=V.T,regu=regu,epsilon=0.1,iterNum=iterNum)
    return F

def filtering_cyclic_boosted(A, regu=0.1, iterNum=300, verbosity = 25 , error=10e-7):
    ''' Filtering of cyclic signal
    :param A: Gene expression matrix (reordered according to cyclic ordering)
    :param regu: regularization coefficient
    :param iterNum: iteration number
    :return: filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V = ge_to_spectral_matrix(A)
    print("starting filtering")
    A = gradient_descent_full_line_boosted(A, V=V.T, regu=regu, max_evals=iterNum,verbosity=verbosity , error=error)
    return A


#def filter_cyclic_full(A, regu=0.1, iterNum=300):
#    A = cell_normalization(A)
#    V = ge_to_spectral_matrix(A)
#    F = gradient_descent_full(A, F=np.ones(A.shape), V=V.T, regu=regu, iterNum=iterNum)
#    return F

def filter_cyclic_full_line(A, regu=0.1, iterNum=300, verbosity = 25 , error=10e-7):
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A)
    print("starting filtering")
    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=V.T, regu=regu, max_evals=iterNum,verbosity=verbosity , error=error)
    return F

def filter_non_cyclic_full_reverse(A, regu=0.1, iterNum=300):
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A)
    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=V.T, regu=regu, max_evals=iterNum)
    F = np.ones(F.shape)-F
    return F

@jit(nopython=True, parallel=True)
def gradient_ascent_full(A, F, V, regu, epsilon=0.1, iterNum=400):
    j = 0
    epsilon_t = epsilon
    while (j < iterNum):
        value = 0
        if j % 50 == 1:
            print("Iteration number: " + str(j))
        epsilon_t *= 0.995
        tmp_value, grad = fAndG_full(A=A, B=F, V=V, alpha=regu)
        F = F + epsilon_t * grad
        F = numba_clip(F,F.shape[0],F.shape[1],0,1)
        j += 1
    return F

@jit(nopython=True, parallel=True)
def stochastic_gradient_ascent_full(A, F, V, regu, epsilon=0.1, iterNum=400 , regu_norm='L1' , verbosity=25):
    '''
    :param A: gene expression matrix
    :param F: filtering matrix
    :param V: theoretic spectrum of covariance matrix
    :param regu: regularization coefficient
    :param epsilon: step size (learning rate)
    :param iterNum: iteration number
    :return: filtering matrix
    '''
    #print(A.shape)
    #print(F.shape)
    #print(V.shape)
    j = 0
    epsilon_t = epsilon
    VVT = V.dot(V.T)
    while (j < iterNum):
        if j % verbosity == 1:
            value, grad = fAndG_full_acc(A=A, B=F, V=V, VVT=VVT, alpha=regu, regu_norm=regu_norm)
            print("Iteration number: ")
            print((j))
            print("function value: ")
            print((value) )
        epsilon_t *= 0.995
        grad = G_full(A=A, B=F, V=V, alpha=regu , regu_norm = regu_norm)
        F = F + epsilon_t * (grad +np.random.normal(0,0.01,grad.shape))
        F = numba_clip(F,F.shape[0],F.shape[1],0,1)
        j += 1
    return F

@jit(nopython=True, parallel=True)
def gradient_descent_full(A, F, V, regu, epsilon=0.1, iterNum=400 , regu_norm ='L1'):
    print(A.shape)
    print(F.shape)
    print(V.shape)
    j = 0
    epsilon_t = epsilon
    while j < iterNum:
        if j % 100 == 1:
            print("Iteration number: " + str(j))
        epsilon_t *= 0.995
        tmp_value, grad = fAndG_full(A=A, B=F, V=V, alpha=regu, regu_norm=regu_norm)
        F = F - epsilon_t * grad
        F = numba_clip(F,F.shape[0],F.shape[1],0,1)
        j += 1
    return F


@jit(nopython=True, parallel=True)
def gradient_descent_full_line(A,F,V,regu, gamma = 1e-04,
                               max_evals = 250,
                               verbosity = float('inf'),error=1e-07, regu_norm='L1'):
    '''
    :param A:
    :param F:
    :param V:
    :param regu:
    :param gamma:
    :param max_evals:
    :param verbosity:
    :param error:
    :return:
    '''
    VVT = V.dot(V.T)
    w = F
    evals = 0
    loss, grad = fAndG_full_acc(A=A, B=F, V=V,VVT=VVT, alpha=regu, regu_norm=regu_norm)
    alpha = 1 / np.linalg.norm(grad)
    #alpha=0.1
    prev_w = np.zeros(w.shape)
    while evals < max_evals and np.linalg.norm(w-prev_w) > error:
        prev_w=np.copy(w)
        evals += 1
        if evals % verbosity == 0:
            print(str(evals))
            print('th Iteration    Loss :: ')
            print((loss))
            #+ ' gradient :: ' +  str(np.linalg.norm(grad)))
        gTg  = np.linalg.norm(grad)
        gTg = gTg*gTg
        new_w = w - alpha * grad
        new_w = numba_clip(new_w,new_w.shape[0],new_w.shape[1],0,1)
        #new_w = new_w.clip(min=0, max=1)
        new_loss, new_grad = fAndG_full_acc(A=A, B=new_w, V=V,
                                            VVT=VVT,alpha=regu, regu_norm=regu_norm)
        while new_loss > loss - gamma * alpha * gTg:
            alpha = ((alpha ** 2) * gTg) / (2 * (new_loss + alpha * gTg - loss))
            new_w = w - alpha * grad
            new_w = numba_clip(new_w, new_w.shape[0], new_w.shape[1], 0, 1)
            #new_w = new_w.clip(min=0, max=1)
            new_loss, new_grad = fAndG_full_acc(A=A, B=new_w, V=V,VVT=VVT,
                                                alpha=regu, regu_norm=regu_norm)
        alpha = min(1, 2 * (loss - new_loss) / gTg)
        loss = new_loss
        grad = new_grad
        w = new_w
    return w

@jit(nopython=True, parallel=True)
def fAndG_filtering_boosted(A, V, alpha_matrix):
    functionValue = np.trace((((V.T).dot(A)).dot(A.T)).dot(V)) - np.linalg.norm(alpha_matrix*A,1)
    gradient = (2 * ((V).dot(V.T)).dot(A)) - alpha_matrix*np.sign(A)
    return functionValue, gradient


@jit(nopython=True, parallel=True)
def gradient_descent_full_line_boosted(A,V,regu, gamma = 1e-04, max_evals = 250, verbosity = float('inf'),error=1e-07):
    '''
    :param A:
    :param F:
    :param V:
    :param regu:
    :param gamma:
    :param max_evals:
    :param verbosity:
    :param error:
    :return:
    '''

    alpha_matrix = (regu/(A+10e-5))/(A+10e-5)
    tmp_A = copy.deepcopy(A)
    VVT = V.dot(V.T)
    w = A
    evals = 0
    loss, grad = fAndG_filtering_boosted(A, V,alpha_matrix)
    alpha = 1 / np.linalg.norm(grad)
    #alpha=0.1
    prev_w = np.zeros(w.shape)
    while evals < max_evals and np.linalg.norm(w-prev_w) > error:
        prev_w=copy.deepcopy(w)
        evals += 1
        if evals % verbosity == 0:
            print(str(evals) + 'th Iteration    Loss :: ' + str(loss) + ' gradient :: ' +  str(np.linalg.norm(grad)))
        gTg  = np.linalg.norm(grad)
        gTg = gTg*gTg
        new_w = w - alpha * grad
        new_w = new_w.clip(min=0, max=tmp_A)
        new_loss, new_grad = fAndG_filtering_boosted(A=new_w,  V=V, alpha_matrix=alpha_matrix)
        while new_loss > loss - gamma * alpha * gTg:
            alpha = ((alpha ** 2) * gTg) / (2 * (new_loss + alpha * gTg - loss))
            new_w = w - alpha * grad
            new_w = new_w.clip(min=0, max=tmp_A)
            new_loss, new_grad = fAndG_filtering_boosted(A=new_w, V=V,alpha_matrix=alpha_matrix)
        alpha = min(1, 2 * (loss - new_loss) / gTg)
        loss = new_loss
        grad = new_grad
        w = new_w
    return w
@jit(nopython=True, parallel=True)
def fAndG_full(A, B, V, alpha, regu_norm='L2'):
    '''
    :param A: Gene expression matrix
    :param B: filtering matrix
    :param V: spectral matrix
    :param alpha: correlation between neighbors
    :return:projection over theoretic spectrum and gradient according to 'B'
    '''

    if regu_norm=='L1':
        T_0 = (A * B)
        t_1 = np.linalg.norm(B, 1)
        functionValue = (np.trace((((V.T).dot(T_0)).dot(T_0.T)).dot(V)) - (alpha * t_1))
        gradient = ((2 * (((V).dot(V.T)).dot(T_0) * A)) - ((alpha ) * np.sign(B)))
    else:
        T_0 = (A * B)
        t_1 = np.linalg.norm(A * B, 'fro')
        functionValue = (np.trace((((V.T).dot(T_0)).dot(T_0.T)).dot(V)) - (alpha * t_1))
        gradient = ((2 * (((V).dot(V.T)).dot(T_0) * A))- ((alpha / t_1) * B))
    return functionValue, gradient

@jit(nopython=True, parallel=True)
def fAndG_full_acc(A, B, V, VVT, alpha,regu_norm):
    '''
    :param A: Gene expression matrix
    :param B: filtering matrix
    :param V: spectral matrix
    :param VVT: V.dot(V.T)
    :param alpha: correlation between neighbors
    :param regu_norm: regularization norm (L1/L2)
    :return:projection over theoretic spectrum and gradient according to 'B'
    '''
    if regu_norm=='L1':
        T_0 = (A * B)
        t_1 = np.linalg.norm(B, 1)
        functionValue = (np.trace((((V.T).dot(T_0)).dot(T_0.T)).dot(V)) - (alpha * t_1))
        gradient = ((2 * ((VVT).dot(T_0) * A)) - ((alpha ) * np.sign(B)))
    else:
        T_0 = (A * B)
        t_1 = np.linalg.norm(A * B, 'fro')
        functionValue = (np.trace((((V.T).dot(T_0)).dot(T_0.T)).dot(V)) - (alpha * t_1))
        gradient = ((2 * ((VVT).dot(T_0) * A)) - ((alpha / t_1) * B))
    return functionValue, gradient

@jit(nopython=True, parallel=True)
def G_full(A, B, V, alpha, regu_norm='L1'):
    '''
    :param A: Gene expression matrix
    :param B: filtering matrix
    :param V: spectral matrix
    :param alpha: correlation between neighbors
    :return:projection over theoretic spectrum and gradient according to 'B'
    '''
    if regu_norm=='L1':
        T_0 = (A * B)
        gradient = ((2 * (((V).dot(V.T)).dot(T_0) * A)) - ((alpha) * np.sign(B)))
        functionValue = (np.trace((((V.T).dot(T_0)).dot(T_0.T)).dot(V)) - (alpha * np.linalg.norm(B,1)))
    else:
        T_0 = (A * B)
        gradient = ((2 * (((V).dot(V.T)).dot(T_0) * A)) - (alpha) * 2*B)
    return gradient


def sga_m_linear_reorder_rows_matrix(A, iterNum=1000, batch_size=400):
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    # u, s, vh = np.linalg.svd(A)
    np1 = min(n, p)
    # for i in range(np1):
    #    s[i]*=s[i]
    # alpha = optimize_alpha_p(s,15)
    # V, eigenvalues = get_numeric_eigen_values(get_alpha(A.shape[0], 10), n)
    eigenvalues, V = linalg.eig(A.dot(A.T))
    V, eigenvalues = get_psuedo_data(A.shape[1], A.shape[0], 10)
    for i in range(A.shape[0]):
        V[:, i] = V[:, i] * (eigenvalues[i])
    E = sga_matrix_momentum(A, E=np.ones((n, n)) / n, V=V[:, 1:], iterNum=iterNum, batch_size=batch_size)
    E_recon = reconstruct_e(E)
    return E, E_recon


def filter_non_cyclic_genes_vector(A, alpha=0.99, regu=2, iterNum=500):
    A = np.array(A).astype('float64')
    A = gene_normalization(A)
    n = A.shape[0]
    p = A.shape[1]

    eigenvectors, eigenvalues = generate_eigenvectors_circulant()

    D = gradient_ascent_filter(A, D=np.identity((p)), eigenvectors_list=eigenvectors[1:],
                               eigenvalues_list=eigenvalues[1:], regu=regu, iterNum=iterNum)
    return D


def filter_linear_full(A, method, regu=0.1, iterNum=300 , lr=0.1, regu_norm='L1'):
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(ngenes=A.shape[1], optimized_alpha=False, eigenvals=real_eigenvalues)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    F = gradient_descent_full(A, F=np.ones(A.shape), V=eigenvectors[:,1:], regu=regu,
                             iterNum=iterNum, epsilon=lr, regu_norm=regu_norm)
    return F

def enhancement_linear(A, regu=0.1, iterNum=300 , method='numeric'):
    ''' Enhancement of linear signal
    :param A: Gene expression matrix (reordered according to linear ordering)
    :param regu: regularization coefficient
    :param iterNum: iteration number
    :return: filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(optimized_alpha=True, eigenvals=real_eigenvalues)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=eigenvectors[:,1:], regu=regu, iterNum=iterNum)
    return F

def filtering_linear(A, method,regu=0.1, iterNum=300, verbosity = 25 ,
                     error=10e-7, optimized_alpha=True, regu_norm='L1'):
    ''' Filtering of linear signal
    :param A: Gene expression matrix (reordered according to linear ordering)
    :param regu: regularization coefficient
    :param iterNumenhance_linear_full: iteration number
    :return: filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    print(A.shape)
    alpha = get_alpha(ngenes=A.shape[1], optimized_alpha=optimized_alpha, eigenvals=real_eigenvalues)
    print(alpha)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=eigenvectors[:,1:],
                                   regu=regu, max_evals=iterNum,verbosity=verbosity ,
                                   error=error , regu_norm=regu_norm)
    return F
#    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=eigenvectors[:,1:],
#                                   regu=regu, max_evals=iterNum,verbosity=verbosity ,
#                                   error=error , regu_norm=regu_norm)

#

def enhance_linear_genes(A, method, regu=2, iterNum=500, lr=0.1):
    A = np.array(A).astype('float64')
    A = gene_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(optimized_alpha=True, eigenvals=real_eigenvalues)
    n = A.shape[0]
    p = A.shape[1]
    eigenvectors = get_linear_eig_data(n, alpha, method=method,
                                       normalize_vectors=True)
    D = gradient_ascent_filter_matrix(A, D=np.identity((p)) / 2,
                                      U=eigenvectors[:, 1:], regu=regu,
                                      iterNum=iterNum, lr=lr)
    return D


def filter_linear_genes(A, method='numeric', regu=2, iterNum=500, lr=0.1):
    A = np.array(A).astype('float64')
    A = gene_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(optimized_alpha=True, eigenvals=real_eigenvalues)
    n = A.shape[0]
    p = A.shape[1]
    eigenvectors = get_linear_eig_data(n, alpha, method=method,
                                       normalize_vectors=True)
    D = gradient_ascent_filter_matrix(A, D=np.identity((p)) / 2,
                                      U=eigenvectors[:, 1:], regu=regu,
                                      iterNum=iterNum, lr=lr, ascent=-1)
    return D

@jit(nopython=True)
def numba_min_clip(A,n,m,a_min):
    '''
    Implementing np.clip using numba
    :param A: Array
    :param n: number of rows
    :param m: number of columns
    :param a_min: a_min
    :return: clipped array
    '''
    for i in range(n):
        for j in range(m):
            if A[i,j]<a_min:
                A[i,j]=a_min
    return A

@jit(nopython=True)
def numba_clip(A,n,m,a_min,a_max):
    '''
    Implementing np.clip using numba
    :param A: Array
    :param n: number of rows
    :param m: number of columns
    :param a_min: a_min
    :param a_max: a_max
    :return: clipped array
    '''
    for i in range(n):
        for j in range(m):
            if A[i,j]<a_min:
                A[i,j]=a_min
            elif A[i,j]>a_max:
                A[i, j] = a_max
    return A


@jit(nopython=True)
def numba_vec_clip(v,n,a_min,a_max):
    '''
    Implementing np.clip using numba for vectors
    :param A: Array
    :param n: number of entries
    :param a_min: a_min
    :param a_max: a_max
    :return: clipped vector
    '''
    for i in range(n):
        if v[i]<a_min:
            v[i]=a_min
        elif v[i]>a_max:
            v[i] = a_max
    return v

@jit(nopython=True, parallel=True)
def BBS(E, iterNum=1000):
    ''' Bregmanian Bi-Stochastication algorithm as described inL
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/KAIS_BBS_final.pdf
    :param E: permutation matrix
    :param iterNum:iteration number
    :return:Bi-Stochastic matrix
    '''
    n = E.shape[0]
    prev_E = np.empty(E.shape)
    I = np.identity(n)
    # print(ones)
    ones_m = np.ones((n, n))
    # print(ones_m)
    for i in range(iterNum):
        if i % 15 == 1:
            prev_E = np.copy(E)
        ones_E = ones_m.dot(E)
        # print(ones_E)
        #E = E + (1 / n) * (I - E + (1 / n) * (ones_E)).dot(ones_m) - (1 / n) * ones_E
        E = E + (1 / n) * (I - E + (1 / n) * (ones_E)).dot(ones_m) - (1 / n) * ones_E
        E = numba_min_clip(E,E.shape[0],E.shape[0],0)
        if i % 15 == 1:
            if np.linalg.norm(E - prev_E) < ((10e-6) * n):
                break
    #print(np.sum(E[:, 0]))
    #print(np.sum(E[0, :]))
    return E


def mm_dataset(V, IterNum=8):
    for i in range(IterNum):
        E, E_rec = sga_m_reorder_rows_matrix(V, iterNum=50, batch_size=4000)
        # V = E_rec.dot(V)
        F = filter_full(E_rec.dot(V), iterNum=30, regu=15)

        plt.imshow(E_rec)
        plt.show()
        V = V * (E_rec.T).dot(F)
    return V, E_rec, F


def reorder_indicator(A,IN, iterNum=300, batch_size=20, gama=0, lr=0.1):
    '''
    Cyclic reorder rows using stochastic gradient ascent
    :param A: gene expression matrix
    :param iterNum:  iteration number
    :param batch_size: batch size
    :param gama: momentum parameter
    :return: permutation matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V = ge_to_spectral_matrix(A)
    E = np.ones((n, n)) / n
    E = E *IN
    E =BBS(E) #*IN
    #plt.imshow(E)
    #plt.show()
    E = sga_matrix_momentum_indicator(A, E, V=V.T, IN =IN, iterNum=iterNum, batch_size=batch_size, gama=gama, lr=lr)
    E_recon = reconstruct_e(E)
    return E, E_recon


def enhance_general_topology(A, V, regu=0.5, iterNum=300):
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=V, regu=regu, iterNum=iterNum)
    return F

def gene_inference_general_topology(A, V, regu=0.5, iterNum=100 , lr=0.1):
    A = np.array(A).astype('float64')
    A = gene_normalization(A)
    p  =A.shape[1]
    D = gradient_ascent_filter_matrix(A, D=np.identity((p))/2, U=V, regu=regu, lr=lr, iterNum=iterNum)
    return D



def sort_data_crit(adata,crit,crit_list):
    '''
    Sort the cells of an AnnData object according to a field (obs)
    :param adata: AnnData to be sorted
    :param crit: 'obs' field
    :param crit_list: list of 'obs' possible values, sorted according to the desired ordering (e.g ['0','6','12','18])
    :return:
    '''
    adata = shuffle_adata(adata) #for avoiding batch effects
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

def E_to_range(E):
    order =[]
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if E[i,j]==1:
                order.append(j)
    return np.array(order)
