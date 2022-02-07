import copy

import numpy as np
from numpy import random
from scipy.optimize.linesearch import line_search

from pre_processing import *
from visualizations import *
from sinkhorn_knopp import sinkhorn_knopp as skp


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

def ge_to_spectral_matrix(A):
    '''
    :param A: Gene expression matrix
    :return: Theoretic spectral matrix
    '''
    n=A.shape[0]
    p = A.shape[1]
    min_np = min(n,p)
    u, s, vh = np.linalg.svd(A)
    for i in range(min_np):
        s[i] *= s[i]
    alpha = optimize_alpha_p(s, 15)
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
            plt.imshow(E)
            plt.colorbar()
            plt.show()
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


def fAndG_matrix(A, E, V):
    '''
    :param A: gene expression matrix
    :param E: permutation matrix
    :param V: spectral matrix
    :return: function value, gradient of E
    '''
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(E, np.ndarray)
    dim = E.shape
    assert len(dim) == 2
    E_rows = dim[0]
    E_cols = dim[1]
    assert isinstance(V, np.ndarray)
    dim = V.shape
    assert len(dim) == 2
    V_rows = dim[0]
    V_cols = dim[1]
    assert E_rows == V_rows
    assert A_rows == E_cols

    functionValue = np.trace((((((V.T).dot(E)).dot(A)).dot(A.T)).dot(E.T)).dot(V))
    gradient = (2 * ((((V).dot(V.T)).dot(E)).dot(A)).dot(A.T))

    return functionValue, gradient

def G_matrix(A, E, VVT):
    '''
    :param A: gene expression matrix
    :param E: permutation matrix
    :param V: spectral matrix
    :return: gradient of E
    '''
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(E, np.ndarray)
    dim = E.shape
    assert len(dim) == 2
    E_rows = dim[0]
    E_cols = dim[1]
    assert A_rows == E_cols

    #gradient = (2 * ((((V).dot(V.T)).dot(E)).dot(A)).dot(A.T))
    gradient = (2 * (((VVT).dot(E)).dot(A)).dot(A.T))

    return gradient


def sga_matrix_momentum(A, E, V, iterNum=400, batch_size=20, lr=0.1, gama=0.9, projection='BBS'):
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
    if projection=='skp':
        sk = skp.SinkhornKnopp()
    epsilon_t = lr
    step = np.zeros(E.shape)
    while (j < iterNum):
        if j % 25 == 0:
            value, grad = fAndG_matrix(A=A, E=E, V=V)
            print("Iteration number: " + str(j) + " function value= " + str(value))
        A_tmp = A[:, np.random.randint(A.shape[1], size=batch_size)]
        grad = G_matrix(A=A_tmp, E=E, VVT=VVT)
        step = epsilon_t * grad + gama * step
        E = E + step
        if projection=='skp':#skp
            E = np.clip(E , a_max=1,a_min=0)
            E = sk.fit(E)
        else:
            E = BBS(E)
        j += 1
    return E

def sga_matrix_momentum_indicator(A, E, V,IN, iterNum=400, batch_size=20, lr=0.1, gama=0.9,projection='BBS'):
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
    if projection=='skp':
        sk = skp.SinkhornKnopp()
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
        if projection=='BBS':
            E = BBS(E) * IN
        else:
            E = sk.fit(E) * IN
        j += 1
    return E

def sga_m_reorder_rows_matrix(A, iterNum=300, batch_size=None, gama=0.5, lr=0.1, projection='BBS'):
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
    E = sga_matrix_momentum(A, E=np.ones((n, n)) / n, V=V.T, iterNum=iterNum, batch_size=batch_size, gama=gama, lr=lr, projection=projection)
    E_recon = reconstruct_e(E)
    return E, E_recon

def reconstruction_cyclic(A, iterNum=300, batch_size=None, gama=0.5, lr=0.1, projection='BBS'):
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
    E = sga_matrix_momentum(A, E=np.ones((n, n)) / n, V=V.T, iterNum=iterNum, batch_size=batch_size, gama=gama, lr=lr, projection=projection)
    E_recon = reconstruct_e(E)
    return E, E_recon

def filter_non_cyclic_genes(A, regu=0.1, lr=0.1, iterNum=500):
    '''
    :param A: gene expression matrix
    :param regu: regularization parameter
    :param iterNum: iteration number
    :param lr: learning rate
    :return: diagonal filtering matrix
    '''
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
    V = cell_normalization(A)
    p = V.shape[1]
    U =ge_to_spectral_matrix(V)
    A = gene_normalization(A)
    D = gradient_ascent_filter_matrix_line(A, D=np.identity((p)), U=U.T, regu=regu , max_evals=iterNum,verbosity=verbosity)
    return D

def filter_non_cyclic_genes_line(A, regu=0.1, iterNum=500, lr=0.1, verbosity=25):
    '''
    :param A: gene expression matrix
    :param regu: regularization parameter
    :param iterNum: iteration number
    :param lr: learning rate
    :return: diagonal filtering matrix
    '''
    V = cell_normalization(A)
    p = V.shape[1]
    U =ge_to_spectral_matrix(V)
    A = gene_normalization(A)
    D = gradient_ascent_filter_matrix_line(A, D=np.identity((p)), U=U.T, regu=regu , max_evals=iterNum , verbosity=verbosity)
    np.identity(D.shape[1])
    return (np.identity(D.shape[1]) - D)

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
            print("Iteration number: " + str(j) + "function value= " + str(val))
            plot_diag(D)
            im = plt.imshow(D)
            plt.colorbar()
            plt.show()
        epsilon_t *= 0.995
        T = D.diagonal()
        grad = ATUUTA * T - regu * np.sign(D)
        #grad += np.random.normal(0,0.001,grad.shape)
        #val, grad = fAndG_fixed_filter(A=A, D=D, U=U, regu=regu)
        #print(np.allclose(grad,grad1))
        # grad = ATUUTA.dot(D) - (regu * np.sign(D))
        D = D + ascent* epsilon_t * grad
        D = diag_projection(D)
        # print("Iteration number: " + str(j) + " grad value= " + str(grad))
        j += 1
    return D



def loss_filter_genes(A,U,D,regu):
    return np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - regu*np.linalg.norm(D,1)

def gradient_ascent_filter_matrix_line(A, D, U, regu=0.1, gamma = 1e-04, max_evals = 250, verbosity = float('inf')):
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
    loss = loss_filter_genes(A,U,w,regu)
    T = w.diagonal()
    w = np.diag(T)
    grad = ATUUTA * T - regu * np.sign(w)
    G = grad.diagonal()
    grad = np.diag(G)
    alpha = 1 / np.linalg.norm(grad)
    while evals < max_evals and np.linalg.norm(grad) > 1e-07:
        evals += 1
        if evals % verbosity == 0:
            print(str(evals) + 'th Iteration    Loss :: ' + str(loss) + ' gradient :: ' +  str(np.linalg.norm(grad)))
        gTg  = np.linalg.norm(grad)
        gTg = gTg*gTg
        new_w = w - alpha * grad
        new_loss = loss_filter_genes(A, U, new_w, regu)
        T = new_w.diagonal()
        new_w = np.diag(T)
        new_grad = ATUUTA * T - regu * np.sign(new_w)
        G = new_grad.diagonal()
        new_grad = np.diag(G)
        while new_loss > loss - gamma * alpha * gTg:
            alpha = ((alpha ** 2) * gTg) / (2 * (new_loss + alpha * gTg - loss))
            new_w = w - alpha * grad
            new_loss = loss_filter_genes(A, U, new_w, regu)
            T = new_w.diagonal()
            new_w = np.diag(T)
            new_grad = ATUUTA * T - regu * np.sign(new_w)
            G = new_grad.diagonal()
            new_grad = np.diag(G)

        alpha = min(1, 2 * (loss - new_loss) / gTg)
        loss = new_loss
        grad = new_grad
        w = new_w
    return w

def fAndG_filter_matrix(A, D, U, alpha):
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(D, np.ndarray)
    dim = D.shape
    assert len(dim) == 2
    D_rows = dim[0]
    assert isinstance(U, np.ndarray)
    dim = U.shape
    assert len(dim) == 2
    U_rows = dim[0]
    if isinstance(alpha, np.ndarray):
        dim = alpha.shape
        assert dim == (1,)
    assert U_rows == A_rows
    assert D_rows == A_cols

    functionValue = (np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - (alpha * np.sum(np.abs(D))))
    gradient = ((2 * ((((A.T).dot(U)).dot(U.T)).dot(A)).dot(D)) - (alpha * np.sign(D)))
    return gradient, functionValue


def fAndG_fixed_filter(A, D, U, regu):
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(D, np.ndarray)
    dim = D.shape
    assert len(dim) == 2
    D_rows = dim[0]
    D_cols = dim[1]
    assert isinstance(U, np.ndarray)
    dim = U.shape
    assert len(dim) == 2
    U_rows = dim[0]
    assert A_rows == U_rows
    assert D_rows == A_cols

    t_0 = np.linalg.norm((A).dot(D), 'fro')
    functionValue = np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - regu * t_0
    gradient = (2 * ((((A.T).dot(U)).dot(U.T)).dot(A)).dot(D)) - regu*np.sign(D)#((1 / t_0) * ((A.T).dot(A)).dot(D))

    return functionValue, gradient


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
            # print(D.diagonal())
            plot_diag(D)
            plt.imshow(D)
            plt.colorbar()
            plt.show()
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


def diag_projection(D):
    T = D.diagonal()
    # T[T < 0] = 0
    # T = T.clip(0,D.shape[0])
    # T = T#*((np.sum(T)/D.shape[0]))
    T = T.clip(0, 1)
    return np.diag(T)


def fAndG_regu(A, E, alpha, x):
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(E, np.ndarray)
    dim = E.shape
    assert len(dim) == 2
    E_rows = dim[0]
    E_cols = dim[1]
    if isinstance(alpha, np.ndarray):
        dim = alpha.shape
        assert dim == (1,)
    assert isinstance(x, np.ndarray)
    dim = x.shape
    assert len(dim) == 1
    x_rows = dim[0]
    assert A_rows == x_rows
    assert E_rows == A_cols

    t_0 = (A.T).dot(x)
    t_1 = np.linalg.norm(E, 'fro')
    functionValue = ((x).dot((A).dot((E).dot((E.T).dot(t_0)))) - (alpha * t_1))
    gradient = ((2 * np.multiply.outer(t_0, ((x).dot(A)).dot(E))) - ((alpha / t_1) * E))

    return functionValue, gradient


def calculate_roc_auc(y_target, y_true):
    return roc_auc_score(y_true, y_target)


def filter_full(A, regu=0.1, iterNum=300):
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V =ge_to_spectral_matrix(A)
    #F = gradient_ascent_full(A, F=np.ones(A.shape), V=V.T, regu=regu, iterNum=iterNum)
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=V.T, regu=regu, iterNum=iterNum)
    return F

def enhancement_cyclic(A, regu=0.1, iterNum=300):
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
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=V.T, regu=regu, iterNum=iterNum)
    return F

def filtering_cyclic(A, regu=0.1, iterNum=300, verbosity = 25 , error=10e-7):
    ''' Filtering of cyclic signal
    :param A: Gene expression matrix (reordered according to cyclic ordering)
    :param regu: regularization coefficient
    :param iterNum: iteration number
    :return: filtering matrix
    '''
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A)
    print("starting filtering")
    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=V.T, regu=regu, max_evals=iterNum,verbosity=verbosity , error=error)
    return F

def filtering_cyclic_boosted(A, regu=0.1, iterNum=300, verbosity = 25 , error=10e-7):
    ''' Filtering of cyclic signal
    :param A: Gene expression matrix (reordered according to cyclic ordering)
    :param regu: regularization coefficient
    :param iterNum: iteration number
    :return: filtering matrix
    '''
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
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A)
    print("starting filtering")
    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=V.T, regu=regu, max_evals=iterNum,verbosity=verbosity , error=error)
    return F

def filter_non_cyclic_full_reverse(A, regu=0.1, iterNum=300):
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A)
    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=V.T, regu=regu, max_evals=iterNum)
    F = np.ones(F.shape)-F
    return F

def gradient_ascent_full(A, F, V, regu, epsilon=0.1, iterNum=400):
    print(A.shape)
    print(F.shape)
    print(V.shape)
    j = 0
    epsilon_t = epsilon
    while (j < iterNum):
        value = 0
        if j % 50 == 1:
            print("Iteration number: " + str(j))
            plt.imshow(F)
            plt.colorbar()
            plt.show()
        epsilon_t *= 0.995
        tmp_value, grad = fAndG_full(A=A, B=F, V=V, alpha=regu)
        F = F + epsilon_t * grad
        F = F.clip(min=0, max=1)
        j += 1
    plt.imshow(F)
    plt.colorbar()
    plt.show()
    return F

def stochastic_gradient_ascent_full(A, F, V, regu, epsilon=0.1, iterNum=400 , regu_norm='L1'):
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
        if j % 25 == 1:
            value, grad = fAndG_full_acc(A=A, B=F, V=V, VVT=VVT, alpha=regu, regu_norm=regu_norm)
            print("Iteration number: " + str(j) , "function value: " +str(value) )
        epsilon_t *= 0.995
        grad = G_full(A=A, B=F, V=V, alpha=regu , regu_norm = regu_norm)
        F = F + epsilon_t * (grad +np.random.normal(0,0.01,grad.shape))
        F = F.clip(min=0, max=1)
        j += 1
    plt.imshow(F)
    plt.colorbar()
    plt.show()
    return F

def gradient_descent_full(A, F, V, regu, epsilon=0.1, iterNum=400 , regu_norm ='L1'):
    print(A.shape)
    print(F.shape)
    print(V.shape)
    j = 0
    epsilon_t = epsilon
    while j < iterNum:
        if j % 100 == 1:
            print("Iteration number: " + str(j))
            plt.imshow(F)
            plt.colorbar()
            plt.show()
        epsilon_t *= 0.995
        tmp_value, grad = fAndG_full(A=A, B=F, V=V, alpha=regu, regu_norm=regu_norm)
        F = F - epsilon_t * grad
        F = F.clip(min=0, max=1)
        j += 1
    plt.imshow(F)
    plt.colorbar()
    plt.show()
    return F



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
        prev_w=copy.deepcopy(w)
        evals += 1
        if evals % verbosity == 0:
            print(str(evals) + 'th Iteration    Loss :: ' + str(loss) + ' gradient :: ' +  str(np.linalg.norm(grad)))
        gTg  = np.linalg.norm(grad)
        gTg = gTg*gTg
        new_w = w - alpha * grad
        new_w = new_w.clip(min=0, max=1)
        new_loss, new_grad = fAndG_full_acc(A=A, B=new_w, V=V,
                                            VVT=VVT,alpha=regu, regu_norm=regu_norm)
        while new_loss > loss - gamma * alpha * gTg:
            alpha = ((alpha ** 2) * gTg) / (2 * (new_loss + alpha * gTg - loss))
            new_w = w - alpha * grad
            new_w = new_w.clip(min=0, max=1)
            new_loss, new_grad = fAndG_full_acc(A=A, B=new_w, V=V,VVT=VVT,
                                                alpha=regu, regu_norm=regu_norm)
        alpha = min(1, 2 * (loss - new_loss) / gTg)
        loss = new_loss
        grad = new_grad
        w = new_w
    return w


def fAndG_filtering_boosted(A, V, alpha_matrix):
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(V, np.ndarray)
    dim = V.shape
    assert len(dim) == 2
    V_rows = dim[0]
    V_cols = dim[1]
    assert A_rows == V_rows
    functionValue = np.trace((((V.T).dot(A)).dot(A.T)).dot(V)) - np.linalg.norm(alpha_matrix*A,1)
    gradient = (2 * ((V).dot(V.T)).dot(A)) - alpha_matrix*np.sign(A)
    return functionValue, gradient



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

def fAndG_full(A, B, V, alpha, regu_norm='L2'):
    '''
    :param A: Gene expression matrix
    :param B: filtering matrix
    :param V: spectral matrix
    :param alpha: correlation between neighbors
    :return:projection over theoretic spectrum and gradient according to 'B'
    '''
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(B, np.ndarray)
    dim = B.shape
    assert len(dim) == 2
    B_rows = dim[0]
    B_cols = dim[1]
    assert isinstance(V, np.ndarray)
    dim = V.shape
    assert len(dim) == 2
    V_rows = dim[0]
    V_cols = dim[1]
    if isinstance(alpha, np.ndarray):
        dim = alpha.shape
        assert dim == (1,)
    assert A_rows == V_rows == B_rows
    assert A_cols == B_cols
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
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(B, np.ndarray)
    dim = B.shape
    assert len(dim) == 2
    B_rows = dim[0]
    B_cols = dim[1]
    assert isinstance(V, np.ndarray)
    dim = V.shape
    assert len(dim) == 2
    V_rows = dim[0]
    V_cols = dim[1]
    if isinstance(alpha, np.ndarray):
        dim = alpha.shape
        assert dim == (1,)
    assert A_rows == V_rows == B_rows
    assert A_cols == B_cols
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

def G_full(A, B, V, alpha, regu_norm='L1'):
    '''
    :param A: Gene expression matrix
    :param B: filtering matrix
    :param V: spectral matrix
    :param alpha: correlation between neighbors
    :return:projection over theoretic spectrum and gradient according to 'B'
    '''
    assert isinstance(A, np.ndarray)
    dim = A.shape
    assert len(dim) == 2
    A_rows = dim[0]
    A_cols = dim[1]
    assert isinstance(B, np.ndarray)
    dim = B.shape
    assert len(dim) == 2
    B_rows = dim[0]
    B_cols = dim[1]
    assert isinstance(V, np.ndarray)
    dim = V.shape
    assert len(dim) == 2
    V_rows = dim[0]
    V_cols = dim[1]
    if isinstance(alpha, np.ndarray):
        dim = alpha.shape
        assert dim == (1,)
    assert A_rows == V_rows == B_rows
    assert A_cols == B_cols
    if regu_norm=='L1':
        T_0 = (A * B)
        gradient = ((2 * (((V).dot(V.T)).dot(T_0) * A)) - ((alpha) * np.sign(B)))
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
    A = gene_normalization(A)
    n = A.shape[0]
    p = A.shape[1]

    eigenvectors, eigenvalues = generate_eigenvectors_circulant()

    D = gradient_ascent_filter(A, D=np.identity((p)), eigenvectors_list=eigenvectors[1:],
                               eigenvalues_list=eigenvalues[1:], regu=regu, iterNum=iterNum)
    return D


def filter_linear_full(A, method, regu=0.1, iterNum=300 , lr=0.1, regu_norm='L2'):
    A = cell_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(optimized_alpha=True, eigenvals=real_eigenvalues)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    F = gradient_descent_full(A, F=np.ones(A.shape), V=eigenvectors, regu=regu,
                             iterNum=iterNum, epsilon=lr, regu_norm=regu_norm)
    return F

def filtering_linear(A, method,regu=0.1, iterNum=300, verbosity = 25 ,
                     error=10e-7, optimized_alpha=True, regu_norm='L2'):
    ''' Filtering of linear signal
    :param A: Gene expression matrix (reordered according to linear ordering)
    :param regu: regularization coefficient
    :param iterNumenhance_linear_full: iteration number
    :return: filtering matrix
    '''

    A = cell_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(ngenes=A.shape[1], optimized_alpha=optimized_alpha, eigenvals=real_eigenvalues)
    print(alpha)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=eigenvectors,
                                   regu=regu, max_evals=iterNum,verbosity=verbosity ,
                                   error=error , regu_norm=regu_norm)
    return F

def enhance_linear_genes(A, method, regu=2, iterNum=500, lr=0.1):
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


def filter_linear_genes(A, method, regu=2, iterNum=500, lr=0.1):
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
            prev_E = copy.deepcopy(E)
        ones_E = ones_m.dot(E)
        # print(ones_E)
        #E = E + (1 / n) * (I - E + (1 / n) * (ones_E)).dot(ones_m) - (1 / n) * ones_E
        E = np.clip(E + (1 / n) * (I - E + (1 / n) * (ones_E)).dot(ones_m) - (1 / n) * ones_E, a_min=0, a_max=None)
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


def signal_classification(A):
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    u, s, vh = np.linalg.svd(A)
    np1 = min(n, p)
    for i in range(np1):
        s[i] *= s[i]
    _, loss_cyclic = optimize_alpha_with_loss(s, loss_alpha_func=loss_alpha,
                                              generation_func=generate_eigenvalues_circulant)
    _, loss_linear = optimize_alpha_with_loss(s, loss_alpha_func=loss_alpha_linear,
                                              generation_func=get_pseudo_eigenvalues_for_loss)
    print("loss cyclic: " + str(loss_cyclic))
    print("loss linear: " + str(loss_linear))
    if loss_cyclic < loss_linear:
        print("cyclic!!")
    else:
        print("linear!!")
    pass


def reorder_indicator(A,IN, iterNum=300, batch_size=20, gama=0, lr=0.1):
    '''
    Cyclic reorder rows using stochastic gradient ascent
    :param A: gene expression matrix
    :param iterNum:  iteration number
    :param batch_size: batch size
    :param gama: momentum parameter
    :return: permutation matrix
    '''
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

def enhance_linear_full(A, method, regu=0.1, iterNum=300):
    A = cell_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(optimized_alpha=True, eigenvals=real_eigenvalues)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    F = gradient_ascent_full(A, F=np.ones(A.shape), V=eigenvectors, regu=regu,
                             iterNum=iterNum, epsilon=0.2)
    return F


def filter_layers_full(A, cov, regu=0.1, iterNum=300):
    A = cell_normalization(A)
    t_eigenvalues, t_vec = linalg.eigh(cov)
    for i , eig in enumerate(t_eigenvalues):
        t_vec[:, i]*=eig
    eigenvectors = t_vec
    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=eigenvectors, regu=regu,
                             max_evals=iterNum)#, epsilon=0.1)
    return F

def enhance_layers_full(A, cov, regu=0.1, iterNum=300):
    print(1111111)
    A = cell_normalization(A)
    t_eigenvalues, t_vec = linalg.eigh(cov)
    for i , eig in enumerate(t_eigenvalues):
        t_vec[:, i]*=eig
    eigenvectors = t_vec
    F = gradient_ascent_full(A, F=np.ones(A.shape), V=eigenvectors, regu=regu,
                             iterNum=iterNum, epsilon=0.1)
    return F
