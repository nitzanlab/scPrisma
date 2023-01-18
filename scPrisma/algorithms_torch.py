import copy

import torch
import numpy as np
from numpy import random

from scPrisma.pre_processing import *
from numba import jit

def reconstruct_e(E_prob):
    '''
    Greedy algorithm to reconstruct permutation matrix from Bi-Stochastic matrix
    Parameters
    ----------
    E_prob: np.array
        2D Bi-Stochastic matrix
    Returns: np.array
        2D Permutation matrix
    -------
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
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    optimize_alpha: bool
        Find the alpha value using optimization problem or by using the close formula


    Returns: np.array
        spectral matrix (concatenated eigenvectors multiplied by their appropriate eigenvalues)
    -------

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

def ge_to_spectral_matrix_torch(A , device):
    '''
    Parameters
    ----------
    A: torch.tensor
        Gene expression matrix
    optimize_alpha: bool
        Find the alpha value using optimization problem or by using the close formula


    Returns: torch.tensor
        spectral matrix (concatenated eigenvectors multiplied by their appropriate eigenvalues)
    -------

    '''
    n=A.shape[0]
    p = A.shape[1]
    alpha = np.exp(-2/p)
    V = generate_spectral_matrix_torch(n=n , device=device , alpha=alpha)
    V = V[1:, :]
    return V



def generate_spectral_matrix_torch(n, device,alpha=0.99):
    '''
    :param n: number of cells
    :param alpha: correlation between neighbors
    :return: spectral matrix, each eigenvector is multiplied by the sqrt of the appropriate eigenvalue
    '''
    eigen_vectors = torch.zeros((n,n))
    k = torch.arange(n)
    eigen_vectors = eigen_vectors.to(device)
    k = k.to(device)
    for i in range(n):
        v = np.sqrt(2 / n) * torch.cos(math.pi * (((2 * (i) * (k)) / n) - 1 / 4))
        eigen_vectors[i,:]=v
    eigen_values = torch.zeros(n)
    eigen_values = eigen_values.to(device)
    k_1 = k[:int(n/2)]
    k_2 = k[int(n/2):]
    for i in range(n):
        eigen_values[i]=torch.sum((alpha** k_1) * torch.cos((2 * np.pi * i * k_1) / n))
        eigen_values[i]+=torch.sum((alpha **  (n-k_2)) * torch.cos((2 * np.pi * i * k_2) / n))
    for i in range(n):
        eigen_vectors[i]*=torch.sqrt(eigen_values[i])
    return eigen_vectors



def function_and_gradient_matrix_torch(A, E, V):
    '''
    Calculate the function value and the gradient of A matrix
    Parameters
    ----------
    A: torch.tensor
        Gene expression matrix
    E: torch.tensor
        Bi-Stochastic matrix (should be constant)
    V: torch.tensor:
        Theoretical spectrum

    Returns
    -------
    functionValue: float
        function value
    gradient: torch.tensor
        gradient of E
    '''
    functionValue = torch.trace((((((V.T).mm(E)).mm(A)).mm(A.T)).mm(E.T)).mm(V))
    gradient = (2 * ((((V).mm(V.T)).mm(E)).mm(A)).mm(A.T))

    return functionValue, gradient

def g_matrix_torch(A, E, VVT):
    '''
    Calculate the gradient of A matrix, using boosted formula
    Parameters
    ----------
    A: torch.tensor
        Gene expression matrix
    E: torch.tensor
        Bi-Stochastic matrix (should be constant)
    VVT: torch.tensor:
        Theoretical spectrum (V) multiplied by his transform (V.T)

    Returns
    -------
    gradient: torch.tensor
        gradient of E
    '''
    gradient = (2 * (((VVT).mm(E)).mm(A)).mm(A.T))
    return gradient

def sga_matrix_momentum_torch(A, E, V, step, iterNum=400, batch_size=20, lr=0.1, gamma=0.9 , verbose=True , device='cpu'):
    """
    Perform stochastic gradient ascent for matrix momentum optimization.

    Parameters
    ----------
    A : torch.tensor
        Gene expression matrix.
    E : torch.tensor
        Permutation matrix initial value.
    V : torch.tensor
        Eigenvectors matrix multiple by sqrt of diagonal eigenvalues matrix.
    step : torch.tensor
        momentum step
    iterNum : int, optional
        Number of iteration, by default 400
    batch_size : int, optional
        batch size, by default 20
    lr : float, optional
        learning rate, by default 0.1
    gamma : float, optional
        momentum parameter, by default 0.9
    verbose : bool, optional
        Print iteration number if True, by default True
    device : str, optional
        Device to perform the computation on, by default 'cpu'

    Returns
    -------
    np.ndarray
        permutation matrix
    """
    j = 0
    value = 0
    VVT = (V).mm(V.T) #for runtime optimization
    del V
    torch.cuda.empty_cache()
    epsilon_t = lr
    prev_E = torch.empty(E.shape , dtype= torch.float32)
    prev_E = prev_E.to(device)
    I = torch.eye(E.shape[0], dtype= torch.float32)
    I = I.to(device)
    ones_m = torch.ones((E.shape[0], E.shape[0]) , dtype= torch.float32)
    ones_m = ones_m.to(device)
    while (j < iterNum ):
        if (j % 25 == 0) & verbose:
            print("Iteration number: ")
            print(j)
        A_tmp = A[:, torch.randint(low= 0, high= A.shape[1], size=(batch_size,))]
        grad = g_matrix_torch(A=A_tmp, E=E, VVT=VVT)
        step = epsilon_t * grad + gamma * step
        E = E + step
        E = BBS_torch(E=E, prev_E=prev_E, I=I, ones_m=ones_m)
        j += 1
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    return E


def sga_matrix_boosted_torch(A, E, V, iterNum=400, batch_size=20, lr=0.1 , verbose=True , device='cpu'):
    '''
    :param A: gene expression matrix
    :param E: permutation matrix initial value
    :param V: Eigenvectors matrix multiple by sqrt of diagonal eigenvalues matrix
    :param iterNum: iteration number
    :param batch_size: batch size
    :param lr: learning rate
    :param gamma: momentum parameter
    :return: permutation matrix
    '''
    j = 0
    value = 0
    VVT = (V).mm(V.T) #for runtime optimization
    del V
    torch.cuda.empty_cache()
    epsilon_t = lr
    prev_E = torch.empty(E.shape , dtype= torch.float32)
    prev_E = prev_E.to(device)
    I = torch.eye(E.shape[0], dtype= torch.float32)
    I = I.to(device)
    ones_m = torch.ones((E.shape[0], E.shape[0]) , dtype= torch.float32)
    ones_m = ones_m.to(device)
    while (j < iterNum ):
        if (j % 25 == 0) & verbose:
            value, grad = function_and_gradient_matrix_torch(A=A, E=E, V=V)
            print("Iteration number: ")
            print(j)
            print(" function value= ")
            print(value)
        A_tmp = A[:, torch.randint(low= 0, high= A.shape[1], size=(batch_size,))]
        grad = g_matrix_torch(A=A_tmp, E=E, VVT=VVT)
        E = E + epsilon_t * grad
        E = BBS_torch(E=E, prev_E=prev_E, I=I, ones_m=ones_m)
        j += 1
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    return E

def sga_matrix_momentum_indicator_torch(A, E, V,IN, iterNum=400, batch_size=20, lr=0.1, gamma=0.9):
    '''
    :param A: gene expression matrix
    :param E: permutation matrix initial value
    :param V: Eigenvectors matrix multiple by sqrt of diagonal eigenvalues matrix
    :param iterNum: iteration number
    :param batch_size: batch size
    :param lr: learning rate
    :param gamma: momentum parameter
    :return: permutation matrix
    '''
    j = 0
    value = 0
    epsilon_t = lr
    step = torch.zeros(E.shape)
    E = E * IN
    E = BBS_torch(E) * IN
    while (j < iterNum):
        if j % 25 == 0:
            print("Iteration number: " + str(j) + " function value= " + str(value))
        A_tmp = A[:, torch.randint(low= 0, high= A.shape[1], size=(batch_size,))]
        value, grad = function_and_gradient_matrix_torch(A=A_tmp, E=E, V=V)
        grad = grad
        step = epsilon_t * grad + gamma * step
        E = E + step
        E = BBS_torch(E) * IN
        j += 1
    return E


def reconstruction_cyclic_torch(A, iterNum=300, batch_size=None, gamma=0.5, lr=0.1 , verbose=True , final_loss=False, boosted=False):
    '''
    Cyclic reorder rows using stochastic gradient ascent
    :param A: gene expression matrix
    :param iterNum:  iteration number
    :param batch_size: batch size
    :param gamma: momentum parameter
    :return: permutation matrix
    '''
    if batch_size==None:
        batch_size= int((A.shape[0])*0.75)
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    V = ge_to_spectral_matrix(A , optimize_alpha=False)
    A = torch.from_numpy(A)
    A = A.type(torch.float32)
    A = A.float()
    V = torch.from_numpy(V.T )
    V = V.type(torch.float32)
    E = torch.ones((n,n) )
    E = E.type(torch.float32)
    step = torch.zeros(E.shape )
    step = step.type(torch.float32)
    A = A.to(device)
    E = E.to(device)
    step = step.to(device)
    V = V.to(device)
    E = sga_matrix_momentum_torch(A, E=E / n, V=V, iterNum=iterNum, step=step, batch_size=batch_size, gamma=gamma, device=device, lr=lr , verbose=verbose)
    E = (E.cpu()).numpy()
    E_recon = reconstruct_e(E)
    return E ,  E_recon

def reconstruction_cyclic_torch_boosted(A, iterNum=300, batch_size=None, gamma=0.9, lr=0.1 , verbose=True , final_loss=False, boosted=False):
    '''
    Cyclic reorder rows using stochastic gradient ascent
    :param A: gene expression matrix
    :param iterNum:  iteration number
    :param batch_size: batch size
    :param gamma: momentum parameter
    :return: permutation matrix
    '''
    if batch_size==None:
        batch_size= int((A.shape[0])*0.5)
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    V = ge_to_spectral_matrix(A , optimize_alpha=False)
    A = torch.from_numpy(A)
    A = A.type(torch.float32)
    A = A.float()
    V = torch.from_numpy(V.T )
    V = V.type(torch.float32)
    E = torch.ones((n,n) )
    E = E.type(torch.float32)
    step = torch.zeros(E.shape )
    step = step.type(torch.float32)
    A = A.to(device)
    E = E.to(device)
    step = step.to(device)
    if ~boosted:
        V = V.to(device)
        print(V)
        E = sga_matrix_momentum_torch(A, E=E / n, V=V, iterNum=iterNum, step=step, batch_size=batch_size, gamma=gamma, device=device, lr=lr , verbose=verbose)
    else:
        V_1 = V[:,:int(n/4)]
        V_2 = V[:,int(n/4):]
        V_boosted = torch.empty((A.shape[0],V_1.shape[1]+ V_2.shape[1]))
        V_boosted[:,:int(n/4)] = V_1
        V_boosted[:,int(n/4):] = V_2
        del V_1
        del V_2
        del V
        V_boosted = V_boosted.to(device)
        E = sga_matrix_boosted_torch(A, E=E / n, V=V_boosted, iterNum=iterNum, batch_size=batch_size, device=device, lr=lr , verbose=verbose)
    E = (E.cpu()).numpy()
    E_recon = reconstruct_e(E)
    return E ,  E_recon

def filter_non_cyclic_genes_torch(A, regu=0.1, lr=0.1, iterNum=500):
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
    U= U.T
    A = gene_normalization(A)
    T = np.ones((p))/2
    A = torch.from_numpy(A)
    U = torch.from_numpy(U)
    T = torch.from_numpy(T)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    U = U.to(device)
    T = T.to(device)
    D = gradient_ascent_filter_matrix_torch(A, T=T, U=U, regu=regu, lr=lr, iterNum=iterNum)
    return D




def gradient_ascent_filter_matrix_torch(A, T, U, ascent=1, lr=0.1, regu=0.1, iterNum=400):
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
    ATUUTA = 2* ((A.T).mm(U)).mm(U.T).mm(A)
    while (j < iterNum):
        if j % 25 == 1:
            print("Iteration number: ")
            print(j)
        epsilon_t *= 0.995
        #T = D.diagonal()#numba_diagonal(D)#.diagonal()
        #grad = ATUUTA * T - regu * torch.sign(D)
        T = T + ascent* epsilon_t * (ATUUTA * T).diag() - regu * torch.sign(T)
        T = torch.clip(T,0,1)
        j += 1
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    return T.diag()



def enhancement_cyclic_torch(A, regu=0.1, iterNum=100, verbosity = 25 , error=10e-7, optimize_alpha=False, line_search=False):
    ''' Enhancement of cyclic signal
    :param A: Gene expression matrix (reordered according to cyclic ordering)
    :param regu: regularization coefficient
    :param iterNum: iteration number
    :return: filtering matrix
    '''
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A , optimize_alpha=False)
    V= V.T
    F = torch.ones(A.shape)
    V =torch.from_numpy(V)
    A =torch.from_numpy(A)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    V = V.to(device)
    VVT = (V).mm(V.T)
    del V
    torch.cuda.empty_cache()
    F = F.to(device)
    print(A.get_device())
    print(device)
    print("starting filtering")
    F = gradient_ascent_full_torch(A,F,VVT=VVT,regu=regu,epsilon=0.1,iterNum=iterNum , device=device)
    return F

def filtering_cyclic_torch(A, regu=0.1, iterNum=100, verbosity = 25 , error=10e-7, optimize_alpha=False, line_search=False):
    """
    This function filters the cyclic signal by applying gradient descent method.
    Parameters:
    A (torch.tensor): The gene expression matrix reordered according to cyclic ordering.
    regu (float, optional): The regularization coefficient. Default is 0.1.
    iterNum (int, optional): The number of iterations for the gradient descent. Default is 300.
    verbosity (int, optional): The verbosity level. Default is 25.
    error (float, optional): The stopping criteria for the gradient descent. Default is 10e-7.
    optimize_alpha (bool, optional): Whether to optimize the regularization parameter. Default is True.
    line_search (bool, optional): Whether to use line search. Default is True.

    Returns:
    torch.tensor: Filtering matrix.
    """
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A , optimize_alpha=optimize_alpha)
    V= V.T
    F = torch.ones(A.shape).to(float)
    V =torch.from_numpy(V).to(float)
    A =torch.from_numpy(A).to(float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    V = V.to(device)
    VVT = (V).mm(V.T)
    del V
    torch.cuda.empty_cache()
    F = F.to(device)
    print(A.get_device())
    print(device)
    print("starting filtering")
    F = gradient_descent_full_torch(A,F,VVT=VVT,regu=regu,epsilon=0.1,iterNum=iterNum , device=device)
    return F


def gradient_descent_full_torch(A, F, VVT, regu, epsilon=0.1, iterNum=400 , regu_norm ='L1' , device='cpu'):
    j = 0
    epsilon_t = epsilon
    while j < iterNum:
        if j % 100 == 1:
            print("Iteration number: " + str(j))
        epsilon_t *= 0.995
        grad = G_full_torch(A=A, B=F, VVT=VVT, alpha=regu, regu_norm=regu_norm)
        F = F - epsilon_t * grad
        F = torch.clip(F,0,1)
        j += 1
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    return F

def gradient_ascent_full_torch(A, F, VVT, regu, epsilon=0.1, iterNum=400 , regu_norm ='L1' , device='cpu'):
    """
    This function enhances the signal by applying gradient ascent method.
    Parameters:
    A (torch.tensor): The gene expression matrix.
    F (torch.tensor): The enhancement matrix.
    VVT (torch.tensor):  Spectral matrix.
    regu (float): The regularization coefficient.
    epsilon (float, optional): The step size. Default is 0.1.
    iterNum (int, optional): The number of iterations for the gradient ascent. Default is 400.

    Returns:
    numpy.ndarray: The enhancement matrix.
    """
    j = 0
    epsilon_t = epsilon
    while j < iterNum:
        if j % 100 == 1:
            print("Iteration number: " + str(j))
        epsilon_t *= 0.995
        grad = G_full_torch(A=A, B=F, VVT=VVT, alpha=regu, regu_norm=regu_norm)
        F = F + epsilon_t * grad
        F = torch.clip(F,0,1)
        j += 1
    del grad
    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    return F




def G_full_torch(A, B, VVT, alpha):
    '''
    :param A: Gene expression matrix
    :param B: filtering matrix
    :param V: spectral matrix
    :param alpha: correlation between neighbors
    :return:projection over theoretic spectrum and gradient according to 'B'
    '''
    T_0 = (A * B)
    gradient = ((2 * ((VVT).mm(T_0) * A)) - ((alpha ) * torch.sign(B)))
    return gradient



def BBS_torch(E, prev_E, I ,ones_m , iterNum=1000):
    ''' Bregmanian Bi-Stochastication algorithm as described inL
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/KAIS_BBS_final.pdf
    :param E: permutation matrix
    :param iterNum:iteration number
    :return:Bi-Stochastic matrix
    '''
    n = E.shape[0]
    for i in range(iterNum):
        if i % 10 == 1:
            prev_E = torch.clone(E)
        ones_E = ones_m.mm(E)
        E = E + (1 / n) * (I - E + (1 / n) * (ones_E)).mm(ones_m) - (1 / n) * ones_E
        E = torch.clip(E,min=0, max=1)# E,E.shape[0],E.shape[0],0)
        if i % 10 == 1:
            if torch.norm(E - prev_E) < ((10e-7) * n*n):
                break
    return E



def enhance_general_topology_torch(A, V, regu=0.5, iterNum=300):
    A = cell_normalization(A)
    F = torch.ones(A.shape)
    V =torch.from_numpy(V)
    A =torch.from_numpy(A)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    V = V.to(device)
    VVT = (V).mm(V.T)
    del V
    torch.cuda.empty_cache()
    F = F.to(device)
    print(A.get_device())
    print(device)
    print("starting filtering")
    F = gradient_ascent_full_torch(A, F=F, VVT=VVT, regu=regu, iterNum=iterNum)
    return F

def gene_inference_general_topology_torch(A, V, regu=0.5, iterNum=100 , lr=0.1):
    A = gene_normalization(A)
    p  =A.shape[1]
    T = np.ones((p))/2
    A = torch.from_numpy(A)
    V = torch.from_numpy(V)
    T = torch.from_numpy(T)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    V = V.to(device)
    T = T.to(device)
    D = gradient_ascent_filter_matrix_torch(A, T=T, U=V, regu=regu, lr=lr, iterNum=iterNum)
    del A
    del V
    del T
    return D


def filter_general_topology_torch(A, V, regu=0.5, iterNum=300):
    A = cell_normalization(A)
    F = torch.ones(A.shape)
    V =torch.from_numpy(V)
    A =torch.from_numpy(A)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    V = V.to(device)
    VVT = (V).mm(V.T)
    del V
    torch.cuda.empty_cache()
    F = F.to(device)
    print(A.get_device())
    print(device)
    print("starting filtering")
    F = gradient_descent_full_torch(A,F=F,VVT=VVT,regu=regu,epsilon=0.1,iterNum=iterNum , device=device)
    return F




def filter_general_covariance_torch(A, cov, regu=0, epsilon=0.1, iterNum=100, device='cpu'):
    """
    Filters `adata` based on a discrete label by running gradient descent with L1 regularization.

    Parameters
    ----------
    A : ndarray
        AnnData object to filter.
    cov : ndarray
        The covariance matrix to calculate eigenvalues and eigenvectors for.
    regu : float
        Regularization coefficient.
    epsilon : float, optional
        Step size (learning rate).
    iterNum : int, optional
        Number of iterations to run gradient descent.
    regu_norm : str, optional
        Regularization norm to use, either 'L1' or 'L2'.
    device : str, optional
        Device to use for computations, either 'cpu' or 'cuda'.

    Returns
    -------
    F: np.array
         Filtering matrix
    """
    B = A.copy()
    V = torch.tensor(np.array(get_theoretic_eigen(cov)).astype(float), device=device)
    VVT = (V).mm(V.T)
    del V
    B = normalize(B, axis=1, norm='l2')
    B = torch.tensor(B.astype(float), device=device)
    F_gpu = torch.tensor(np.ones(B.shape).astype(float), device=device)
    F_gpu = gradient_descent_full_torch(B, F_gpu.type(torch.float), VVT=VVT, regu=regu, epsilon=epsilon, iterNum=iterNum)
    del A, V
    F = F_gpu.cpu().detach().numpy()
    del F_gpu
    return F
