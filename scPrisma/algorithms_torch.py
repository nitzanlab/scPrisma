
import torch
import numpy as np
from numpy import random

from scPrisma.pre_processing import *


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


def ge_to_spectral_matrix(A, optimize_alpha=True):
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
    n = A.shape[0]
    p = A.shape[1]
    min_np = min(n, p)
    if optimize_alpha:
        u, s, vh = np.linalg.svd(A)
        for i in range(min_np):
            s[i] *= s[i]
        alpha = optimize_alpha_p(s, 15)
    else:
        alpha = np.exp(-2 / p)
    V = generate_spectral_matrix(n=n, alpha=alpha)
    V = V[1:, :]
    return V


def ge_to_spectral_matrix_torch(A, device):
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
    n = A.shape[0]
    p = A.shape[1]
    alpha = np.exp(-2 / p)
    V = generate_spectral_matrix_torch(n=n, device=device, alpha=alpha)
    V = V[1:, :]
    return V


def generate_spectral_matrix_torch(n, device, alpha=0.99):
    '''
    :param n: number of cells
    :param alpha: correlation between neighbors
    :return: spectral matrix, each eigenvector is multiplied by the sqrt of the appropriate eigenvalue
    '''
    eigen_vectors = torch.zeros((n, n))
    k = torch.arange(n)
    eigen_vectors = eigen_vectors.to(device)
    k = k.to(device)
    for i in range(n):
        v = np.sqrt(2 / n) * torch.cos(math.pi * (((2 * (i) * (k)) / n) - 1 / 4))
        eigen_vectors[i, :] = v
    eigen_values = torch.zeros(n)
    eigen_values = eigen_values.to(device)
    k_1 = k[:int(n / 2)]
    k_2 = k[int(n / 2):]
    for i in range(n):
        eigen_values[i] = torch.sum((alpha ** k_1) * torch.cos((2 * np.pi * i * k_1) / n))
        eigen_values[i] += torch.sum((alpha ** (n - k_2)) * torch.cos((2 * np.pi * i * k_2) / n))
    for i in range(n):
        eigen_vectors[i] *= torch.sqrt(eigen_values[i])
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


def sga_matrix_momentum_torch(A, E, V, step, iterNum=400, batch_size=20, lr=0.1, gamma=0.9, verbose=True, device='cpu'):
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
    torch.tensor
        permutation matrix
    """
    j = 0
    value = 0
    VVT = (V).mm(V.T)  # for runtime optimization
    del V
    torch.cuda.empty_cache()
    epsilon_t = lr
    prev_E = torch.empty(E.shape, dtype=torch.float32)
    prev_E = prev_E.to(device)
    I = torch.eye(E.shape[0], dtype=torch.float32)
    I = I.to(device)
    ones_m = torch.ones((E.shape[0], E.shape[0]), dtype=torch.float32)
    ones_m = ones_m.to(device)
    while (j < iterNum):
        if (j % 25 == 0) & verbose:
            print("Iteration number: ")
            print(j)
        A_tmp = A[:, torch.randint(low=0, high=A.shape[1], size=(batch_size,))]
        grad = g_matrix_torch(A=A_tmp, E=E, VVT=VVT)
        step = epsilon_t * grad + gamma * step
        E = E + step
        E = BBS_torch(E=E, prev_E=prev_E, I=I, ones_m=ones_m)
        j += 1
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    return E


def sga_matrix_boosted_torch(A, E, V, iterNum=400, batch_size=20, lr=0.1, verbose=True, device='cpu'):
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
    VVT = (V).mm(V.T)  # for runtime optimization
    torch.cuda.empty_cache()
    epsilon_t = lr
    prev_E = torch.empty(E.shape, dtype=torch.float32)
    prev_E = prev_E.to(device)
    I = torch.eye(E.shape[0], dtype=torch.float32)
    I = I.to(device)
    ones_m = torch.ones((E.shape[0], E.shape[0]), dtype=torch.float32)
    ones_m = ones_m.to(device)
    while (j < iterNum):
        if (j % 25 == 0) & verbose:
            value, grad = function_and_gradient_matrix_torch(A=A, E=E, V=V)
            print("Iteration number: ")
            print(j)
            print(" function value= ")
            print(value)
        A_tmp = A[:, torch.randint(low=0, high=A.shape[1], size=(batch_size,))]
        grad = g_matrix_torch(A=A_tmp, E=E, VVT=VVT)
        E = E + epsilon_t * grad
        E = BBS_torch(E=E, prev_E=prev_E, I=I, ones_m=ones_m)
        j += 1
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    return E


def reconstruction_cyclic_torch(A, iterNum=300, batch_size=None, gamma=0.5, lr=0.1, verbose=True, final_loss=False,
                                boosted=False):
    '''
    Cyclic reorder rows using stochastic gradient ascent
    :param A: gene expression matrix
    :param iterNum:  iteration number
    :param batch_size: batch size
    :param gamma: momentum parameter
    :return: permutation matrix
    '''
    if batch_size == None:
        batch_size = int((A.shape[0]) * 0.75)
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    V = ge_to_spectral_matrix(A, optimize_alpha=False)
    V = V.T
    A = torch.tensor(A, dtype=torch.float32, device=device)
    V = torch.tensor(V,dtype=torch.float32, device=device)
    E = torch.ones((n, n), dtype=torch.float32, device=device)
    step = torch.zeros(E.shape, dtype=torch.float32, device=device)



    E = sga_matrix_momentum_torch(A, E=E / n, V=V, iterNum=iterNum, step=step, batch_size=batch_size, gamma=gamma,
                                  device=device, lr=lr, verbose=verbose)
    E_cpu = (E.cpu()).numpy()
    E_recon = reconstruct_e(E_cpu)
    del A, E, V, step
    return E_cpu, E_recon


def reconstruction_cyclic_torch_boosted(A, iterNum=300, batch_size=None, gamma=0.9, lr=0.1, verbose=True,
                                        final_loss=False, boosted=False):
    '''
    Cyclic reorder rows using stochastic gradient ascent
    :param A: gene expression matrix
    :param iterNum:  iteration number
    :param batch_size: batch size
    :param gamma: momentum parameter
    :return: permutation matrix
    '''
    if batch_size == None:
        batch_size = int((A.shape[0]) * 0.5)
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    V = ge_to_spectral_matrix(A, optimize_alpha=False)
    A = torch.from_numpy(A)
    A = A.type(torch.float32)
    A = A.float()
    V = torch.from_numpy(V.T)
    V = V.type(torch.float32)
    E = torch.ones((n, n))
    E = E.type(torch.float32)
    step = torch.zeros(E.shape)
    step = step.type(torch.float32)
    A = A.to(device)
    E = E.to(device)
    step = step.to(device)
    if ~boosted:
        V = V.to(device)
        print(V)
        E = sga_matrix_momentum_torch(A, E=E / n, V=V, iterNum=iterNum, step=step, batch_size=batch_size, gamma=gamma,
                                      device=device, lr=lr, verbose=verbose)
    else:
        V_1 = V[:, :int(n / 4)]
        V_2 = V[:, int(n / 4):]
        V_boosted = torch.empty((A.shape[0], V_1.shape[1] + V_2.shape[1]))
        V_boosted[:, :int(n / 4)] = V_1
        V_boosted[:, int(n / 4):] = V_2
        del V_1
        del V_2
        del V
        V_boosted = V_boosted.to(device)
        E = sga_matrix_boosted_torch(A, E=E / n, V=V_boosted, iterNum=iterNum, batch_size=batch_size, device=device,
                                     lr=lr, verbose=verbose)
    E = (E.cpu()).numpy()
    E_recon = reconstruct_e(E)
    return E, E_recon


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
    U = U.T
    A = gene_normalization(A)
    T = np.ones((p)) / 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.tensor(A.astype(float), device=device)
    U = torch.tensor(U.astype(float), device=device)
    T = torch.tensor(T.astype(float), device=device)
    D_gpu = gradient_ascent_filter_matrix_torch(A, T=T, U=U, regu=regu, lr=lr, iterNum=iterNum)
    D = D_gpu.cpu().detach().numpy()
    del T
    del U
    del A
    return D


def filter_cyclic_genes_torch(A, regu=0.1, lr=0.1, iterNum=500):
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
    U = U.T
    A = gene_normalization(A)
    T = np.ones((p)) / 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.tensor(A.astype(float), device=device)
    U = torch.tensor(U.astype(float), device=device)
    T = torch.tensor(T.astype(float), device=device)
    D_gpu = gradient_ascent_filter_matrix_torch(A, T=T, U=U, ascent=-1, regu=regu, lr=lr, iterNum=iterNum)
    D = D_gpu.cpu().detach().numpy()
    del T
    del U
    del A
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
    ATUUTA = 2 * ((A.T).mm(U)).mm(U.T).mm(A)
    while (j < iterNum):
        if j % 25 == 1:
            print("Iteration number: ")
            print(j)
        epsilon_t *= 0.995
        # T = D.diagonal()#numba_diagonal(D)#.diagonal()
        # grad = ATUUTA * T - regu * torch.sign(D)
        T = T + ascent * epsilon_t * (ATUUTA * T).diag() - ascent*regu * torch.sign(T)
        T = torch.clip(T, 0, 1)
        j += 1
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    return T.diag()


def gradient_descent_filter_matrix_torch(A, T, U, ascent=1, lr=0.1, regu=0.1, iterNum=400):
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
    ATUUTA = 2 * ((A.T).mm(U)).mm(U.T).mm(A)
    while (j < iterNum):
        if j % 25 == 1:
            print("Iteration number: ")
            print(j)
        epsilon_t *= 0.995
        # T = D.diagonal()#numba_diagonal(D)#.diagonal()
        # grad = ATUUTA * T - regu * torch.sign(D)
        T = T - ascent *( epsilon_t * (ATUUTA * T).diag() - regu * torch.sign(T))
        T = torch.clip(T, 0, 1)
        j += 1
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    return T.diag()


def enhancement_cyclic_torch(A, regu=0.1, iterNum=100, verbosity=25, error=10e-7, optimize_alpha=False,
                             line_search=False):
    ''' Enhancement of cyclic signal
    :param A: Gene expression matrix (reordered according to cyclic ordering)
    :param regu: regularization coefficient
    :param iterNum: iteration number
    :return: filtering matrix
    '''
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A, optimize_alpha=False)
    V = V.T
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.tensor(A, dtype=torch.float32, device=device)
    V = torch.tensor(V, dtype=torch.float32, device=device)
    F_gpu = torch.ones(A.shape, dtype=torch.float32, device=device)
    VVT = (V).mm(V.T)
    del V
    torch.cuda.empty_cache()
    print("starting enhancement on " + str(device))
    F_gpu = stochastic_gradient_ascent_full_torch(A, F_gpu, VVT=VVT, regu=regu, epsilon=0.1, iterNum=iterNum, device=device)
    F = F_gpu.cpu().detach().numpy()
    del F_gpu
    return F


def filtering_cyclic_torch(A, regu=0.1, iterNum=100, verbosity=25, error=10e-7, optimize_alpha=False,
                           line_search=False):
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
    V = ge_to_spectral_matrix(A, optimize_alpha=optimize_alpha)
    V = V.T
    F = torch.ones(A.shape).to(float)
    V = torch.from_numpy(V).to(float)
    A = torch.from_numpy(A).to(float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    V = V.to(device)
    torch.cuda.empty_cache()
    F_gpu = F.to(device)
    print(A.get_device())
    print(device)
    print("starting filtering")
    if line_search:
        F_gpu = gradient_descent_full_line_torch(A, F=F_gpu, V=V, regu=regu, max_evals=iterNum, verbosity=verbosity,
                                                 error=error, device=device)
    else:
        F_gpu = gradient_descent_full_torch(A, F_gpu, VVT=(V).mm(V.T), regu=regu, epsilon=0.1, iterNum=iterNum,
                                            device=device)
    del A
    F = F_gpu.cpu().detach().numpy()
    del F_gpu
    return F


def gradient_descent_full_torch(A, F, VVT, regu, epsilon=0.1, iterNum=400, regu_norm='L1', device='cpu'):
    j = 0
    epsilon_t = epsilon
    while j < iterNum:
        if j % 100 == 1:
            print("Iteration number: " + str(j))
        epsilon_t *= 0.995
        grad = G_full_torch(A=A, B=F, VVT=VVT, regu=regu)
        F = F - epsilon_t * grad
        F = torch.clip(F, 0, 1)
        j += 1
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    return F


def function_and_gradient_full_acc_torch(A, B, V, VVT, regu, regu_norm):
    """
    Computes function value and gradient for the optimization problem with the added
    advantage of precomputation of V.dot(V.T)

    Parameters
    ----------
    A : numpy array
        Gene expression matrix.
    B : numpy array
        Filtering matrix.
    V : numpy array
        Spectral matrix.
    VVT : numpy array
        The precomputed value of V.dot(V.T)
    regu : float
        Regularization coefficient.
    regu_norm : str, optional
        Type of regularization norm ('L1' or 'L2').

    Returns
    -------
    tuple
        A tuple containing the projection over theoretic spectrum and the gradient according to 'B'.
    """
    T_0 = (A * B)
    t_1 = torch.norm(B, 1)
    functionValue = (torch.trace((((V.T).mm(T_0)).mm(T_0.T)).mm(V)) - (regu * t_1))
    gradient = ((2 * ((VVT).mm(T_0) * A)) - ((regu) * torch.sign(B)))
    return functionValue, gradient


def gradient_descent_full_line_torch(A, F, V, regu, gamma=1e-04, max_evals=250, verbosity=float('inf'), error=1e-07,
                                     regu_norm='L1', device='cpu'):
    """
    The function gradient_descent_full_line filters the signal by applying gradient descent method with line search.
    Parameters:
    A (numpy.ndarray): The gene expression matrix.
    F (numpy.ndarray): The filtering matrix.
    V (numpy.ndarray): The theoretic spectrum of covariance matrix.
    regu (float): The regularization coefficient.
    gamma (float, optional): The parameter for line search step size. Default is 1e-04.
    max_evals (int, optional): The maximum number of evaluations for the line search. Default is 250.
    verbosity (float, optional): The verbosity of the output. Default is float('inf').
    error (float, optional): The error tolerance for the line search. Default is 1e-07.
    regu_norm (str, optional): The type of regularization to apply. Default is 'L1'.

    Returns:
    numpy.ndarray: The filtering matrix.
    """
    VVT = V.mm(V.T)
    w = F
    evals = 0
    loss, grad = function_and_gradient_full_acc_torch(A=A, B=F, V=V, VVT=VVT, regu=regu, regu_norm=regu_norm)
    alpha = 1 / torch.norm(grad)
    prev_w = torch.zeros(w.shape, device=device)
    while evals < max_evals and torch.norm(w - prev_w) > error:
        prev_w = torch.clone(w)
        evals += 1
        if evals % verbosity == 0:
            print((evals))
            print('th Iteration    Loss :: ')
            print((loss))
        gTg = torch.norm(grad)
        gTg = gTg * gTg
        new_w = w - alpha * grad
        new_w = torch.clip(new_w, 0, 1)
        new_loss, new_grad = function_and_gradient_full_acc_torch(A=A, B=new_w, V=V,
                                                                  VVT=VVT, regu=regu, regu_norm=regu_norm)
        while new_loss > loss - gamma * alpha * gTg:
            alpha = ((alpha ** 2) * gTg) / (2 * (new_loss + alpha * gTg - loss))
            new_w = w - alpha * grad
            new_w = torch.clip(new_w, 0, 1)
            new_loss, new_grad = function_and_gradient_full_acc_torch(A=A, B=new_w, V=V, VVT=VVT,
                                                                      regu=regu, regu_norm=regu_norm)
        alpha = min(1, 2 * (loss - new_loss) / gTg)
        loss = new_loss
        grad = new_grad
        w = new_w
    del grad, loss, prev_w
    return w


def stochastic_gradient_ascent_full_torch(A, F, VVT, regu, epsilon=0.1, iterNum=400, regu_norm='L1', device='cpu'):
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
        grad = G_full_torch(A=A, B=F, VVT=VVT, regu=regu)
        F = F + epsilon_t * (grad + torch.normal(0, 0.01, grad.shape, device=device))
        F = torch.clip(F, 0, 1)
        j += 1
    del grad
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    return F


def G_full_torch(A, B, VVT, regu):
    '''
    :param A: Gene expression matrix
    :param B: filtering matrix
    :param VVT: original covariance matrix
    :param regu: correlation between neighbors
    :return:projection over theoretic spectrum and gradient according to 'B'
    '''
    T_0 = (A * B)
    gradient = ((2 * ((VVT).mm(T_0) * A)) - ((regu) * torch.sign(B)))
    return gradient


def BBS_torch(E, prev_E, I, ones_m, iterNum=1000):
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
        E = torch.clip(E, min=0, max=1)  # E,E.shape[0],E.shape[0],0)
        if i % 10 == 1:
            if torch.norm(E - prev_E) < ((10e-7) * n * n):
                break
    return E


def enhance_general_topology_torch(A, V, regu=0.5, iterNum=300):
    A = cell_normalization(A)
    F = torch.ones(A.shape)
    V = torch.from_numpy(V)
    A = torch.from_numpy(A)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    V = V.to(device)
    VVT = (V).mm(V.T)
    del V
    torch.cuda.empty_cache()
    F_gpu = F.to(device)
    print(A.get_device())
    print(device)
    print("starting filtering")
    F_gpu = stochastic_gradient_ascent_full_torch(A, F=F_gpu, VVT=VVT, regu=regu, iterNum=iterNum, device=device)
    F = F_gpu.cpu().detach().numpy()
    del F_gpu
    return F


def gene_inference_general_topology_torch(A, V, regu=0.5, iterNum=100, lr=0.1):
    A = gene_normalization(A)
    p = A.shape[1]
    T = np.ones((p)) / 2
    A = torch.from_numpy(A)
    V = torch.from_numpy(V)
    T = torch.from_numpy(T)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    V = V.to(device)
    T = T.to(device)
    D = gradient_ascent_filter_matrix_torch(A, T=T, U=V, ascent=-1, regu=regu, lr=lr, iterNum=iterNum)
    del A
    del V
    del T
    return D.cpu().detach().numpy()


def filter_general_topology_torch(A, V, regu=0.5, iterNum=300):
    A = cell_normalization(A)
    F = torch.ones(A.shape)
    V = torch.from_numpy(V)
    A = torch.from_numpy(A)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    V = V.to(device)
    VVT = (V).mm(V.T)
    del V
    torch.cuda.empty_cache()
    F_gpu = F.to(device)
    print(A.get_device())
    print(device)
    print("starting filtering")
    F_gpu = gradient_descent_full_torch(A, F=F_gpu, VVT=VVT, regu=regu, epsilon=0.1, iterNum=iterNum, device=device)
    F = F_gpu.cpu().detach().numpy()
    del F_gpu
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
    F_gpu = gradient_descent_full_torch(B, F_gpu.type(torch.float), VVT=VVT, regu=regu, epsilon=epsilon,
                                        iterNum=iterNum)
    F = F_gpu.cpu().detach().numpy()
    del F_gpu, VVT , B
    return F


def e_to_range(E):
    """
    Convert the permutation matrix to a range of integers representing the order of rows.
    Parameters
    ----------
    E : numpy array
        The permutation matrix.
    Returns
    -------
    numpy array
        The range of integers representing the order of rows.
    """
    order = []
    for i in range(E.shape[0]):
        for j in range(E.shape[1]):
            if E[i, j] == 1:
                order.append(j)
    return np.array(order)


def reorder_indicator_torch(A, IN, iterNum=300, batch_size=20, gamma=0, lr=0.1):
    """
    Cyclic reorder rows of a gene expression matrix using stochastic gradient ascent. With optional usage of prior knowledge.

    Parameters
    ----------
    A : torch.tensor
        The gene expression matrix.
    IN : torch.tensor
        The indicator matrix.
    iterNum : int, optional
        Number of iterations. Default is 300.
    batch_size : int, optional
        Batch size. Default is 20.
    gamma : float, optional
        Momentum parameter. Default is 0.
    lr : float, optional
        Learning rate. Default is 0.1.

    Returns
    -------
    tuple of torch.tensor
        Permutation matrix (which is calculated by greedy rounding of 'E' matrix).
    """
    if batch_size == None:
        batch_size = int((A.shape[0]) * 0.75)
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    V = ge_to_spectral_matrix(A, optimize_alpha=False)
    A = torch.tensor(A, dtype=torch.float32, device=device)
    IN = torch.tensor(IN, dtype=torch.float32, device=device)
    V = torch.tensor(V, dtype=torch.float32, device=device)
    E = torch.ones((n, n), dtype=torch.float32, device=device)
    step = torch.zeros((n, n), dtype=torch.float32, device=device)
    E = E * IN
    E = sga_matrix_momentum_indicator_torch(A, E, V=V.T, IN=IN, iterNum=iterNum, batch_size=batch_size, gamma=gamma,
                                            lr=lr, step=step, device=device)
    E_cpu = (E.cpu()).numpy()
    E_recon = reconstruct_e(E_cpu)
    del A, E, V, IN, step
    return E_cpu, E_recon


def sga_matrix_momentum_indicator_torch(A, E, V, IN, step, iterNum=400, batch_size=20, lr=0.1, gamma=0.9, device='cpu'):
    '''
    Reconstruction algorithm with optional use of prior knowledge
    Parameters
    ----------
    A: torch.tensor
        Gene expression matrix
    E: torch.tensor
        Initial Bi-Stochastic matrix (should be constant)
    V: torch.tensor:
        Theoretical spectrum
    IN: torch.tensor:
        Indicator matrix which will be later entry-wise multiplied by the permutation matrix
    iterNum: int
        Number of stochastic gradient ascent iterations
    batch_size: int
        batch size, number of genes sampled per batch
    lr: float
        Learning rate
    gamma: float
        Momentum parameter

    Returns
        E: torch.tensor
            Bi-Stochastic matrix

    -------
    '''
    j = 0
    prev_E = torch.empty(E.shape, dtype=torch.float32)
    prev_E = prev_E.to(device)
    I = torch.eye(E.shape[0], dtype=torch.float32)
    I = I.to(device)
    ones_m = torch.ones((E.shape[0], E.shape[0]), dtype=torch.float32)
    ones_m = ones_m.to(device)
    E = BBS_torch(E=E, prev_E=prev_E, I=I, ones_m=ones_m) * IN
    value = 0
    epsilon_t = lr
    while (j < iterNum):
        if j % 25 == 0:
            print("Iteration number: " + str(j) + " function value= " + str(value))
        A_tmp = A[:, torch.randint(low=0, high=A.shape[1], size=(batch_size,))]
        value, grad = function_and_gradient_matrix_torch(A=A_tmp, E=E, V=V)
        grad = grad
        step = epsilon_t * grad + gamma * step
        E = E + step
        E = BBS_torch(E=E, prev_E=prev_E, I=I, ones_m=ones_m) * IN
        j += 1
    return E



def sort_data_crit(adata, crit, crit_list):
    '''
    Sort the cells of an AnnData object according to a field (obs)
    :param adata: AnnData to be sorted
    :param crit: 'obs' field
    :param crit_list: list of 'obs' possible values, sorted according to the desired ordering (e.g ['0','6','12','18])
    :return:
    '''
    adata = shuffle_adata(adata)  # for avoiding batch effects
    layers = [[] for i in range(len(crit_list))]
    obs = adata.obs
    for i, row in obs.iterrows():
        layer = (row[crit])
        for j, item in enumerate(crit_list):
            if item == layer:
                layers[j].append(i)
    order = sum(layers, [])
    sorted_data = adata[order, :]
    return sorted_data


def shuffle_adata(adata):
    '''
    Shuffle the rows(obs/cells) of adata
    :param adata: adata
    :return: shuffled adata
    '''
    perm = np.random.permutation(range(adata.X.shape[0]))
    return adata[perm, :]


def filter_linear_full_torch(A: np.ndarray, method: str, regu: float = 0.1, iterNum: int = 300, lr: float = 0.1,
                             verbosity=25, error=10e-7,
                             regu_norm: str = 'L1', line_search=False) -> np.ndarray:
    """
    Apply linear filtering on the full data.

    Parameters
    ----------
    A : np.ndarray
        gene expression matrix
    method : str
        method to get eigenvectors
    regu : float, optional
        regularization coefficient, by default 0.1
    iterNum : int, optional
        iteration number, by default 300
    lr : float, optional
        learning rate, by default 0.1
    regu_norm : str, optional
        Regularization norm, by default 'L1'

    Returns
    -------
    np.ndarray
        filtered matrix
    """
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(ngenes=A.shape[1], optimized_alpha=False, eigenvals=real_eigenvalues)
    V = get_linear_eig_data(A.shape[0], alpha, method=method,
                            normalize_vectors=True)
    V = V[:, 1:]
    F = torch.ones(A.shape).to(float)
    V = torch.from_numpy(V).to(float)
    A = torch.from_numpy(A).to(float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    V = V.to(device)
    torch.cuda.empty_cache()
    F_gpu = F.to(device)
    print(A.get_device())
    print(device)
    print("starting filtering")
    if line_search:
        F_gpu = gradient_descent_full_line_torch(A, F=F_gpu, V=V, regu=regu, max_evals=iterNum, verbosity=verbosity,
                                                 error=error, device=device)
    else:
        F_gpu = gradient_descent_full_torch(A, F_gpu, VVT=(V).mm(V.T), regu=regu, epsilon=0.1, iterNum=iterNum,
                                            device=device)
    del A
    F = F_gpu.cpu().detach().numpy()
    del F_gpu
    return F

def filter_linear_genes_torch(A: np.ndarray, regu: float=0.1, iterNum: int=500, verbosity: int=25, method: str='numeric', lr=0.1) -> np.ndarray:
    """
    Filter linear genes from gene expression matrix using line-search gradient descent method.

    :param A: gene expression matrix
    :param regu: regularization parameter
    :param iterNum: iteration number
    :param lr: learning rate
    :return: diagonal filtering matrix
    """
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(optimized_alpha=True, eigenvals=real_eigenvalues)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    U = eigenvectors[:, 1:]
    A = gene_normalization(A)
    T = np.ones((p)) / 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = torch.tensor(A.astype(float), device=device)
    U = torch.tensor(U.astype(float), device=device)
    T = torch.tensor(T.astype(float), device=device)
    D_gpu = gradient_ascent_filter_matrix_torch2(A, T=T, U=U, ascent=-1, regu=regu, lr=lr, iterNum=iterNum)
    D = D_gpu.cpu().detach().numpy()
    del T
    del U
    del A
    return D

def enhancement_linear_torch(A: np.ndarray, regu: float = 0.1, iterNum: int = 300, method: str = 'numeric') -> np.ndarray:
    """
    Enhancement of linear signal
    Parameters
    ----------
    A : np.ndarray
        Gene expression matrix (reordered according to linear ordering)
    regu : float, optional
        regularization coefficient, by default 0.1
    iterNum : int, optional
        iteration number, by default 300

    Returns
    -------
    np.ndarray
        filtering matrix
    """
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(optimized_alpha=True, eigenvals=real_eigenvalues)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    V = eigenvectors[:,1:]
    F = torch.ones(A.shape)
    V = torch.from_numpy(V)
    A = torch.from_numpy(A)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    V = V.to(device)
    VVT = (V).mm(V.T)
    del V
    torch.cuda.empty_cache()
    F_gpu = F.to(device)
    print(A.get_device())
    print(device)
    print("starting filtering")
    F_gpu = stochastic_gradient_ascent_full_torch(A, F_gpu, VVT=VVT, regu=regu, epsilon=0.1, iterNum=iterNum, device=device)
    F = F_gpu.cpu().detach().numpy()
    del F_gpu
    return F

def filter_genes_by_proj_torch(A: torch.tensor, V: torch.tensor, n_genes: int = None, percent_genes: float = None, device='cpu') -> np.ndarray:
    """
    Filters genes from a matrix A based on a projection matrix V.
    If n_genes is not provided, the function will select the top half of the genes by default.
    If percent_genes is not provided, the function will select the top n_genes genes by default.

    Parameters
    ----------
    A : torch.tensor
        A matrix of shape (n,p) where m is the number of samples and n is the number of genes.
    V : torch.tensor
        A matrix of shape (k,n) where k is the number of components to project onto.
    n_genes : int, optional
        The number of genes to select, by default None
    percent_genes : float, optional
        The percent of genes to select, by default None

    Returns
    -------
    np.ndarray
        A matrix of shape (p,p) where the top n_genes or percent_genes genes are set to 1 and the rest are set to 0.

    """
    if n_genes==None and percent_genes==None:
        n_genes=int(A.shape[0]/2)
    elif  n_genes==None:
        if percent_genes<1 and percent_genes>0:
            n_genes= int(A.shape[0] * percent_genes)
        else:
            print("percent_genes should be between 0 and 1")
            return None
    score_array = np.zeros(A.shape[1])
    for i in range(A.shape[1]):
        gene = A[:,i]
        score_array[i]= torch.trace((V.T).mm(torch.outer(gene,gene)).mm(V)).cpu().numpy()
    x = np.argsort(score_array)[::-1][:n_genes]
    D = np.zeros((A.shape[1],A.shape[1]))
    D[x,x]=1
    return D

def filter_non_cyclic_genes_by_proj_torch(A: np.ndarray,  n_genes: int = None, percent_genes: float = None) -> np.ndarray:
    """
    Filters non cyclic genes from a matrix A.
    If n_genes is not provided, the function will select the top half of the genes by default.
    If percent_genes is not provided, the function will select the top n_genes genes by default.

    Parameters
    ----------
    A : np.ndarray
        A matrix of shape (n,p) where m is the number of samples and n is the number of genes.
    n_genes : int, optional
        The number of genes to select, by default None
    percent_genes : float, optional
        The percent of genes to select, by default None

    Returns
    -------
    np.ndarray
        A matrix of shape (p,p) where the top n_genes or percent_genes genes are set to 1 and the rest are set to 0.

    """
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A)
    A = gene_normalization(A)
    V=V.T
    V = torch.from_numpy(V).to(float)
    A = torch.from_numpy(A).to(float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    V = V.to(device)
    D =  filter_genes_by_proj_torch(A=A, V=V, n_genes=n_genes, percent_genes=percent_genes)
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

def shuffle_adata(adata):
    '''
    Shuffle the rows(obs/cells) of adata
    :param adata: adata
    :return: shuffled adata
    '''
    perm = np.random.permutation(range(adata.X.shape[0]))
    return adata[perm,:]

def filter_general_genes_by_proj_torch(A: np.ndarray, cov: np.ndarray, n_genes: int = None, percent_genes: float = None) -> np.ndarray:
    """
    Filters genes from a matrix A based on their projection over the theoretic spectrum of covaraince matrix cov.
    If n_genes is not provided, the function will select the top half of the genes by default.
    If percent_genes is not provided, the function will select the top n_genes genes by default.

    Parameters
    ----------
    A : np.ndarray
        A matrix of shape (n,p) where m is the number of samples and n is the number of genes.
    cov : np.ndarray
        A matrix of shape (n,n), the theoretical covaraince matrix.
    n_genes : int, optional
        The number of genes to select, by default None
    percent_genes : float, optional
        The percent of genes to select, by default None

    Returns
    -------
    np.ndarray
        A matrix of shape (p,p) where the top n_genes or percent_genes genes are set to 1 and the rest are set to 0.

    """
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    V = get_theoretic_eigen(cov)
    A = gene_normalization(A)
    V = torch.from_numpy(V).to(float)
    A = torch.from_numpy(A).to(float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    A = A.to(device)
    V = V.to(device)
    D =  filter_genes_by_proj_torch(A=A, V=V, n_genes=n_genes, percent_genes=percent_genes, device=device)
    return D

def enhance_general_covariance_torch(A, cov, regu=0, epsilon=0.1, iterNum=100, regu_norm='L1', device='cpu'):
    """
    Enhances a signal based on a eigendecomposition of the theoretical covariance matrix describes the signal by running stochastic gradient ascent with L1 regularization.

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
    F_gpu = stochastic_gradient_ascent_full_torch(B, F_gpu, VVT=VVT, regu=regu, epsilon=epsilon, iterNum=iterNum)
    del A, VVT
    F = F_gpu.cpu().detach().numpy()
    del F_gpu
    return F


def gradient_ascent_filter_matrix_torch2(A, D, U, ascent, lr,regu, iterNum, verbose= True):
    """
    Finds the diagonal filter matrix by gradient ascent/descent optimization.

    Parameters
    ----------
    A : np.ndarray
        Gene expression matrix.
    D : np.ndarray
        Diagonal filter matrix (initial value).
    U : np.ndarray
        Eigenvectors matrix multiple by sqrt of diagonal eigenvalues matrix.
    ascent : int, optional
        1 - gradient ascent , -1 - gradient decent, by default 1.
    lr : float, optional
        Learning rate, by default 0.1.
    regu : float, optional
        Regularization parameter, by default 0.1.
    iterNum : int, optional
        Number of iterations, by default 400.
    verbose : bool, optional
        Prints the iteration number and function value if true, by default True.

    Returns
    -------
    np.ndarray
        Diagonal filter matrix.
    """
    j = 0
    val = 0
    epsilon_t = lr
    ATUUTA = (2 * ((((A.T).mm(U)).mm(U.T)).mm(A)))
    while (j < iterNum):
        if j % 25 == 1:
            if verbose:
                print("Iteration number: ")
                print(j)
                print("function value= ")
                print(val)
        epsilon_t *= 0.995
        T = D.diagonal()  # .diagonal()
        grad = ATUUTA * T - regu * torch.sign(D)
        D = D + ascent * epsilon_t * grad
        D = D.diagonal()
        D = torch.clip(D,0,1)
        D = D.diagonal()
        j += 1
    return D
