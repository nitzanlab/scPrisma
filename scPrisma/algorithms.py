
import numpy as np
from numpy import random
from sklearn.metrics import roc_auc_score

from scPrisma.data_gen import simulate_spatial_cyclic
from scPrisma.pre_processing import *
from numba import jit


@jit(nopython=True, parallel=True)
def numba_diagonal(A):
    d = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        d[i] = A[i, i]
    return d


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
            if E_prob[i, j] > tmp:
                if not (j in res):
                    tmp = E_prob[i, j]
                    pointer = j
        res.append(pointer)
    res_array = np.zeros(E_prob.shape)
    for i, item in enumerate(res):
        res_array[i, item] = 1
    return res_array

@jit(nopython=True, parallel=True)
def boosted_reconstruct_e(E_prob):
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
    res_array = np.zeros(E_prob.shape)
    for i in range(E_prob.shape[0]):
        tmp = -1
        pointer = -1
        for j in range(E_prob.shape[1]):
            if E_prob[i, j] > tmp and not (j in res):
                tmp = E_prob[i, j]
                pointer = j
        res.append(pointer)
        res_array[i, pointer] = 1
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
    # Removing the eigenvector which is related to the largest eigenvalue improve the results
    # This eigenvector is the 'offset' of the data
    V = V[1:, :]
    return V


def sga_reorder_rows_matrix(A, iterNum=300, batch_size=20):
    '''
    Reconstruction algorithm (without momentum)
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    iterNum: int
        Number of stochastic gradient ascent iterations
    batch_size: int
        batch size, number of genes sampled per batch

    Returns
    E: np.array
        Bi-Stochastic matrix
    E_recon: np.array
        Permutation matrix (which is calculated by greedy rounding of 'E' matrix).

    -------

    '''
    A = cell_normalization(A)
    n = A.shape[0]
    V = ge_to_spectral_matrix(A)
    E = sga_matrix(A, E=np.ones((n, n)) / n, V=V.T, iterNum=iterNum, batch_size=batch_size)
    E_recon = reconstruct_e(E)
    return E , E_recon


@jit(nopython=True, parallel=True)
def sga_matrix(A, E, V, lr, iterNum, batch_size):
    """
    Perform stochastic gradient descent optimization to find the optimal value of the bi-stochastic matrix E.

    Parameters
    ----------
    A : numpy array, shape (n, p)
        A gene expression matrix of inputs, where p is the number of genes and n is the number of cells.
    E : numpy array, shape (n, n)
        The current value of the bi-stochastic matrix E to be optimized, where n is the number of cells.
    V : numpy array, shape (n, n-1)
        Theoretical spectrum
    lr : float
        The initial learning rate for the gradient descent updates.
    iterNum : int
        The number of iterations to perform the optimization for.
    batch_size : int
        The number of examples to use in each batch for the gradient descent updates.

    Returns
    -------
    E : numpy array, shape (n, n)
        The optimized value of the bi-stochastic matrix E.
    """
    # Initialize loop counter and function value
    j = 0
    value = 0

    # Set the initial step size for the gradient descent updates
    epsilon_t = lr

    # Begin loop to perform gradient descent optimization
    while (j < iterNum):
        # Print iteration number and function value every 25 iterations
        if j % 25 == 0:
            print("Iteration number: " + str(j) + " function value= " + str(value))
            print("Iteration number: " + str(j))

        # Decrease the step size for the gradient descent updates
        epsilon_t *= 0.995

        # Select a batch of columns from the matrix A
        A_tmp = A[:, np.random.randint(A.shape[1], size=batch_size)]

        # Evaluate the function and its gradient at the current value of E
        value, grad = function_and_gradient_matrix(A=A_tmp, E=E, V=V)

        # Perform a gradient descent update on E
        E = E + epsilon_t * grad

        # Apply the BBS function to E (Bi-Stochastic projection)
        E = BBS(E)

        # Print the iteration number and function value
        print("Iteration number: " + str(j) + " function value= " + str(value))

        # Increment loop counter
        j += 1

    # Return the optimized value of E
    return E


@jit(nopython=True, parallel=True)
def function_and_gradient_matrix(A, E, V):
    '''
    Calculate the function value and the gradient of A matrix
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    E: np.array
        Bi-Stochastic matrix (should be constant)
    V: np.array:
        Theoretical spectrum

    Returns
    -------
    functionValue: float
        function value
    gradient: np.array
        gradient of E
    '''
    functionValue = np.trace((((((V.T).dot(E)).dot(A)).dot(A.T)).dot(E.T)).dot(V))
    gradient = (2 * ((((V).dot(V.T)).dot(E)).dot(A)).dot(A.T))

    return functionValue, gradient


@jit(nopython=True, parallel=True)
def g_matrix(A, E, VVT):
    '''
    Calculate the gradient of A matrix, using boosted formula
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    E: np.array
        Bi-Stochastic matrix (should be constant)
    VVT: np.array:
        Theoretical spectrum (V) multiplied by his transform (V.T)

    Returns
    -------
    gradient: np.array
        gradient of E
    '''
    gradient = (2 * (((VVT).dot(E)).dot(A)).dot(A.T))
    return gradient


@jit(nopython=True, parallel=True)
def sga_matrix_momentum(A, E, V, iterNum=400, batch_size=20, lr=0.1, gamma=0.9, verbose=True):
    """
    Perform stochastic gradient ascent optimization with momentum to find the optimal value of the bi-stochastic matrix E.

    Parameters
    ----------
    A : numpy array, shape (n, p)
        A matrix of inputs, where p is the number of genes and n is the number of cells.
    E : numpy array, shape (n, n)
        The current value of the bi-stochastic matrix E to be optimized, where n is the number of cells.
    V : numpy array, shape (n, n-1)
        Theoretical spectrum.
    iterNum : int, optional
        The number of iterations to perform the optimization for. Default is 400.
    batch_size : int, optional
        The number of examples to use in each batch for the gradient descent updates. Default is 20.
    lr : float, optional
        The learning rate for the gradient ascent updates. Default is 0.1.
    gamma : float, optional
        The momentum parameter. Default is 0.9.
    verbose : bool, optional
        A flag indicating whether to print the iteration number and function value every 25 iterations. Default is True.

    Returns
    -------
    E : numpy array, shape (n, n)
        The optimized value of the bi-stochastic matrix E.
    """
    # Initialize loop counter and function value
    j = 0
    value = 0

    # Pre-compute the product VVT for runtime optimization
    VVT = (V).dot(V.T)

    # Set the initial step size for the gradient ascent updates
    epsilon_t = lr

    # Initialize the momentum step
    step = np.zeros(E.shape)

    # Begin loop to perform gradient ascent optimization with momentum
    while (j < iterNum):
        # Print iteration number and function value every 25 iterations if verbose flag is set
        if (j % 25 == 0) & verbose:
            value, grad = function_and_gradient_matrix(A=A, E=E, V=V)
            print("Iteration number: ")
            print(j)
            print(" function value= ")
            print(value)

        # Select a batch of columns from the matrix A
        A_tmp = A[:, np.random.randint(0, A.shape[1], batch_size)]

        # Evaluate the gradient at the current value of E
        grad = g_matrix(A=A_tmp, E=E, VVT=VVT)

        # Update the momentum step
        step = epsilon_t * grad + gamma * step

        # Perform a gradient ascent update on E
        E = E + step

        # Apply the BBS function to E
        E = BBS(E)

        # Increment loop counter
        j += 1

    # Return the optimized value of E
    return E


def sga_matrix_momentum_indicator(A, E, V, IN, iterNum=400, batch_size=20, lr=0.1, gamma=0.9):
    '''
    Reconstruction algorithm with optional use of prior knowledge
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    E: np.array
        Initial Bi-Stochastic matrix (should be constant)
    V: np.array:
        Theoretical spectrum
    IN: np.array:
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
        E: np.array
            Bi-Stochastic matrix

    -------

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
        value, grad = function_and_gradient_matrix(A=A_tmp, E=E, V=V)
        grad = grad
        step = epsilon_t * grad + gamma * step
        E = E + step
        E = BBS(E) * IN
        j += 1
    return E


def sga_m_reorder_rows_matrix(A, iterNum=300, batch_size=None, gamma=0.5, lr=0.1):
    '''
    Cyclic reorder rows using stochastic gradient ascent
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    iterNum: int
        Number of stochastic gradient ascent iterations
    batch_size: int
        batch size, number of genes sampled per batch
    lr: float
        Learning rate
    gamma: float
        Momentum parameter

    Returns
    E: np.array
        Bi-Stochastic matrix
    E_recon: np.array
        Permutation matrix (which is calculated by greedy rounding of 'E' matrix).
    -------

    '''
    if batch_size == None:
        batch_size = int((A.shape[0]) * 0.75)
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V = ge_to_spectral_matrix(A)
    E = sga_matrix_momentum(A, E=np.ones((n, n)) / n, V=V.T, iterNum=iterNum, batch_size=batch_size, gamma=gamma, lr=lr)
    E_recon = reconstruct_e(E)
    return E, E_recon


def reconstruction_cyclic(A, iterNum=300, batch_size=None, gamma=0.5, lr=0.1, verbose=True, final_loss=False):
    '''
    Cyclic reorder rows using stochastic gradient ascent
    Parameters
    ----------
    A: np.array
        Gene expression matrix
    iterNum: int
        Number of stochastic gradient ascent iterations
    batch_size: int
        batch size, number of genes sampled per batch
    lr: float
        Learning rate
    gamma: float
        Momentum parameter
    verbose: bool
        verbosity
    final_loss: bool
        For validation, retain False
    Returns
    E: np.array
        Bi-Stochastic matrix
    E_recon: np.array
        Permutation matrix (which is calculated by greedy rounding of 'E' matrix).
    -------

    '''
    A = np.array(A).astype('float64')
    if batch_size == None:
        batch_size = int((A.shape[0]) * 0.75)
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    V = ge_to_spectral_matrix(A)
    E = sga_matrix_momentum(A, E=np.ones((n, n)) / n, V=V.T, iterNum=iterNum, batch_size=batch_size, gamma=gamma, lr=lr, verbose=verbose)
    E_recon = reconstruct_e(E)
    if final_loss:
        value, grad = function_and_gradient_matrix(A=((1 / A.shape[0]) * A), E=E_recon, V=V.T)
        return E, E_recon, value
    return E, E_recon


def filter_non_cyclic_genes(A, regu=0.1, lr=0.1, iterNum=500) -> np.array:
    '''
    Filter out genes which are not smooth over the inferred circular topology. As a prior step for this algorithm, the reconstruction algorithm should be applied.

    Parameters
    ----------
    A: np.array
        Gene expression matrix
    regu: float
        Regularization coefficient, large regularization would lead to more non-cyclic genes which will be filtered out
    lr: float
        Learning rate
    iterNum: int
        Number of stochastic gradient ascent iterations

    Returns
    -------
    D: np.array
        diagonal filtering matrix
    '''
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    n = A.shape[0]
    p = A.shape[1]
    U = ge_to_spectral_matrix(A)
    A = gene_normalization(A)
    D = gradient_ascent_filter_matrix(A, D=np.identity((p)) / 2, U=U.T, regu=regu, lr=lr, iterNum=iterNum)
    return D


def filter_cyclic_genes(A, regu=0.1, iterNum=500, lr=0.1, verbose=True):
    '''
    Filter out genes which are smooth over the inferred circular topology. As a prior step for this algorithm, the reconstruction algorithm should be applied.

    Parameters
    ----------
    A: np.array
        Gene expression matrix
    regu: float
        Regularization coefficient, large regularization would lead to more cyclic genes which will be filtered out
    lr: float
        Learning rate
    iterNum: int
        Number of stochastic gradient ascent iterations

    Returns
    -------
    D: np.array
        diagonal filtering matrix

    '''
    A = np.array(A).astype('float64')
    V = cell_normalization(A)
    p = V.shape[1]
    U = ge_to_spectral_matrix(V)
    A = gene_normalization(A)
    D = gradient_ascent_filter_matrix(A, D=np.identity((p)) / 2, ascent=-1, U=U.T, regu=regu, iterNum=iterNum, lr=lr,
                                      verbose=verbose)
    return D


def filter_cyclic_genes_line(A: np.ndarray, regu: float=0.1, iterNum: int=500,  verbosity: int=25) -> np.ndarray:
    """
    Filter cyclic genes from gene expression matrix using line-search gradient descent method.

    Parameters
    ----------
    A : numpy.ndarray
        Gene expression matrix.
    regu : float, optional
        Regularization parameter.
    iterNum : int, optional
        Number of iterations.
    verbosity : int, optional
        Verbosity of the gradient descent algorithm.

    Returns
    -------
    numpy.ndarray
        Diagonal filtering matrix.
    """

    A = np.array(A, dtype=np.float64)
    V = cell_normalization(A)
    p = V.shape[1]
    U = ge_to_spectral_matrix(V)
    A = gene_normalization(A)
    D = gradient_descent_filter_matrix_line(A, D=np.identity((p)), U=U.T, regu=regu, max_evals=iterNum,
                                            verbosity=verbosity)
    return D


def filter_linear_genes_line(A: np.ndarray, regu: float=0.1, iterNum: int=500, verbosity: int=25, method: str='numeric') -> np.ndarray:
    """
    Filter linear genes from gene expression matrix using line-search gradient descent method.

    Parameters
    ----------
    A : numpy.ndarray
        Gene expression matrix.
    regu : float, optional
        Regularization parameter.
    iterNum : int, optional
        Number of iterations.
    verbosity : int, optional
        Verbosity of the gradient descent algorithm.
    method : str, optional
        Method to use to compute eigenvectors. 'numeric' or 'analytic'

    Returns
    -------
    numpy.ndarray
        Diagonal filtering matrix.
    """

    A = np.array(A, dtype=np.float64)
    A = cell_normalization(A)
    p = A.shape[1]
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(optimized_alpha=True, eigenvals=real_eigenvalues)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    A = gene_normalization(A)
    D = gradient_descent_filter_matrix_line(A, D=np.identity((p)), U=eigenvectors[:, 1:], regu=regu, max_evals=iterNum,
                                            verbosity=verbosity)
    return D


def filter_non_cyclic_genes_line(A: np.ndarray, regu: float = 0.1, iterNum: int = 500,
                                 verbosity: int = 25) -> np.ndarray:
    """
    Filter out non-cyclic genes from the gene expression matrix.

    Parameters
    ----------
    A : np.ndarray
        Gene expression matrix.
    regu : float, optional
        Regularization parameter, by default 0.1.
    iterNum : int, optional
        Number of iterations, by default 500.
    verbosity : int, optional
        Verbosity level, by default 25.

    Returns
    -------
    np.ndarray
        Diagonal filtering matrix.
    """
    A = np.array(A).astype('float64')
    V = cell_normalization(A)
    p = V.shape[1]
    U = ge_to_spectral_matrix(V)
    A = gene_normalization(A)
    D = gradient_descent_filter_matrix_line(A, D=np.identity((p)), U=U.T, regu=regu, max_evals=iterNum,
                                            verbosity=verbosity)
    np.identity(D.shape[1])
    return (np.identity(D.shape[1]) - D)


@jit(nopython=True, parallel=True)
def gradient_ascent_filter_matrix(A: np.ndarray, D: np.ndarray, U: np.ndarray, ascent: int = 1, lr: float = 0.1,
                                  regu: float = 0.1, iterNum: int = 400, verbose: bool = True) -> np.ndarray:
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
    ATUUTA = (2 * ((((A.T).dot(U)).dot(U.T)).dot(A)))
    while (j < iterNum):
        if j % 25 == 1:
            if verbose:
                val = np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - regu * np.linalg.norm(D, 1)
                print("Iteration number: ")
                print(j)
                print("function value= ")
                print(val)
        epsilon_t *= 0.995
        T = numba_diagonal(D)  # .diagonal()
        grad = ATUUTA * T - regu * np.sign(D)
        D = D + ascent * epsilon_t * grad
        D = diag_projection(D)
        j += 1
    return D


# def loss_filter_genes(A,U,D,regu):
#    return np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - regu*np.linalg.norm(D,1)

@jit(nopython=True, parallel=True)
def loss_filter_genes(ATU, D, regu):
    D_diag = numba_diagonal(D)
    return np.trace((ATU.T * D_diag * D_diag).dot(ATU)) - regu * np.linalg.norm(D, 1)


@jit(nopython=True, parallel=True)
def gradient_descent_filter_matrix_line(A, D, U, regu=0.1, gamma=1e-04, max_evals=250, verbosity=float('inf')):
    """
    Perform gradient descent on a filter matrix using a line search algorithm.

    Parameters
    ----------
    A : numpy.ndarray
        A matrix representing gene expression data.
    D : numpy.ndarray
        A diagonal filter matrix (initial value).
    U : numpy.ndarray
        A matrix representing eigenvectors multiplied by the square root of the diagonal eigenvalues matrix.
    regu : float, optional
        The regularization parameter (default is 0.1).
    gamma : float, optional
        A parameter for the line search algorithm (default is 1e-04).
    max_evals : int, optional
        The maximum number of iterations (default is 250).
    verbosity : float, optional
        A parameter for controlling the frequency of print statements during the iterations (default is inf).

    Returns
    -------
    numpy.ndarray
        The updated diagonal filter matrix.
    """
    ATUUTA = (2 * ((((A.T).dot(U)).dot(U.T)).dot(A)))
    w = D
    evals = 0
    ATU = (A.T).dot(U)
    loss = loss_filter_genes(ATU=ATU, D=w, regu=regu)
    w = diag_projection(w)

    grad = ATUUTA * w - regu * np.sign(w)
    G = numba_diagonal(grad)  # .diagonal()
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
        gTg = np.linalg.norm(grad)
        gTg = gTg * gTg
        new_w = w - alpha * grad
        new_loss = loss_filter_genes(ATU, new_w, regu)
        new_w = diag_projection(new_w)
        new_grad = ATUUTA * new_w - regu * np.sign(new_w)
        G = numba_diagonal(new_grad)  # .diagonal()
        new_grad = np.diag(G)
        while new_loss > loss - gamma * alpha * gTg:
            alpha = ((alpha ** 2) * gTg) / (2 * (new_loss + alpha * gTg - loss))
            new_w = w - alpha * grad
            new_loss = loss_filter_genes(ATU, new_w, regu)
            new_w = diag_projection(new_w)
            new_grad = ATUUTA * new_w - regu * np.sign(new_w)
            G = numba_diagonal(new_grad)  # .diagonal()
            new_grad = np.diag(G)

        alpha = min(1, 2 * (loss - new_loss) / gTg)
        loss = new_loss
        grad = new_grad
        w = new_w
    return w

@jit(nopython=True, parallel=True)
def function_and_gradient_filter_matrix(A, D, U, alpha):
    functionValue = (np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - (alpha * np.sum(np.abs(D))))
    gradient = ((2 * ((((A.T).dot(U)).dot(U.T)).dot(A)).dot(D)) - (alpha * np.sign(D)))
    return gradient, functionValue


@jit(nopython=True, parallel=True)
def function_and_gradient_fixed_filter(A, D, U, regu):
    t_0 = np.linalg.norm((A).dot(D), 'fro')
    functionValue = np.trace((((((U.T).dot(A)).dot(D)).dot(D.T)).dot(A.T)).dot(U)) - regu * t_0
    gradient = (2 * ((((A.T).dot(U)).dot(U.T)).dot(A)).dot(D)) - regu * np.sign(
        D)

    return functionValue, gradient


@jit(nopython=True, parallel=True)
def gradient_ascent_filter(A, D, eigenvectors_list, eigenvalues_list, epsilon=0.1, regu=0.1, iterNum=400):
    """
    Perform gradient ascent on a filter matrix.

    Parameters
    ----------
    A : numpy.ndarray
        A matrix representing gene expression data.
    D : numpy.ndarray
        A diagonal filter matrix (initial value).
    eigenvectors_list : list of numpy.ndarray
        A list of eigenvectors.
    eigenvalues_list : list of float
        A list of eigenvalues.
    epsilon : float, optional
        A parameter for the gradient ascent algorithm (default is 0.1).
    regu : float, optional
        The regularization parameter (default is 0.1).
    iterNum : int, optional
        The number of iterations (default is 400).

    Returns
    -------
    numpy.ndarray
        The updated diagonal filter matrix.
    """
    j = 0
    epsilon_t = epsilon
    while (j < iterNum):
        value = 0
        if j % 25 == 0:
            print("Iteration number: " + str(j))
        epsilon_t *= 0.995
        grad = np.zeros(D.shape)
        for i, v in enumerate(eigenvectors_list):
            tmp_value, tmp_grad = function_and_gradient_regu(A=A, E=D, regu=regu, x=v * eigenvalues_list[i])
            grad += tmp_grad
            value += tmp_value
        D = D + epsilon_t * grad
        D = diag_projection(D)
        print("Iteration number: " + str(j) + " function value= " + str(value))
        j += 1
    return D


@jit(nopython=True)
def diag_projection(D):
    T = numba_diagonal(D)  # .diagonal()
    T = numba_vec_clip(T, len(T), 0, 1)
    return np.diag(T)


@jit(nopython=True, parallel=True)
def function_and_gradient_regu(A, E, regu, x):
    """
    The function function_and_gradient_regu calculates the function value and gradient of the regularized function defined as:
    f(x) = x^T A E E^T A^T x - regu * ||E||_F
    g(x) = 2 A E E^T A^T x - (regu / ||E||_F) * E

    Parameters:
    A (numpy.ndarray): The matrix A.
    E (numpy.ndarray): The matrix E.
    regu (float): The regularization parameter.
    x (numpy.ndarray): The input vector x.

    Returns:
    tuple: A tuple containing the function value and gradient.
    """
    t_0 = (A.T).dot(x)
    t_1 = np.linalg.norm(E, 'fro')
    functionValue = ((x).dot((A).dot((E).dot((E.T).dot(t_0)))) - (regu * t_1))
    gradient = ((2 * np.multiply.outer(t_0, ((x).dot(A)).dot(E))) - ((regu / t_1) * E))
    return functionValue, gradient


from sklearn.metrics import roc_auc_score


def calculate_roc_auc(y_target, y_true):
    """
    The function calculate_roc_auc calculates the area under the receiver operating characteristic curve
    (ROC AUC) score for binary classification problem.

    Parameters:
    y_target (numpy.ndarray): The predicted target values.
    y_true (numpy.ndarray): The true target values.

    Returns:
    float: The ROC AUC score.
    """
    return roc_auc_score(y_true, y_target)


def filter_full(A, regu=0.1, iterNum=300):
    """
    This function filters out the reconstructed cyclic signal from gene expressio matrix by applying stochastic gradient ascent method.

    Parameters:
    A (numpy.ndarray): The full gene expression matrix.
    regu (float, optional): The regularization parameter. Default is 0.1.
    iterNum (int, optional): The number of iterations for the gradient ascent. Default is 300.

    Returns:
    numpy.ndarray: Filtering matrix.
    """
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A)
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=V.T, regu=regu, iterNum=iterNum)
    return F


def enhancement_cyclic(A, regu=0.1, iterNum=300, verbosity=25):
    """
    The function enhancement_cyclic enhances the cyclic signal by applying stochastic gradient ascent method.
    Parameters:
    A (numpy.ndarray): The gene expression matrix reordered according to cyclic ordering.
    regu (float, optional): The regularization coefficient. Default is 0.1.
    iterNum (int, optional): The number of iterations for the gradient ascent. Default is 300.
    verbosity (int, optional): The verbosity level. Default is 25.

    Returns:
    numpy.ndarray: Enhancement matrix.
    """
    A = np.array(A, dtype='float64')
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A)
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=V.T, regu=regu, iterNum=iterNum, verbosity=verbosity)
    return F


def filtering_cyclic(A, regu=0.1, iterNum=300, verbosity=25, error=10e-7, optimize_alpha=True, line_search=True):
    """
    The function filtering_cyclic filters the cyclic signal by applying gradient descent method.
    Parameters:
    A (numpy.ndarray): The gene expression matrix reordered according to cyclic ordering.
    regu (float, optional): The regularization coefficient. Default is 0.1.
    iterNum (int, optional): The number of iterations for the gradient descent. Default is 300.
    verbosity (int, optional): The verbosity level. Default is 25.
    error (float, optional): The stopping criteria for the gradient descent. Default is 10e-7.
    optimize_alpha (bool, optional): Whether to optimize the regularization parameter. Default is True.
    line_search (bool, optional): Whether to use line search. Default is True.

    Returns:
    numpy.ndarray: Filtering matrix.
    """
    A = np.array(A, dtype='float64')
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A, optimize_alpha=optimize_alpha)
    print("starting filtering")
    if line_search:
        F = gradient_descent_full_line(A, F=np.ones(A.shape), V=V.T, regu=regu, max_evals=iterNum, verbosity=verbosity,
                                       error=error)
    else:
        F = gradient_descent_full(A, np.ones(A.shape), V=V.T, regu=regu, epsilon=0.1, iterNum=iterNum)
    return F


def filtering_cyclic_boosted(A, regu=0.1, iterNum=300, verbosity=25, error=10e-7):
    """
    The function filtering_cyclic_boosted filters the cyclic signal by applying gradient descent method with boost.
    Parameters:
    A (numpy.ndarray): The gene expression matrix reordered according to cyclic ordering.
    regu (float, optional): The regularization coefficient. Default is 0.1.
    iterNum (int, optional): The number of iterations for the gradient descent. Default is 300.
    verbosity (int, optional): The verbosity level. Default is 25.
    error (float, optional): The stopping criteria for the gradient descent. Default is 10e-7.

    Returns:
    numpy.ndarray: Filtering matrix.
    """
    A = np.array(A, dtype='float64')
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A)
    print("starting filtering")
    A = gradient_descent_full_line_boosted(A, V=V.T, regu=regu, max_evals=iterNum, verbosity=verbosity, error=error)
    return A


def filtering_cyclic_boosted(A, regu=0.1, iterNum=300, verbosity=25, error=10e-7):
    """
    The function filtering_cyclic_boosted filters the cyclic signal by applying gradient descent method with boost.
    Parameters:
    A (numpy.ndarray): The gene expression matrix reordered according to cyclic ordering.
    regu (float, optional): The regularization coefficient. Default is 0.1.
    iterNum (int, optional): The number of iterations for the gradient descent. Default is 300.
    verbosity (int, optional): The verbosity level. Default is 25.
    error (float, optional): The stopping criteria for the gradient descent. Default is 10e-7.

    Returns:
    numpy.ndarray: The filtering matrix.
    """
    A = np.array(A, dtype='float64')
    A = cell_normalization(A)
    V = ge_to_spectral_matrix(A)
    print("starting filtering")
    A = gradient_descent_full_line_boosted(A, V=V.T, regu=regu, max_evals=iterNum, verbosity=verbosity, error=error)
    return A


@jit(nopython=True, parallel=True)
def gradient_ascent_full(A, F, V, regu, epsilon=0.1, iterNum=400):
    """
    The function gradient_ascent_full enhances the signal by applying gradient ascent method.
    Parameters:
    A (numpy.ndarray): The gene expression matrix.
    F (numpy.ndarray): The enhancement matrix.
    V (numpy.ndarray): The spectral matrix.
    regu (float): The regularization coefficient.
    epsilon (float, optional): The step size. Default is 0.1.
    iterNum (int, optional): The number of iterations for the gradient ascent. Default is 400.

    Returns:
    numpy.ndarray: The enhancement matrix.
    """
    j = 0
    epsilon_t = epsilon
    while (j < iterNum):
        value = 0
        if j % 50 == 1:
            print("Iteration number: " + str(j))
        epsilon_t *= 0.995
        tmp_value, grad = function_and_gradient_full(A=A, B=F, V=V, regu=regu)
        F = F + epsilon_t * grad
        F = numba_clip(F, F.shape[0], F.shape[1], 0, 1)
        j += 1
    return F


@jit(nopython=True, parallel=True)
def stochastic_gradient_ascent_full(A, F, V, regu, epsilon=0.1, iterNum=400, regu_norm='L1', verbosity=25):
    """
    The function stochastic_gradient_ascent_full enhances the cyclic signal by applying stochastic gradient ascent method.
    Parameters:
    A (numpy.ndarray): The gene expression matrix.
    F (numpy.ndarray): The enhancement matrix.
    V (numpy.ndarray): The theoretic spectrum of covariance matrix.
    regu (float): The regularization coefficient.
    epsilon (float, optional): The step size. Default is 0.1.
    iterNum (int, optional): The number of iterations for the gradient ascent. Default is 400.
    regu_norm (str, optional): The type of regularization to apply. Default is 'L1'.
    verbosity (int, optional): The verbosity level. Default is 25.

    Returns:
    numpy.ndarray: The enhancement matrix.
    """
    j = 0
    epsilon_t = epsilon
    VVT = V.dot(V.T)
    while (j < iterNum):
        if j % verbosity == 1:
            value, grad = function_and_gradient_full_acc(A=A, B=F, V=V, VVT=VVT, regu=regu, regu_norm=regu_norm)
            print("Iteration number: ")
            print((j))
            print("function value: ")
            print((value))
        epsilon_t *= 0.995
        grad = G_full(A=A, B=F, V=V, regu=regu, regu_norm=regu_norm)
        F = F + epsilon_t * (grad + np.random.normal(0, 0.01, grad.shape))
        F = numba_clip(F, F.shape[0], F.shape[1], 0, 1)
        j += 1
    return F


@jit(nopython=True, parallel=True)
def gradient_descent_full(A, F, V, regu, epsilon=0.1, iterNum=400, regu_norm='L1'):
    """
    The function gradient_descent_full filters the cyclic signal by applying gradient descent method.
    Parameters:
    A (numpy.ndarray): The gene expression matrix.
    F (numpy.ndarray): The filtering matrix.
    V (numpy.ndarray): The theoretic spectrum of covariance matrix.
    regu (float): The regularization coefficient.
    epsilon (float, optional): The step size. Default is 0.1.
    iterNum (int, optional): The number of iterations for the gradient descent. Default is 400.
    regu_norm (str, optional): The type of regularization to apply. Default is 'L1'.

    Returns:
    numpy.ndarray: The filtering matrix.
    """
    j = 0
    epsilon_t = epsilon
    while j < iterNum:
        if j % 100 == 1:
            print("Iteration number: " + str(j))
        epsilon_t *= 0.995
        tmp_value, grad = function_and_gradient_full(A=A, B=F, V=V, regu=regu, regu_norm=regu_norm)
        F = F - epsilon_t * grad
        F = numba_clip(F, F.shape[0], F.shape[1], 0, 1)
        j += 1
    return F


@jit(nopython=True, parallel=True)
def gradient_descent_full_line(A, F, V, regu, gamma=1e-04, max_evals=250, verbosity=float('inf'), error=1e-07, regu_norm='L1'):
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
    VVT = V.dot(V.T)
    w = F
    evals = 0
    loss, grad = function_and_gradient_full_acc(A=A, B=F, V=V, VVT=VVT, regu=regu, regu_norm=regu_norm)
    alpha = 1 / np.linalg.norm(grad)
    prev_w = np.zeros(w.shape)
    while evals < max_evals and np.linalg.norm(w - prev_w) > error:
        prev_w = np.copy(w)
        evals += 1
        if evals % verbosity == 0:
            print((evals))
            print('th Iteration    Loss :: ')
            print((loss))
        gTg = np.linalg.norm(grad)
        gTg = gTg * gTg
        new_w = w - alpha * grad
        new_w = numba_clip(new_w, new_w.shape[0], new_w.shape[1], 0, 1)
        new_loss, new_grad = function_and_gradient_full_acc(A=A, B=new_w, V=V,
                                            VVT=VVT, regu=regu, regu_norm=regu_norm)
        while new_loss > loss - gamma * alpha * gTg:
            alpha = ((alpha ** 2) * gTg) / (2 * (new_loss + alpha * gTg - loss))
            new_w = w - alpha * grad
            new_w = numba_clip(new_w, new_w.shape[0], new_w.shape[1], 0, 1)
            new_loss, new_grad = function_and_gradient_full_acc(A=A, B=new_w, V=V, VVT=VVT,
                                                regu=regu, regu_norm=regu_norm)
        alpha = min(1, 2 * (loss - new_loss) / gTg)
        loss = new_loss
        grad = new_grad
        w = new_w
    return w


@jit(nopython=True, parallel=True)
def function_and_gradient_filtering_boosted(A, V, regu_matrix):
    """
    Perform filtering on input gene expression matrix A, using spectral matrix V.
    Returns the function value and gradient of the filtering-boosting process.

    Parameters
    ----------
    A : numpy array
        Input gene expression matrix of shape (n,p)
    V : numpy array
        Matrix of shape (p,k) used in the filtering process
    regu_matrix : float
        regularization matrix

    Returns
    -------
    functionValue : float
        The value of the function calculated using trace and l1 norm
    gradient : numpy array
        The gradient of the function calculated using dot product
    """
    functionValue = np.trace((((V.T).dot(A)).dot(A.T)).dot(V)) - np.linalg.norm(regu_matrix * A, 1)
    gradient = (2 * ((V).dot(V.T)).dot(A)) - regu_matrix * np.sign(A)
    return functionValue, gradient


@jit(nopython=True, parallel=True)
def gradient_descent_full_line_boosted(A, V, regu, gamma=1e-04, max_evals=250, verbosity=float('inf'), error=1e-07):
    """
    Applies gradient descent with backtracking line search to optimize the function and gradient filtering.

    Parameters
    ----------
    A : numpy array
        Gene expression matrix
    V : numpy array
        Spectral matrix
    regu : float
        Regularization term
    gamma : float, optional
        Backtracking line search parameter, by default 1e-04
    max_evals : int, optional
        Maximum number of iterations, by default 250
    verbosity : float, optional
        Frequency at which the optimization process is printed, by default float('inf')
    error : float, optional
        Tolerance for stopping criterion, by default 1e-07

    Returns
    -------
    numpy array
        Optimized gene expression matrix
    """
    regu_matrix = (regu / (A + 10e-5)) / (A + 10e-5)
    tmp_A = copy.deepcopy(A)
    VVT = V.dot(V.T)
    w = A
    evals = 0
    loss, grad = function_and_gradient_filtering_boosted(A, V, regu_matrix)
    alpha = 1 / np.linalg.norm(grad)
    prev_w = np.zeros(w.shape)
    while evals < max_evals and np.linalg.norm(w - prev_w) > error:
        prev_w = copy.deepcopy(w)
        evals += 1
        if evals % verbosity == 0:
            print(str(evals) + 'th Iteration    Loss :: ' + str(loss) + ' gradient :: ' + str(np.linalg.norm(grad)))
        gTg = np.linalg.norm(grad)
        gTg = gTg * gTg
        new_w = w - alpha * grad
        new_w = new_w.clip(min=0, max=tmp_A)
        new_loss, new_grad = function_and_gradient_filtering_boosted(A=new_w, V=V, regu_matrix=regu_matrix)
        while new_loss > loss - gamma * alpha * gTg:
            alpha = ((alpha ** 2) * gTg) / (2 * (new_loss + alpha * gTg - loss))
            new_w = w - alpha * grad
            new_w = new_w.clip(min=0, max=tmp_A)
            new_loss, new_grad = function_and_gradient_filtering_boosted(A=new_w, V=V, regu_matrix=regu_matrix)
        alpha = min(1, 2 * (loss - new_loss) / gTg)
        loss = new_loss
        grad = new_grad
        w = new_w
    return w


@jit(nopython=True, parallel=True)
def function_and_gradient_full(A, B, V, regu, regu_norm='L2'):
    """
    Computes function value and gradient for the optimization problem.

    Parameters
    ----------
    A : numpy array
        Gene expression matrix.
    B : numpy array
        Filtering matrix.
    V : numpy array
        Spectral matrix.
    regu : float
        Regularization coefficient.
    regu_norm : str, optional
        Type of regularization ('L1' or 'L2'). Default is 'L2'.

    Returns
    -------
    tuple
        A tuple containing the projection over theoretic spectrum and the gradient according to 'B'.
    """
    if regu_norm == 'L1':
        T_0 = (A * B)
        t_1 = np.linalg.norm(B, 1)
        functionValue = (np.trace((((V.T).dot(T_0)).dot(T_0.T)).dot(V)) - (regu * t_1))
        gradient = ((2 * (((V).dot(V.T)).dot(T_0) * A)) - ((regu) * np.sign(B)))
    else:
        T_0 = (A * B)
        t_1 = np.linalg.norm(A * B, 'fro')
        functionValue = (np.trace((((V.T).dot(T_0)).dot(T_0.T)).dot(V)) - (regu * t_1))
        gradient = ((2 * (((V).dot(V.T)).dot(T_0) * A)) - ((regu / t_1) * B))
    return functionValue, gradient


@jit(nopython=True, parallel=True)
def function_and_gradient_full_acc(A, B, V, VVT, regu, regu_norm):
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
    t_1 = np.linalg.norm(B, 1)
    functionValue = (np.trace((((V.T).dot(T_0)).dot(T_0.T)).dot(V)) - (regu * t_1))
    gradient = ((2 * ((VVT).dot(T_0) * A)) - ((regu) * np.sign(B)))
    return functionValue, gradient


@jit(nopython=True, parallel=True)
def G_full(A, B, V, regu, regu_norm='L1'):
    """
    Computes the gradient for the optimization problem

    Parameters
    ----------
    A : numpy array
        Gene expression matrix.
    B : numpy array
        Filtering matrix.
    V : numpy array
        Spectral matrix.
    regu : float
        Correlation between neighbors.
    regu_norm : str, optional
        Type of regularization ('L1' or 'L2'). Default is 'L1'.

    Returns
    -------
    numpy array
        The gradient according to 'B'.
    """
    if regu_norm == 'L1':
        T_0 = (A * B)
        gradient = ((2 * (((V).dot(V.T)).dot(T_0) * A)) - ((regu) * np.sign(B)))
    else:
        T_0 = (A * B)
        gradient = ((2 * (((V).dot(V.T)).dot(T_0) * A)) - (regu) * 2 * B)
    return gradient


def filter_non_cyclic_genes_vector(A, regu=2, iterNum=500):
    """
    Filters non-cyclic genes from a given gene expression matrix.

    Parameters
    ----------
    A : numpy array
        Gene expression matrix.
    regu : float, optional
        Regularization coefficient. Default is 2.
    iterNum : int, optional
        Number of iterations. Default is 500.

    Returns
    -------
    numpy array
        Genes filtering matrix.
    """
    A = np.array(A).astype('float64')
    A = gene_normalization(A)
    n = A.shape[0]
    p = A.shape[1]

    eigenvectors, eigenvalues = generate_eigenvectors_circulant()

    D = gradient_ascent_filter(A, D=np.identity((p)), eigenvectors_list=eigenvectors[1:],
                               eigenvalues_list=eigenvalues[1:], regu=regu, iterNum=iterNum)
    return D


def filter_linear_full(A: np.ndarray, method: str, regu: float = 0.1, iterNum: int = 300, lr: float = 0.1,
                       regu_norm: str = 'L1') -> np.ndarray:
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
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    F = gradient_descent_full(A, F=np.ones(A.shape), V=eigenvectors[:, 1:], regu=regu,
                              iterNum=iterNum, epsilon=lr, regu_norm=regu_norm)
    return F


def enhancement_linear(A: np.ndarray, regu: float = 0.1, iterNum: int = 300, method: str = 'numeric') -> np.ndarray:
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
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=eigenvectors[:, 1:], regu=regu, iterNum=iterNum)
    return F


def filtering_linear(A, method, regu=0.1, iterNum=300, verbosity=25,
                     error=10e-7, optimized_alpha=True, regu_norm='L1'):
    """
    Filtering of linear signal

    Parameters
    ----------
    A : numpy array
        Gene expression matrix (reordered according to linear ordering)
    method : str
        method of generating eigenvectors
    regu : float, optional
        regularization coefficient. Default is 0.1.
    iterNum : int, optional
        Number of iterations. Default is 300.
    verbosity : int, optional
        Level of verbosity. Default is 25.
    error : float, optional
        The error tolerance. Default is 10e-7.
    optimized_alpha : bool, optional
        Whether to use optimized alpha. Default is True.
    regu_norm : str, optional
        The regularization norm (L1/L2). Default is 'L1'.

    Returns
    -------
    numpy array
        The filtering matrix.
    """
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    real_eigenvalues, real_vec = linalg.eig(A.dot(A.T))
    alpha = get_alpha(ngenes=A.shape[1], optimized_alpha=optimized_alpha, eigenvals=real_eigenvalues)
    eigenvectors = get_linear_eig_data(A.shape[0], alpha, method=method,
                                       normalize_vectors=True)
    F = gradient_descent_full_line(A, F=np.ones(A.shape), V=eigenvectors[:, 1:],
                                   regu=regu, max_evals=iterNum, verbosity=verbosity,
                                   error=error, regu_norm=regu_norm)
    return F


def enhance_linear_genes(A, method, regu=2, iterNum=500, lr=0.1):
    """
    Enhance linear genes in the given gene expression matrix.

    Parameters
    ----------
    A : numpy array
        The gene expression matrix.
    method : str
        method of generating eigenvectors
    regu : float, optional
        Regularization coefficient. Default is 2.
    iterNum : int, optional
        Number of iterations. Default is 500.
    lr : float, optional
        Learning rate. Default is 0.1.

    Returns
    -------
    numpy array
        Diagonal filtering matrix.
    """
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
    """
    Filters the linear genes in the given gene expression matrix.
    Parameters
    ----------
    A : numpy array
        The gene expression matrix.
    method : str, optional
        The method of generating eigenvectors. Default is 'numeric'.
    regu : float, optional
        Regularization coefficient. Default is 2.
    iterNum : int, optional
        Number of iterations. Default is 500.
    lr : float, optional
        Learning rate. Default is 0.1.

    Returns
    -------
    numpy array
        Diagonal filtering matrix.
    """
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
def numba_min_clip(A: np.ndarray, a_min: int) -> np.ndarray:
    """
    Implementing np.clip for a minimum value using numba

    Parameters
    ----------
    A : np.ndarray
        Array
    a_min : int
        minimum value to clip

    Returns
    -------
    np.ndarray
        clipped array
    """
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            if A[i, j] < a_min:
                A[i, j] = a_min
    return A


@jit(nopython=True)
def numba_clip(A: np.ndarray, n: int, m: int, a_min: int, a_max: int) -> np.ndarray:
    """
    Implementing np.clip using numba

    Parameters
    ----------
    A : np.ndarray
        Array
    n : int
        number of rows
    m : int
        number of columns
    a_min : int
        minimum value to clip
    a_max : int
        maximum value to clip

    Returns
    -------
    np.ndarray
        clipped array
    """
    for i in range(n):
        for j in range(m):
            if A[i, j] < a_min:
                A[i, j] = a_min
            elif A[i, j] > a_max:
                A[i, j] = a_max
    return A


@jit(nopython=True)
def numba_vec_clip(v: list, n: int, a_min: int, a_max: int):
    """
    Implementing np.clip using numba for vectors

    Parameters
    ----------
    v : list
        vector
    n : int
        number of entries
    a_min : int
        minimum value to clip
    a_max : int
        maximum value to clip

    Returns
    -------
    np.ndarray
        clipped vector
    """
    for i in range(n):
        if v[i] < a_min:
            v[i] = a_min
        elif v[i] > a_max:
            v[i] = a_max
    return v


@jit(nopython=True, parallel=True)
def BBS(E: np.ndarray, iterNum: int = 1000, early_exit: int = 15) -> np.ndarray:
    """
    Bregmanian Bi-Stochastication algorithm as described in
    https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/KAIS_BBS_final.pdf

    Parameters
    ----------
    E : np.ndarray
        permutation matrix
    iterNum : int, optional
        iteration number, by default 1000
    early_exit : int, optional
        early exit number, by default 15

    Returns
    -------
    np.ndarray
        Bi-Stochastic matrix
    """
    n = E.shape[0]
    prev_E = np.empty(E.shape)
    I = np.identity(n)
    a_min=0
    ones_m = np.ones((n, n))
    for i in range(iterNum):
        if i % early_exit == 1:
            prev_E = np.copy(E)
        ones_E = ones_m.dot(E)
        E = E + (1 / n) * (I - E + (1 / n) * (ones_E)).dot(ones_m) - (1 / n) * ones_E
        E = numba_min_clip(E, 0)
        if i % early_exit == 1:
            if np.linalg.norm(E - prev_E) < ((10e-6) * n):
                break
    return E


def reorder_indicator(A, IN, iterNum=300, batch_size=20, gamma=0, lr=0.1):
    """
    Cyclic reorder rows of a gene expression matrix using stochastic gradient ascent. With optional usage of prior knowledge.

    Parameters
    ----------
    A : numpy array
        The gene expression matrix.
    IN : numpy array
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
    tuple of numpy array
        Permutation matrix (which is calculated by greedy rounding of 'E' matrix).
    """
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    n = A.shape[0]
    V = ge_to_spectral_matrix(A)
    E = np.ones((n, n)) / n
    E = E * IN
    E = BBS(E)
    E = sga_matrix_momentum_indicator(A, E, V=V.T, IN=IN, iterNum=iterNum, batch_size=batch_size, gamma=gamma, lr=lr)
    E_recon = reconstruct_e(E)
    return E, E_recon


def enhance_general_topology(A, V, regu=0.5, iterNum=300):
    """
    Enhance general topology (that is given by the user) of a gene expression matrix using stochastic gradient ascent.
    Parameters
    ----------
    A : numpy array
        The gene expression matrix.
    V : numpy array
        The topological eigenvectors.
    regu : float, optional
        Regularization coefficient. Default is 0.5.
    iterNum : int, optional
        Number of iterations. Default is 300.

    Returns
    -------
    numpy array
        The filtering matrix.
    """
    A = np.array(A).astype('float64')
    A = cell_normalization(A)
    F = stochastic_gradient_ascent_full(A, F=np.ones(A.shape), V=V, regu=regu, iterNum=iterNum)
    return F


def gene_inference_general_topology(A, V, regu=0.5, iterNum=100, lr=0.1):
    """
    Infer the genes which are non-smooth over given topology.

    Parameters
    ----------
    A : numpy array
        The gene expression matrix.
    V : numpy array
        The topological eigenvectors multiplied by their appropriate eigenvalues.
    regu : float, optional
        Regularization coefficient. Default is 0.5.
    iterNum : int, optional
        Number of iterations. Default is 100.
    lr : float, optional
        Learning rate. Default is 0.1.

    Returns
    -------
    numpy array
        The filtering matrix.
    """
    A = np.array(A).astype('float64')
    A = gene_normalization(A)
    p = A.shape[1]
    D = gradient_ascent_filter_matrix(A, D=np.identity((p)) / 2,  U=V, ascent=-1, regu=regu, lr=lr, iterNum=iterNum)
    return D


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

def filter_general_covariance(A, cov, regu=0, epsilon=0.1, iterNum=100, regu_norm='L1', device='cpu'):
    """
    Filter out a signal based on a eigendecomposition of the theoretical covariance matrix describes the signal by running gradient ascent with L1 regularization.

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
    V = np.array(get_theoretic_eigen(cov)).astype(float)
    B = normalize(B, axis=1, norm='l2')
    F = gradient_descent_full(B, np.ones(B.shape).astype(float), V=V, regu=regu, epsilon=epsilon, iterNum=iterNum)
    return F

def filter_genes_by_proj(A: np.ndarray, V: np.ndarray, n_genes: int = None, percent_genes: float = None) -> np.ndarray:
    """
    Filters genes from a matrix A based on a projection matrix V.
    If n_genes is not provided, the function will select the top half of the genes by default.
    If percent_genes is not provided, the function will select the top n_genes genes by default.

    Parameters
    ----------
    A : np.ndarray
        A matrix of shape (n,p) where m is the number of samples and n is the number of genes.
    V : np.ndarray
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
        score_array[i]= np.trace((V.T).dot(np.outer(gene,gene)).dot(V))
    x = np.argsort(score_array)[::-1][:n_genes]
    D = np.zeros((A.shape[1],A.shape[1]))
    D[x,x]=1
    return D

def filter_non_cyclic_genes_by_proj(A: np.ndarray,  n_genes: int = None, percent_genes: float = None) -> np.ndarray:
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
    D =  filter_genes_by_proj(A=A, V=V.T, n_genes=n_genes, percent_genes=percent_genes)
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

def filter_general_genes_by_proj(A: np.ndarray, cov: np.ndarray, n_genes: int = None, percent_genes: float = None) -> np.ndarray:
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
    D =  filter_genes_by_proj(A=A, V=V, n_genes=n_genes, percent_genes=percent_genes)
    return D

def enhance_general_covariance(A, cov, regu=0, epsilon=0.1, iterNum=100, regu_norm='L1', device='cpu'):
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
    V = np.array(get_theoretic_eigen(cov)).astype(float)
    B = normalize(B, axis=1, norm='l2')
    F = stochastic_gradient_ascent_full(B, np.ones(B.shape).astype(float), V=V, regu=regu, epsilon=epsilon, iterNum=iterNum)
    return F

def gene_inference_general_covariance(A, cov, regu=0.5, iterNum=100, lr=0.1):
    """
    Infer the genes which are non-smooth over given covariance.
    Parameters
    ----------
    A : numpy array
        The gene expression matrix.
    cov : numpy array
        The theoretical covariance matrix.
    regu : float, optional
        Regularization coefficient. Default is 0.5.
    iterNum : int, optional
        Number of iterations. Default is 100.
    lr : float, optional
        Learning rate. Default is 0.1.
    Returns
    -------
    numpy array
        The filtering matrix.
    """
    A = np.array(A).astype('float64')
    V = np.array(get_theoretic_eigen(cov)).astype(float)
    A = gene_normalization(A)
    p = A.shape[1]
    D = gradient_ascent_filter_matrix(A, D=np.identity((p)) / 2,  U=V, ascent=-1, regu=regu, lr=lr, iterNum=iterNum)
    return D

