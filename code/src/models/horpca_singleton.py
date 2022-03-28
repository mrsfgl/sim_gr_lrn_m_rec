import numpy as np
from numpy.linalg import norm
from src.util.soft_hosvd import soft_hosvd


def horpca(Y, psi=1, beta=None, alpha=None, max_iter=100, err_tol=10**-5,
           verbose=False):
    ''' Higher Order Robust PCA
    Runs the ADMM algorithm for HoRPCA

    Parameters:
        Y: numpy.ndarray
            Corrupted input tensor.

        psi: list of doubles
            Weight parameter for mode-n nuclear norm

        beta: list of doubles
            Weight parameter for sparsity

        alpha: list of doubles
            Lagrange multipliers

    Outputs:
        L: numpy.ndarray
            Output tensor.

        obj_val: list of doubles
            Objective values at each iteration.

        terms: list of list of doubles
            Objective values for each term at each iteration

        lam_val: list of doubles
            Total magnitude of dual variables at each iteration

    The original paper for this algorithm is by:
    Goldfarb, Donald, and Zhiwei Qin. "Robust low-rank tensor recovery: Models
        and algorithms." SIAM Journal on Matrix Analysis and Applications
        35.1 (2014): 225-253.
    '''
    sizes = Y.shape
    n = len(sizes)
    S, Lx, Lambda, psi, beta, alpha = init(Y, psi, beta, alpha)

    t_norm = norm(Y)
    iter = 0
    err = np.inf
    obj_val = []
    lam_val = []
    terms = []
    nuc_norm = [0 for _ in range(n)]
    val, term = compute_obj(Y, Lx, S, Lambda, beta, alpha, nuc_norm)
    while iter != max_iter and err > err_tol:
        # Lx Update
        Lx, nuc_norm = soft_hosvd(Y-S, Lambda[1], psi, 1/alpha[0])
        val, term = compute_obj(Y,Lx,S,Lambda,beta,alpha,nuc_norm)

        # S Update
        S_old = S.copy()
        temp = np.zeros_like(Lambda[0])
        for i in range(n):
            temp = temp + (1/n)*(Y-Lx[i]-Lambda[0])

        S = soft_threshold(temp, beta/(alpha[0]*n))
        val, term = compute_obj(Y,Lx,S,Lambda,beta,alpha,nuc_norm)

        # Dual variable Updates
        (Lambda,
         dual_err,
         change_lambda
         ) = update_Lambda(Y, Lambda, n, S, Lx, t_norm)

        # Objective and error calculations
        val, term = compute_obj(Y,Lx,S,Lambda,beta,alpha,nuc_norm)
        terms.append(term)
        obj_val.append(val)
        lam_val.append(change_lambda)
        err = max(norm(S-S_old)/(np.finfo(float).eps+norm(S_old)), dual_err)
        iter += 1
        if verbose:
            if err <= err_tol:
                print('Converged!')
            elif iter == max_iter:
                print('Max Iter')

    return np.array(Lx).mean(axis=0), obj_val, np.array(terms), lam_val


def compute_obj(Y, Lx, S, Lambda, beta, alpha, nuc_norm):
    ''' Computes the objective function for HoRPCA. '''
    n = len(Lambda)

    term = np.zeros(3)
    for i in range(n):
        term[0] += nuc_norm[i]
        term[2] += alpha[0]/2*norm(Y-S-Lx[i]-Lambda[0][i])**2

    term[3] = beta*norm(S.ravel(), ord=1)
    val = sum(term)
    return val, term


def soft_threshold(T, sigma):
    ''' Soft thresholding of the tensor T with parameter sigma.
    '''
    X = np.clip(np.abs(T)-sigma, 0, None)

    return X*np.sign(T)


def init(Y, psi, beta, alpha):
    ''' Initialize variables.'''
    sizes = Y.shape
    n = len(sizes)

    # Initialize parameters using recommended choices in the paper.
    psi = psi if hasattr(psi, '__len__') else [psi for _ in range(n)]
    beta = beta or 1/np.sqrt(max(sizes))
    std_Y = np.std(Y.ravel().data)
    alpha = alpha or [2/std_Y, 1/(5*std_Y)]

    # Initialize tensor variables.
    Lx = [np.zeros(sizes) for _ in range(n)]
    S = np.zeros(sizes)
    Lambda = [np.zeros(sizes) for _ in range(n)]
    return S, Lx, Lambda, psi, beta, alpha


def update_L(Y, S, Lx, Lambda, alpha):
    '''Updates variable L.'''
    sizes = Y.shape
    n = len(sizes)
    return (1/n)*(Y-S-Lambda[0][i] for i in range(n))


def update_Lambda(Y, Lambda, n, L, S, Lx, t_norm):
    lambda_update = L+S-Y
    Lambda[0] = Lambda[0]+lambda_update
    change_lambda = norm(lambda_update)

    dual_err = 0
    for i in range(n):
        lambda_update = L-Lx[i]
        dual_err += norm(lambda_update)**2
        Lambda[1][i] = Lambda[1][i] - lambda_update
        change_lambda += norm(lambda_update)

    dual_err = np.sqrt(dual_err/n)/t_norm
    return Lambda, dual_err, change_lambda
