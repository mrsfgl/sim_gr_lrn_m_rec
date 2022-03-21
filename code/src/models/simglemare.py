import numpy as np
from numpy.linalg import norm
from soft_hosvd import soft_hosvd

def simglemare(Y, params):  # sourcery skip: avoid-builtin-shadow
    """_summary_

    Parameters
    ----------
    Y : np.masked_array
        Contaminated tensor with possibly missing entries
    params : 
        Algorithm parameters

    Returns
    -------
    L: np.ndarray
        Estimated low graph frequency tensor
    S: np.ndarray
        Estimated sparse tensor
    obj_val: list
        Objective value change over iterations.
    lam_val: list
        Dual variable magnitude change over iterations.
    """
    sizes = Y.shape
    n = len(sizes)

    S, Lx, Gamma, P, d_phi, phi, lamda, beta, alpha = init(Y, params)

    t_norm = norm(Y)
    iter = 0
    err = np.inf
    obj_val = []
    lam_val = []
    terms = []
    val, term = compute_obj(Y, np.zeros(sizes), Lx, S, Gamma, beta, alpha)
    while iter != params.max_iter and err > params.err_tol:
        # L Update
        L = update_L(Y, S, Lx, Gamma, lamda)
        # val, term = compute_obj(Y,L,Lx,S,Lambda,psi,beta,alpha,nuc_norm)

        # S Update
        S_old = S.copy()
        S = soft_threshold(Y-L-Lambda[0], beta/alpha[0])
        # val, term = compute_obj(Y,L,Lx,S,Lambda,psi,beta,alpha,nuc_norm)

        # Lx Update
        Phi = [np.diag(d_phi[i]) for i in range(n)]
        for i in range(n):
            ind = np.triu_indices(sizes[i])
            Phi[ind] = phi[i]
            ind = np.tril_indices(sizes[i])
            Phi[ind] = phi[i]

        L_inv = [
            np.linalg.pinv(np.eye(sizes[i])+(alpha[i]/lamda[2][i])*Phi[i]) 
            for i in range(n)
            ]
        Lx = [L_inv[i]@(L-Gamma[1][i]) for i in range(n)]
        # val, term = compute_obj(Y,L,Lx,S,Lambda,psi,beta,alpha,nuc_norm)

        # phi Update
        p_inv = [
            np.linalg.inv(-beta[1][i]*np.eye(s)+lamda[2][i]*P[i]@P[i].T) 
            for i, s in enumerate(sizes)
            ]
        l = [np.empty((int(s*(s+1)/2),1)) for s in sizes]
        phi = [
            p_inv[i]@(s+lamda[2][i]*P[i]@(d_phi[i]-Gamma[2][i])) 
            for i, s in enumerate(sizes)
            ]

        # Dual variable Updates
        (Lambda,
         dual_err,
         change_lambda
         ) = update_Gamma(Y, Lambda, n, L, S, Lx, t_norm)

        # Objective and error calculations
        # val, term = compute_obj(Y,L,Lx,S,Lambda,psi,beta,alpha,nuc_norm)
        terms.append(term)
        obj_val.append(val)
        lam_val.append(change_lambda)
        err = max(norm(S-S_old)/(np.finfo(float).eps+norm(S_old)), dual_err)
        iter += 1
        if params.verbose:
            if err <= params.err_tol:
                print('Converged!')
            elif iter == params.max_iter:
                print('Max Iter')

    return L, S, obj_val, lam_val


def init(Y, params):
    ''' Initialize variables.'''
    sizes = Y.shape
    Z = np.empty(sizes)
    n = Y.ndim
    S = np.zeros_like(Y)
    Lx = [Z.copy() for _ in range(n)]
    Gamma = [
        Z.copy(), 
        [Z.copy() for _ in range(n)], 
        [np.empty(s) for s in sizes], 
        np.empty(n)
    ]
    d_phi = [np.empty(s) for s in sizes]
    phi = [np.empty(int(s*(s+1)/2)) for s in sizes]
    lamda = params.lamda
    alpha = params.alpha
    beta = params.beta
    P = init_P(sizes)
    
    return S, Lx, Gamma, P, d_phi, phi, lamda, beta, alpha


def init_P(sizes):

    P = [np.zeros((s, int(s*(s+1)/2))) for s in sizes]
    print(P[0].shape)
    for i, s in enumerate(sizes):
        P[i][0, 1:s] = 1
        for j in range(1, s):
            P[i][j, 1:j+1] = 1
            P[i][j, int(j*s-j*(j-1)/2+1):int((j+1)*(s-j/2))] = 1
    
    return P


def update_L(Y, S, Lx, Gamma, lamda):
    '''Updates variable L.'''
    sizes = Y.shape
    n = len(sizes)
    L = np.zeros(sizes)
    temp1 = lamda[0]*(Y-S-Gamma[0])
    temp2 = lamda[1]*sum(Lx[i] + Gamma[1][i] for i in range(n))
    L[~Y.mask] = (temp1[~Y.mask] + temp2[~Y.mask])/(lamda[0]+n*lamda[1])
    L[Y.mask] = temp2[Y.mask]/(n*lamda[1])
    return L


def soft_threshold(T, sigma):
    ''' Soft thresholding of the tensor T with parameter sigma.
    '''
    X = np.clip(np.abs(T)-sigma, 0, None)

    return X*np.sign(T)

