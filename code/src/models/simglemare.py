import enum
from math import gamma
import numpy as np
from numpy.linalg import norm
from src.util.t2m import t2m
from src.util.m2t import m2t

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

    S, X, Gamma, P, Phi, d_phi, phi, lamda, beta, alpha = init(Y, params)

    t_norm = norm(Y)
    iter = 0
    err = np.inf
    obj_val = []
    lam_val = []
    terms = []
    val, term = compute_obj(Y, np.zeros(sizes), X, S, P, Phi, d_phi, phi, Gamma, params)
    while iter != params.max_iter and err > params.err_tol:
        # L Update
        L = update_L(Y, S, X, Gamma, lamda)
        val, term = compute_obj(Y, L, X, S, P, Phi, d_phi, phi, Gamma, params)

        # S Update
        S_old = S.copy()
        S = soft_threshold(Y-L-Gamma[0], 2/lamda[0])
        val, term = compute_obj(Y, L, X, S, P, Phi, d_phi, phi, Gamma, params)

        # X Update
        Phi = [np.diag(d_phi[i]) for i in range(n)]
        for i in range(n):
            ind = np.triu_indices(sizes[i],1)
            Phi[i][ind] = phi[i]
            ind = np.tril_indices(sizes[i],-1)
            Phi[i][ind] = phi[i]

        L_inv = [
            np.linalg.pinv(np.eye(sizes[i])+(alpha[i]/lamda[2][i])*Phi[i]) 
            for i in range(n)
            ]
        X = [L_inv[i]@(t2m(L,i)-Gamma[1][i]) for i in range(n)]
        val, term = compute_obj(Y, L, X, S, P, Phi, d_phi, phi, Gamma, params)

        # phi Update
        p_inv = [
            -np.linalg.inv(beta[1][i]*np.eye(int(s*(s-1)/2))+lamda[2][i]*P[i].T@P[i])
            for i, s in enumerate(sizes)
            ]
        x = [(X[i]@X[i].T)[np.triu_indices(s,1)] for i,s in enumerate(sizes)]
        phi = [
            p_inv[i]@(2*alpha[i]*x[i]+lamda[2][i]*P[i].T@(d_phi[i]-Gamma[2][i]))
            for i in range(n)
            ]
        phi = [np.clip(p, None, 0) for p in phi]
        val, term = compute_obj(Y, L, X, S, P, Phi, d_phi, phi, Gamma, params)

        # d_phi Update
        dp_inv = [
            np.linalg.inv((beta[0][i]+lamda[2][i])*np.eye(s)+lamda[3][i])
            for i,s in enumerate(sizes)
        ]
        dx = [norm(X[i]**2, axis=1).T for i in range(n)]
        d_phi = [
            dp_inv[i]@(lamda[3][i]*(s+Gamma[3][i])-alpha[i]*dx[i]+lamda[2][i]*(Gamma[2][i]-P[i]@phi[i]))
            for i,s in enumerate(sizes)
        ]
        val, term = compute_obj(Y, L, X, S, P, Phi, d_phi, phi, Gamma, params)

        # Dual variable Updates
        Gamma, dual_err = update_Gamma(Y, Gamma, L, S, X, d_phi, phi, P)

        # Objective and error calculations
        val, term = compute_obj(Y, L, X, S, P, Phi, d_phi, phi, Gamma, params)

        terms.append(term)
        obj_val.append(val)
        lam_val.append(np.sqrt(dual_err))
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
    n = Y.ndim
    Z = np.zeros(sizes)
    S = np.zeros(sizes)
    X = [t2m(Z.copy(), i) for i in range(n)]
    d_phi = [np.zeros(s) for s in sizes]
    phi = [np.zeros(int(s*(s-1)/2)) for s in sizes]
    Gamma = [
        Z.copy(), 
        X.copy(), 
        d_phi.copy(), 
        np.zeros(n)
    ]
    P = init_P(sizes)
    lamda = params.lamda
    alpha = params.alpha
    beta = params.beta
    
    Phi = [np.diag(d_phi[i]) for i in range(n)]
    for i in range(n):
        ind = np.triu_indices(sizes[i],1)
        Phi[i][ind] = phi[i]
        ind = np.tril_indices(sizes[i],-1)
        Phi[i][ind] = phi[i]

    return S, X, Gamma, P, Phi, d_phi, phi, lamda, beta, alpha


def init_P(sizes):
    '''Initialize P matrices given size.'''
    P = [np.zeros((s, int(s*(s-1)/2))) for s in sizes]
    for i, s in enumerate(sizes):
        P[i][0, :s-1] = 1
        for j in range(1, s):
            P[i][j, :j] = 1
            P[i][j, int(j*(s-1)-j*(j-1)/2+1):int((j+1)*(s-1-j/2))] = 1
    
    return P


def update_L(Y, S, X, Gamma, lamda):
    '''Updates variable L.'''
    sizes = Y.shape
    n = len(sizes)
    L = np.zeros(sizes)
    temp1 = lamda[0]*(Y-S-Gamma[0])
    temp2 = sum(lamda[1][i]*m2t(X[i] + Gamma[1][i], sizes, i) for i in range(n))

    L[~Y.mask] = (temp1[~Y.mask] + temp2[~Y.mask])/(lamda[0]+sum(lamda[1]))
    L[Y.mask] = temp2[Y.mask]/sum(lamda[1])
    return L


def soft_threshold(T, sigma):
    ''' Soft thresholding of the tensor T with parameter sigma.'''
    X = np.zeros(T.shape)
    X[~T.mask] = np.clip(np.abs(T[~T.mask])-sigma, 0, None)*np.sign(T[~T.mask])

    return X

def update_Gamma(Y, Gamma, L, S, X, d_phi, phi, P):
    '''Dual variable updates and optimality stats.'''
    sizes = L.shape
    n = len(sizes)

    gamma_update = Y-S-L
    dual_err = norm(gamma_update)**2
    Gamma[0] = Gamma[0] - gamma_update
    for i in range(n):
        gamma_update = X[i]-t2m(L,i)
        dual_err += norm(gamma_update)**2
        Gamma[1][i] = Gamma[1][i] - gamma_update

        gamma_update = P[i]@phi[i]+d_phi[i]
        dual_err += norm(gamma_update)**2
        Gamma[2][i] = Gamma[2][i] - gamma_update

        gamma_update = d_phi[i].sum()-sizes[i]
        dual_err += norm(gamma_update)**2
        Gamma[3][i] = Gamma[3][i] - gamma_update

    return Gamma, dual_err


def compute_obj(Y,L,X,S,P,Phi,d_phi,phi,Gamma,params):

    beta = params.beta
    lamda = params.lamda
    alpha = params.alpha
    sizes = Y.shape
    n = Y.ndim
    term = [
        norm(S.ravel(), ord=1),
        lamda[0]*norm(Y-L-S-Gamma[0])**2,
        sum(alpha[i]*np.trace(X[i].T@Phi[i]@X[i]) for i in range(n)),
        sum(beta[0][i]*norm(d_phi[i])**2 for i in range(n)),
        sum(beta[1][i]*norm(phi[i])**2 for i in range(n)),
        sum(lamda[1][i]*norm(t2m(L,i)-X[i]-Gamma[1][i])**2 for i in range(n)),
        sum(lamda[2][i]*norm(P[i]@phi[i]+d_phi[i]-Gamma[2][i])**2 for i in range(n)),
        sum(lamda[3][i]*(sum(d_phi[i])-s-Gamma[3][i])**2 for i, s in enumerate(sizes))
    ]

    return sum(term), term
