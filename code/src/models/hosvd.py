
import numpy as np
from numpy.linalg import norm
from src.util.merge_Tucker import merge_Tucker
from src.util.t2m import t2m
from src.util.m2t import m2t


def hosvd(X, rank, max_iter=1, err_tol=1, track_fval=False):
    ''' Higher Order Singular Value Decomposition.
    Takes the HOSVD of tensor input.

    Parameters:
    ---
        X: Data tensor

        rank: Aimed ranks for the resulting tensor.

        max_iter: Number of iterations. If 1, HOSVD will be applied, otherwise
            HOOI will be run.

        err_tol: Error tolerance. Not relevant for HOSVD.

    Returns:
    ---
        Y: Tensor with ranks `rank`at each mode unfolding.
        U_list: List of factor matrices of HOSVD.
    '''
    sizes = X.shape
    sizes_core = list(sizes)
    n = len(sizes)
    U_list = [np.eye(sizes[i]) for i in range(n)]
    Y = X.copy()

    iter = 0
    change = np.inf
    err_tot = []
    val_change = []
    while iter < max_iter and change > err_tol:
        Y_old = Y.copy()
        for i in range(n):
            ind = np.setdiff1d(np.arange(n), i)
            U_other = [U_list[j] for j in ind]
            C = merge_Tucker(X, U_other, ind, transpose=True)
            U, S, _ = np.linalg.svd(t2m(C, i), full_matrices=False)
            ind = np.argsort(S)[::-1]
            ind = ind[:rank[i]]
            U_list[i] = U[:, ind]
            sizes_core[i] = rank[i]

        C = m2t(U_list[-1].transpose()@t2m(C, n-1), sizes_core, n-1)
        Y = merge_Tucker(C, U_list, np.arange(n))

        change = norm(Y-Y_old)**2
        if track_fval:
            val_change.append(change)
            err_tot.append(norm(Y-X)**2)

        iter += 1

    return Y, U_list, err_tot, val_change
