import numpy as np
from util.merge_Tucker import merge_Tucker
from util.t2m import t2m
from util.m2t import m2t


def generate_low_rank_data(dim, ranks):
    '''Generates low-rank tensor data with dimensions `dim` and ranks `ranks`.

    Parameters:
    ---
        dim: list
            Dimensions of the tensor
        ranks: list
            Ranks of the tensor

    Returns:
    ---
        T: np.ndarray
            Tensor of order `len(dim)`.

    '''
    n = len(dim)
    C = np.random.standard_normal(ranks)
    U = [np.linalg.svd(
        np.random.standard_normal((dim[i], ranks[i])),
        full_matrices=False
        )[0] for i in range(n)]

    return merge_Tucker(C, U, np.arange(n))


def generate_smooth_stationary_data(Phi):
    '''Generates smooth and stationary tensor data with dimensions `dim`
    and ranks `ranks`.The graph is structured as a Cartesian product graph.

    Parameters:
    ---
        Phi: list
            Graph Laplacians for all modes.

    Returns:
    ---
        X: np.ndarray
            Tensor smooth and stationary over Cartesian product graph Phi.
    '''

    n = len(Phi)
    sizes = [Phi[i].shape[0] for i in range(n)]
    V = []
    for i in range(n):
        w, temp_v = np.linalg.eig(Phi[i])
        ind = np.argsort(abs(w))
        V.append(temp_v[:, ind])
        W_all = (np.kron(np.ones(w.size), W_all) +
                     np.kron(w[ind], np.ones(W_all.size).flatten('F'))
                     ) if i else w[ind]
    # Create a random Gaussian core tensor and multiply by the pseudoinverse of
    # the eigenvalues.
    C = np.random.randn(np.prod(sizes))
    C[W_all < 1e-8] = 0
    C[W_all > 1e-8] = C[W_all > 1e-8] / W_all[W_all > 1e-8]
    C = C.reshape(sizes)

    X = C.copy()
    for i in range(n):
        X = m2t(V[i]*t2m(X, i), sizes, i)

    return X, V
