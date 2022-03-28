import numpy as np


def invert_permutation(p):
    '''The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    '''
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def t2m(X, row_dims=[0]):
    ''' Tensor to matrix reshaping (unfolding) function.
    '''

    if ~hasattr(row_dims, '__len__'):
        row_dims = [row_dims]

    shape = np.array(X.shape)
    if max(row_dims) > shape.size:
        shape.append(np.ones(max(row_dims)-shape.size))

    all_dims = np.arange(shape.size)
    column_dims = np.setdiff1d(all_dims, row_dims)
    tensor_reordered_shape = tuple(np.concatenate([row_dims, column_dims]))
    T = X.transpose(tensor_reordered_shape)
    matrix_shape = (np.prod(shape[row_dims]), np.prod(shape[column_dims]))
    T = T.reshape(matrix_shape)
    return T
