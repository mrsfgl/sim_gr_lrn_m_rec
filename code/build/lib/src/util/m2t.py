import numpy as np 

def invert_permutation(p):
    '''The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1. 
    Returns an array s, where s[i] gives the index of i in p.
    '''
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s
    
def m2t(X, dims, row_dims):
    ''' Matrix to tensor reshaping (folding) function.
    '''
    if ~hasattr(row_dims, '__len__'):
        row_dims = [row_dims]

    dims = np.array(dims)
    all_dims = np.arange(len(dims))
    col_dims = np.setdiff1d(all_dims, row_dims)
    tensorized_shape = np.concatenate([row_dims,col_dims])
    T = np.array(X).reshape(dims[tensorized_shape])
    inv_permuted_shape = invert_permutation(tensorized_shape)
    T = T.transpose(inv_permuted_shape)

    return T