import numpy as np

def fastranc(refs, d, N, mu):
    refs = np.reshape(refs, (-1, 1))
    d = np.reshape(d, (-1, 1))
    mANC = len(d)
    
    if len(d) != len(refs):
        raise ValueError('Reference and Input data must be of the same length')
    
    W = np.zeros((N+1, 1))
    r = np.flipud(np.vstack(([0], refs[:N])))
    out = np.zeros((mANC, 1))
    y = np.zeros((mANC, 1))
    
    for E in range(N, mANC):
        r = np.vstack(([refs[E]], r[:-1]))
        y[E] = np.sum(W * r)
        out[E] = d[E] - y[E]
        W = W + 2 * mu * out[E] * r
    
    return out.flatten(), y.flatten()