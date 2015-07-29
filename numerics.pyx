import numpy as np
cimport numpy as np
cimport cython


@cython.cdivision(True)
@cython.boundscheck(False)
def integral_transform(np.ndarray[np.float_t, ndim=2] a,
                       np.ndarray[np.float_t, ndim=1] K,
                       unsigned int i_zero,
                       np.ndarray[np.float_t, ndim=1] b,
                       np.ndarray[np.uint_t, ndim=1] inds_p,
                       np.ndarray[np.uint_t, ndim=1] inds_a):
    cdef:
        unsigned int n_l = a.shape[0]
        unsigned int n_p = a.shape[1]
        unsigned int i_l, i_p
        double tot
    np.mod(i_zero + inds_p, n_p, inds_a)

    for i_l in range(n_l):
        tot = 0.0
        for i_p in range(n_p):
            tot += a[i_l, inds_a[i_p]] * K[i_p]
        b[i_l] = tot
