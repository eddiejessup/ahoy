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

# def sep_min(r_a, r_o, L):
#     sep_mins = np.empty(r_a.shape)
#     sep = np.empty(L.shape)
#     for i_a in range(r_a.shape[0]):
#         for i_o in range(r_o.shape[0]):
#             for i_dim in range(r_o.shape[1]):
#                 sep[i_dim] = r_o[i_o, i_dim] - r_a[i_a, i_dim]
#                 if sep[i_dim] > L_half[i_dim]:
#                     sep[i_dim] -= L
#                 elif sep[i_dim] < -L_half[i_dim]:
#                     sep[i_dim] += L
                
#     seps = csep_periodic(ra, rb, L)
#     seps_sq = np.sum(np.square(seps), axis=-1)

#     i_close = np.argmin(seps_sq, axis=-1)

#     i_all = list(range(len(seps)))
#     sep = seps[i_all, i_close]
#     sep_sq = seps_sq[i_all, i_close]
#     return sep, sep_sq
