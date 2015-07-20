import numpy as np
from ciabatta import vector
cimport numpy as np
cimport cython
from libc.math cimport sin, cos


@cython.cdivision(True)
@cython.boundscheck(False)
def rotate_2d(np.ndarray[np.float_t, ndim=2] v,
              np.ndarray[np.float_t, ndim=1] th):
    cdef:
        unsigned int i
        np.ndarray[np.float_t, ndim=2] v_rot = np.empty((v.shape[0], v.shape[1]))
        double cos_th, sin_th

    for i in range(v.shape[0]):
        cos_th = cos(th[i])
        sin_th = sin(th[i])
        v_rot[i, 0] = cos_th * v[i, 0] - sin_th * v[i, 1]
        v_rot[i, 1] = sin_th * v[i, 0] + cos_th * v[i, 1]
    return v_rot
