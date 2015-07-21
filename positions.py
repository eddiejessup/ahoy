from __future__ import print_function, division
import numpy as np
from ciabatta import fields, vector


class Positions(object):

    def __init__(self, r_0, L):
        self.r = r_0
        self.r_0 = self.r.copy()
        self.n, self.dim = self.r.shape
        self.L = L
        self.wraps = np.zeros_like(self.r, dtype=np.int)
        self.volume = np.product(self.L)
        self.origin_flag = np.allclose(self.r_0, 0.0)

    def _wrap(self):
        wraps_cur = np.zeros([self.n], dtype=np.int)
        for i_dim in np.where(np.isfinite(self.L))[0]:
            wraps_cur[:] = self.r[:, i_dim] > self.L[i_dim] / 2.0
            wraps_cur[:] -= self.r[:, i_dim] < -self.L[i_dim] / 2.0
            self.wraps[:, i_dim] += wraps_cur
            self.r[:, i_dim] -= wraps_cur * self.L[i_dim]

    def displace(self, dr):
        self.r += dr
        self._wrap()
        return self

    def get_density_field(self, dx):
        return fields.density(self.r, self.L, dx)

    def get_unwrapped_r(self):
        r_unwrap = self.r.copy()
        for i_dim in np.where(np.isfinite(self.L))[0]:
            r_unwrap[:, i_dim] += self.wraps[:, i_dim] * self.L[i_dim]
        return r_unwrap

    def get_unwrapped_r_mag(self):
        return vector.vector_mag(self.get_unwrapped_r())

    def get_unwrapped_dr(self):
        return self.get_unwrapped_r() - self.r_0

    def get_unwrapped_dr_mag(self):
        return vector.vector_mag(self.get_unwrapped_dr())

    def __repr__(self):
        def format_inf(x):
            return '{:g}'.format(x) if np.isfinite(x) else 'i'
        L_repr = [format_inf(e) for e in self.L]
        return 'L={},origin={:d}'.format(L_repr, self.origin_flag)
