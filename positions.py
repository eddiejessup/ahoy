from __future__ import print_function, division
import numpy as np
from ciabatta import fields, vector


class Positions(object):

    def __init__(self, r_0):
        self.r = r_0
        self.r_0 = self.r.copy()
        self.n, self.dim = self.r.shape
        self.origin_flag = np.allclose(self.r_0, 0.0)

    def r_mag(self):
        return vector.vector_mag(self.r)

    def dr(self):
        return self.r - self.r_0

    def dr_mag(self):
        return vector.vector_mag(self.dr())

    def r_w(self):
        return self.r

    def r_w_mag(self):
        return vector.vector_mag(self.r_w())

    def __repr__(self):
        return 'Pos(n={},origin={:d})'.format(self.n, self.origin_flag)


class PeriodicPositions(Positions):

    def __init__(self, L, r_0):
        super(PeriodicPositions, self).__init__(r_0)
        self.L = L
        self.volume = np.product(self.L)

    def get_density_field(self, dx):
        return fields.density(self.r_w(), self.L, dx)

    def get_wraps(self):
        wraps = np.zeros(self.r.shape, dtype=np.int)
        for i_dim in np.where(np.isfinite(self.L))[0]:
            wraps_mag = ((np.abs(self.r[:, i_dim]) + self.L[i_dim] / 2.0) //
                         self.L[i_dim])
            wraps[:, i_dim] = np.sign(self.r[:, i_dim]) * wraps_mag
        return wraps

    def r_w(self):
        wraps = self.get_wraps()
        r_w = self.r.copy()
        for i_dim in np.where(np.isfinite(self.L))[0]:
            r_w[:, i_dim] -= wraps[:, i_dim] * self.L[i_dim]
        return r_w

    def L_repr(self):
        def format_inf(x):
            return x if np.isfinite(x) else 'i'
        return [format_inf(e) for e in self.L]

    def __repr__(self):
        return 'PeriodPos(n={},L={},origin={:d})'.format(self.n, self.L_repr(),
                                                         self.origin_flag)


def positions_factory(L, r_0):
    if L is None or np.all(np.isinf(L)):
        return Positions(r_0)
    else:
        return PeriodicPositions(L, r_0)


def get_uniform_points(n, dim, L, rng=None, obstructors=None):
    if rng is None:
        rng = np.random
    r = np.zeros([n, dim])
    for i_n in range(n):
        while True:
            for i_dim in range(dim):
                if L is not None and np.isfinite(L[i_dim]):
                    r[i_n, i_dim] = rng.uniform(-L[i_dim] / 2.0,
                                                L[i_dim] / 2.0)
                else:
                    r[i_n, i_dim] = 0.0
            if obstructors is None:
                break
            elif not obstructors.get_obstructeds(r[i_n]):
                break
    return r
