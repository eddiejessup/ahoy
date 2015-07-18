import numpy as np
from ciabatta import fields


class Positions(object):

    def __init__(self, r_0, L):
        self.r = r_0
        self.n, self.dim = self.r.shape
        self.L = L
        self.wraps = np.zeros_like(self.r, dtype=np.int)

    def _wrap(self):
        wraps_cur = np.zeros_like(self.wraps, dtype=np.int)
        for i_dim in range(self.dim):
            wraps_cur[:, i_dim] += self.r[:, i_dim] > self.L[i_dim] / 2.0
            wraps_cur[:, i_dim] -= self.r[:, i_dim] < -self.L[i_dim] / 2.0
        self.wraps += wraps_cur
        self.r -= wraps_cur * self.L

    def displace(self, dr):
        self.r += dr
        self._wrap()
        return self

    def get_density_field(self, dx):
        return fields.density(self.r, self.L, dx)


class UniformPositions(Positions):
    def __init__(self, n, dim, L):
        r_0 = np.empty([n, dim])
        for i_dim in range(dim):
            r_0[:, i_dim] = np.random.uniform(-L[i_dim] / 2.0, L[i_dim] / 2.0,
                                              size=n)
        Positions.__init__(self, r_0, L)


class OriginPositions(Positions):
    def __init__(self, n, dim, L):
        r_0 = np.zeros((n, dim))
        Positions.__init__(self, r_0, L)
