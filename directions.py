from __future__ import print_function, division
import numpy as np


class Directions1D(object):

    def __init__(self, u_0):
        self.sign = np.sign(u_0[:, 0])
        self.sign_0 = self.sign.copy()
        self.n, self.dim = u_0.shape
        self.aligned_flag = np.all(self.sign_0 == 1)

    def tumble(self, tumblers, rng=None):
        if rng is None:
            rng = np.random
        self.sign[tumblers] = rng.randint(2, size=tumblers.sum()) * 2 - 1
        return self

    def u(self):
        return self.sign[:, np.newaxis].copy()

    def u_0(self):
        return self.sign_0[:, np.newaxis].copy()

    def __repr__(self):
        return 'aligned={:d}'.format(self.aligned_flag)


class Directions2D(Directions1D):

    def __init__(self, u_0):
        self.th = np.arctan2(u_0[:, 1], u_0[:, 0])
        self.th_0 = self.th.copy()
        self.n, self.dim = u_0.shape
        self.aligned_flag = np.allclose(self.th_0, 0.0)

    def tumble(self, tumblers, rng=None):
        if rng is None:
            rng = np.random
        self.th[tumblers] = rng.uniform(-np.pi, np.pi, size=tumblers.sum())
        return self

    def rotate(self, dth):
        self.th += dth
        return self

    def _th_to_u(self, th):
        return np.array([np.cos(self.th), np.sin(self.th)]).T

    def u(self):
        return self._th_to_u(self.th)

    def u_0(self):
        return self._th_to_u(self.th_0)


def directions_factory(u_0, dim):
    if dim == 1:
        return Directions1D(u_0)
    elif dim == 2:
        return Directions2D(u_0)
