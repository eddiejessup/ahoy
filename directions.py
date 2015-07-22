from __future__ import print_function, division
import numpy as np
from ciabatta import vector


class Directions1D(object):

    def __init__(self, u_0=None, aligned_flag=None, n=None, rng=None):
        if aligned_flag is not None:
            if aligned_flag:
                u_0 = get_aligned_directions(n, dim=1)
            else:
                u_0 = get_uniform_directions(n, dim=1, rng=rng)
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
        return 'Ds1D(n={},align={:d})'.format(self.n, self.aligned_flag)


class Directions2D(Directions1D):

    def __init__(self, u_0=None, aligned_flag=None, n=None, rng=None):
        if aligned_flag is not None:
            if aligned_flag:
                u_0 = get_aligned_directions(n, dim=2)
            else:
                u_0 = get_uniform_directions(n, dim=2, rng=rng)
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
        return np.array([np.cos(th), np.sin(th)]).T

    def u(self):
        return self._th_to_u(self.th)

    def u_0(self):
        return self._th_to_u(self.th_0)

    def __repr__(self):
        return 'Ds2D(n={},align={:d})'.format(self.n, self.aligned_flag)


def directions_factory(dim, *args, **kwargs):
    if dim == 1:
        return Directions1D(*args, **kwargs)
    elif dim == 2:
        return Directions2D(*args, **kwargs)


def get_uniform_directions(n, dim, rng=None):
    return vector.sphere_pick(n=n, d=dim, rng=rng)


def get_aligned_directions(n, dim):
    u = np.zeros([n, dim])
    u[:, 0] = 1.0
    return u
