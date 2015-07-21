from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import numpy as np
from ships.ring_buffer import CylinderBuffer


def get_K(t, dt, t_rot_0):
    A = 0.5
    ts = np.arange(0.0, t, dt)
    gs = ts / t_rot_0
    K = np.exp(-gs) * (1.0 - A * (gs + (gs ** 2) / 2.0))
    K[K < 0.0] *= np.abs(K[K >= 0.0].sum() / K[K < 0.0].sum())
    K /= np.sum(K * -ts * dt)
    return K


class CDotEstimators(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_cdots(self, positions, directions):
        return


class SpatialCDotEstimators(CDotEstimators):
    __metaclass__ = ABCMeta

    def __init__(self, v_0):
        self.v_0 = v_0

    @abstractmethod
    def _get_grad_cs(self, positions):
        return

    def get_cdots(self, positions, directions):
        return self.v_0 * np.sum(directions.u() * self._get_grad_cs(positions),
                                 axis=-1)


class LinearSpatialCDotEstimators(SpatialCDotEstimators):

    def _get_grad_cs(self, positions):
        grad_cs = np.zeros_like(positions.r)
        grad_cs[:, 0] = 1.0
        return grad_cs

    def __repr__(self):
        return 'SpatialLinear'


class TemporalCDotEstimators(CDotEstimators):

    def __init__(self, dt, n, t_mem, t_rot_0):
        self.t_mem = t_mem
        self.K_dt = get_K(self.t_mem, dt, t_rot_0) * dt
        self.c_mem = CylinderBuffer(n, self.K_dt.shape[0])

    @abstractmethod
    def _get_cs(self, positions):
        return

    def get_cdots(self, positions, directions):
        self.c_mem.update(self._get_cs(positions))
        return self.c_mem.integral_transform(self.K_dt)


class LinearTemporalCDotEstimators(TemporalCDotEstimators):

    def _get_cs(self, positions):
        return positions.get_unwrapped_dr()[:, 0]

    def __repr__(self):
        return 'TemporalLinear'
