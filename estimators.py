from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import numpy as np


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


class TemporalCDotEstimators(CDotEstimators):
    def __init__(self, t_mem, t_rot_0):
        raise NotImplemented

    @abstractmethod
    def _get_cs(self, positions):
        return

    def get_cdots(self, positions, directions):
        self.c_mem[:, 1:] = self.c_mem.copy()[:, :-1]
        self.c_mem[:, 0] = self._get_cs(positions)
        return np.sum(self.c_mem * self.K * self.dt, axis=1)
