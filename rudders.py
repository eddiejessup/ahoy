from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import numpy as np


class Rudders(object):

    def __init__(self, dt, rng=None):
        self.dt = dt
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

    def rotate(self, directions, noise):
        return directions


class TumbleRudders(Rudders):

    def _get_tumblers(self, directions, noise):
        return self.rng.uniform(size=directions.n) < noise * self.dt

    def rotate(self, directions, noise):
        tumblers = self._get_tumblers(directions, noise)
        return directions.tumble(tumblers, self.rng)


class RotationRudders(Rudders):
    __metaclass__ = ABCMeta

    @abstractmethod
    def _get_dth(self, directions, noise):
        return

    def rotate(self, directions, noise):
        dth = self._get_dth(directions, noise)
        return directions.rotate(dth)


class RotationRudders2D(RotationRudders):

    def _get_dth(self, directions, noise):
        return self.rng.normal(scale=np.sqrt(2.0 * noise * self.dt),
                               size=directions.n)


def rotation_rudders_factory(dt, dim, rng=None):
    if dim == 2:
        return RotationRudders2D(dt, rng)
    else:
        raise NotImplementedError('No rotation rudders implemented in this '
                                  ' dimension')
