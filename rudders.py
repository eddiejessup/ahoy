from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import numpy as np
from ships import measurers


class Rudders(object):

    def __init__(self, noise_measurer, rng=None):
        self.noise_measurer = noise_measurer
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

    def rotate(self, directions, dt):
        noise = self.noise_measurer.get_noise()
        return self._rotate(directions, noise, dt)

    def _rotate(self, directions, noise, dt):
        return directions

    def __repr__(self):
        return 'Rudders(noise={})'.format(self.noise_measurer)

    def is_chemotactic(self):
        return isinstance(self.noise_measurer, measurers.ChemoNoiseMeasurer)

    def get_chi(self):
        if self.is_chemotactic():
            return self.noise_measurer.chi
        else:
            return None


class TumbleRudders(Rudders):

    def _get_tumblers(self, directions, noise, dt):
        return self.rng.uniform(size=directions.n) < noise * dt

    def _rotate(self, directions, noise, dt):
        tumblers = self._get_tumblers(directions, dt, noise)
        return directions.tumble(tumblers, self.rng)

    def __repr__(self):
        return 'Tumblers(noise={})'.format(self.noise_measurer)


class RotationRudders(Rudders):
    __metaclass__ = ABCMeta

    @abstractmethod
    def _get_dth(self, directions, noise, dt):
        return

    def _rotate(self, directions, noise, dt):
        dth = self._get_dth(directions, noise, dt)
        return directions.rotate(dth)


class RotationRudders2D(RotationRudders):

    def _get_dth(self, directions, noise, dt):
        return self.rng.normal(scale=np.sqrt(2.0 * noise * dt),
                               size=directions.n)

    def __repr__(self):
        return 'Rotors2D(noise={})'.format(self.noise_measurer)


def rotation_rudders_factory(dim, *args, **kwargs):
    if dim == 2:
        return RotationRudders2D(*args, **kwargs)
    else:
        raise NotImplementedError('No rotation rudders implemented in this '
                                  ' dimension')
