from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import numpy as np
from ahoy import noise_measurers
from ahoy.utils.meta import make_repr_str
from ahoy.noise_measurers import noise_measurer_factory


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

    def is_chemotactic(self):
        return isinstance(self.noise_measurer,
                          noise_measurers.ChemoNoiseMeasurer)

    def get_chi(self):
        if self.is_chemotactic():
            return self.noise_measurer.chi
        else:
            return None

    def __repr__(self):
        fs = [('noise_measurer', self.noise_measurer), ('rng', self.rng)]
        return make_repr_str(self, fs)


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


class TumbleRudders(Rudders):

    def _get_tumblers(self, directions, noise, dt):
        return self.rng.uniform(size=directions.n) < noise * dt

    def _rotate(self, directions, noise, dt):
        tumblers = self._get_tumblers(directions, dt, noise)
        return directions.tumble(tumblers, self.rng)


def rotation_rudders_factory(dim, *args, **kwargs):
    if dim == 2:
        return RotationRudders2D(*args, **kwargs)
    else:
        raise NotImplementedError('No rotation rudders implemented in this '
                                  ' dimension')


def rudder_set_factory(onesided_flag, chi, dc_dx_measurer, rng,
                       tumble_chemo_flag, p_0,
                       rotation_chemo_flag, Dr_0, dim):
    rudder_sets = []
    if p_0:
        tumble_noise_measurer = noise_measurer_factory(tumble_chemo_flag,
                                                       onesided_flag, p_0, chi,
                                                       dc_dx_measurer)
        rudder_sets.append(TumbleRudders(tumble_noise_measurer, rng))

    if Dr_0 and dim > 1:
        rotation_noise_measurer = noise_measurer_factory(rotation_chemo_flag,
                                                         onesided_flag, Dr_0,
                                                         chi, dc_dx_measurer)
        rudder_sets.append(rotation_rudders_factory(dim,
                                                    rotation_noise_measurer,
                                                    rng))
    return rudder_sets
