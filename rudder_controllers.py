from __future__ import print_function, division
import numpy as np


class RudderControllers(object):

    def __init__(self, rudders, noise_0):
        self.rudders = rudders
        self.noise_0 = noise_0

    def get_noise(self, positions, directions):
        return self.noise_0

    def rotate(self, positions, directions):
        noise = self.get_noise(positions, directions)
        return self.rudders.rotate(directions, noise)


class ChemoRudderControllers(RudderControllers):

    def __init__(self, rudders, noise_0, v_0, chi, estimators):
        RudderControllers.__init__(self, rudders, noise_0)
        self.v_0 = v_0
        self.chi = chi
        self.estimators = estimators

    def get_noise(self, positions, directions):
        cdots = self.estimators.get_cdots(positions, directions)
        fitness = self.chi * cdots / self.v_0
        return self.noise_0 * (1.0 - fitness)


class OneSidedChemoRudderControllers(ChemoRudderControllers):

    def get_noise(self, positions, directions):
        noise_two_sided = super(OneSidedChemoRudderControllers,
                                self).get_noise(positions, directions)
        return np.minimum(self.noise_0, noise_two_sided)


def chemo_rud_conts_factory(onesided_flag, *args, **kwargs):
    if onesided_flag:
        return ChemoRudderControllers(*args, **kwargs)
    else:
        return OneSidedChemoRudderControllers(*args, **kwargs)
