from __future__ import print_function, division
import numpy as np
from ciabatta.meta import make_repr_str


class Swimmers(object):

    def __init__(self, v_0, directions):
        self.v_0 = v_0
        self.directions = directions

    def get_dr(self, dt):
        return self.v_0 * self.directions.u * dt

    # Smell: this does not depend on the object state.
    def displace(self, positions, dr):
        positions.r += dr

    def __repr__(self):
        fs = [('v_0', self.v_0)]
        return make_repr_str(self, fs)


class NoneSwimmers(object):

    def get_dr(self, dt):
        return np.zeros_like(self.directions.u)

    def displace(self, positions, dt):
        pass


def swimmers_factory(spatial_flag, v_0, ds):
    if spatial_flag:
        return Swimmers(v_0, ds)
    else:
        return NoneSwimmers()
