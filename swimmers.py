from __future__ import print_function, division
from ahoy.utils.meta import make_repr_str


class Swimmers(object):

    def __init__(self, v_0, directions):
        self.v_0 = v_0
        self.directions = directions

    def displace(self, positions, dt):
        dr = self.v_0 * self.directions.u() * dt
        positions.r += dr
        return positions, dr

    def __repr__(self):
        fs = [('v_0', self.v_0)]
        return make_repr_str(self, fs)
