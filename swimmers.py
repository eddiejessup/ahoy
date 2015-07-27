from __future__ import print_function, division


class Swimmers(object):

    def __init__(self, v_0, directions):
        self.v_0 = v_0
        self.directions = directions

    def displace(self, positions, dt):
        dr = self.v_0 * self.directions.u() * dt
        positions.r += dr
        return positions, dr

    def __repr__(self):
        dct = {'v_0': self.v_0}
        return '{}({})' % (self.__class__, dct)
