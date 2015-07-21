from __future__ import print_function, division
import numpy as np


class Swimmers(object):

    def __init__(self, dt, v_0):
        self.dt = dt
        self.v_0 = v_0

    def displace(self, directions, positions):
        dr = self.v_0 * directions.u() * self.dt
        return positions.displace(dr)

    def __repr__(self):
        return '{:g}'.format(self.v_0)
