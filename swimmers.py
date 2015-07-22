from __future__ import print_function, division


class Swimmers(object):

    def __init__(self, v_0, direction_measurer):
        self.v_0 = v_0
        self.direction_measurer = direction_measurer

    def displace(self, positions, dt):
        ds = self.direction_measurer.get_directions()
        dr = self.v_0 * ds.u() * dt
        return positions.displace(dr)

    def __repr__(self):
        return 'Swims(v={:g})'.format(self.v_0)
