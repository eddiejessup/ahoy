from __future__ import print_function, division


class Agents(object):

    def __init__(self, directions, rudder_sets):
        self.n = directions.n
        self.directions = directions
        self.rudder_sets = rudder_sets

    def iterate(self, dt):
        for rudders in self.rudder_sets:
            self.directions = rudders.rotate(self.directions, dt)

    def get_chi(self):
        for rudders in self.rudder_sets:
            if rudders.is_chemotactic():
                return rudders.get_chi()

    def __repr__(self):
        return 'Agents(n={},ds={},ruds={})'.format(self.n, self.directions,
                                                   self.rudder_sets)


class SpatialAgents(Agents):

    def __init__(self, directions, positions, rudder_sets, swimmers):
        super(SpatialAgents, self).__init__(directions, rudder_sets)
        self.positions = positions
        self.swimmers = swimmers

    def iterate(self, dt, obstructer):
        super(SpatialAgents, self).iterate(dt)
        self.positions, dr = self.swimmers.displace(self.positions, dt)
        if obstructer is not None:
            obstructer.obstruct(self.positions, dr, self.directions)

    def __repr__(self):
        repr_str = 'SAgents(n={},ds={},ps={},ruds={},sws={})'
        return repr_str.format(self.n, self.directions, self.positions,
                               self.rudder_sets, self.swimmers)
