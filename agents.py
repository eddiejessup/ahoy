from __future__ import print_function, division


class Agents(object):

    def __init__(self, positions, directions, rudder_controller_sets,
                 swimmers):
        self.n = positions.n
        self.positions = positions
        self.directions = directions
        self.rudder_controller_sets = rudder_controller_sets
        self.swimmers = swimmers
        self.density = self.n / self.positions.volume

    def iterate(self):
        for rudder_controllers in self.rudder_controller_sets:
            self.directions = rudder_controllers.rotate(self.positions,
                                                        self.directions)
        self.positions = self.swimmers.displace(self.directions,
                                                self.positions)

    def __repr__(self):
        repr_str = 'Agents_n={:d},{},{},{},rcs={}'
        return repr_str.format(self.n, self.swimmers, self.positions,
                               self.directions,
                               self.rudder_controller_sets)
