class Agents(object):

    def __init__(self, positions, directions, rudder_controller_sets,
                 swimmers):
        self.n = positions.n
        self.dim = positions.dim
        self.positions = positions
        self.directions = directions
        self.rudder_controller_sets = rudder_controller_sets
        self.swimmers = swimmers

    def iterate(self):
        for rudder_controllers in self.rudder_controller_sets:
            self.directions = rudder_controllers.rotate(self.positions,
                                                        self.directions)
        self.positions = self.swimmers.displace(self.directions,
                                                self.positions)
