class Swimmers(object):

    def __init__(self, dt, v_0):
        self.dt = dt
        self.v_0 = v_0

    def displace(self, directions, positions):
        dr = self.v_0 * directions.u * self.dt
        positions = positions.displace(dr)
        return positions
