from __future__ import print_function, division


class Model(object):

    def __init__(self, dt, agents):
        self.dt = dt
        self.dim = agents.positions.dim
        self.agents = agents

        self.t = 0.0
        self.i = 0

    def iterate(self):
        self.agents.iterate()

        self.t += self.dt
        self.i += 1

    def __repr__(self):
        return 'Ships_d={},{}'.format(self.dim, self.agents)
