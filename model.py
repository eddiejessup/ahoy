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


def model_factory(seed, dim, dt, L, n, v_0, p_0, origin_flag=False,
                  aligned_flag=False,
                  chi=None, onesided_flag=False, tumble_chemo_flag=False,
                  D_rot_0=None, D_rot_chemo_flag=False):
    """D_rot* parameters only relevant in dim > 1"""
    agents = ships.agents.agents_factory(seed, dim, dt, L, n, v_0, p_0,
                                         origin_flag, aligned_flag, chi,
                                         onesided_flag, tumble_chemo_flag,
                                         D_rot_0, D_rot_chemo_flag)
    model = ships.model.Model(dt, agents)
    return model
