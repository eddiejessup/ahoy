from __future__ import print_function, division
import numpy as np
import ships
from ships.rudder_controllers import rud_cont_sets_factory


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


def agents_factory(seed, dim, dt, L, n, v_0, p_0, origin_flag=False,
                   aligned_flag=False,
                   chi=None, onesided_flag=False, tumble_chemo_flag=False,
                   D_rot_0=None, D_rot_chemo_flag=False):
    """D_rot* parameters only relevant in dim > 1"""
    rng = np.random.RandomState(seed)
    ps = ships.positions.Positions(L, origin_flag=origin_flag, n=n, dim=dim,
                                   rng=rng)
    ds = ships.directions.directions_factory(dim, n=n,
                                             aligned_flag=aligned_flag,
                                             rng=rng)
    rud_cont_sets = rud_cont_sets_factory(dim, dt, v_0, p_0, chi,
                                          onesided_flag, tumble_chemo_flag,
                                          D_rot_0, D_rot_chemo_flag, rng)
    swims = ships.swimmers.Swimmers(dt, v_0)
    agents = ships.agents.Agents(ps, ds, rud_cont_sets, swims)
    return agents
