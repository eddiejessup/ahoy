from __future__ import print_function, division
from ahoy.utils.meta import make_repr_str
from ahoy import (directions, measurers, rudders, positions,
                  swimmers)


class Agents(object):

    def __init__(self, directions, rudder_sets):
        self.directions = directions
        self.rudder_sets = rudder_sets

    def iterate(self, dt, rng):
        for ruds in self.rudder_sets:
            self.directions = ruds.rotate(self.directions, dt, rng)

    def get_chi(self):
        for ruds in self.rudder_sets:
            if ruds.is_chemotactic():
                return ruds.get_chi()

    def does_tumbling(self):
        for rs in self.rudder_sets:
            if isinstance(rs, rudders.TumbleRudders):
                return True
        return False

    def does_rotation(self):
        for rs in self.rudder_sets:
            if isinstance(rs, rudders.RotationRudders):
                return True
        return False

    def get_chemo_rudders(self):
        for rs in self.rudder_sets:
            if rs.is_chemotactic():
                return rs
        raise Exception

    def __repr__(self):
        fs = [('directions', self.directions),
              ('rudder_sets', self.rudder_sets)]
        return make_repr_str(self, fs)


class SpatialAgents(Agents):

    def __init__(self, directions, positions, rudder_sets, swimmers):
        super(SpatialAgents, self).__init__(directions, rudder_sets)
        self.positions = positions
        self.swimmers = swimmers

    def iterate(self, dt, rng, obstructor):
        super(SpatialAgents, self).iterate(dt, rng)
        self.positions, dr = self.swimmers.displace(self.positions, dt)
        obstructor.obstruct(self.positions, dr, self.directions)

    def __repr__(self):
        fs = [('directions', self.directions), ('positions', self.positions),
              ('rudder_sets', self.rudder_sets), ('swimmers', self.swimmers)]
        return make_repr_str(self, fs)


def agents_factory(rng, dim, aligned_flag,
                   n=None, rho_0=None,
                   chi=None, onesided_flag=None,
                   p_0=None, tumble_chemo_flag=None,
                   Dr_0=None, rotation_chemo_flag=None,
                   temporal_chemo_flag=None, dt_mem=None, t_mem=None, time=None,
                   spatial_flag=None, v_0=None,
                   periodic_flag=None, L=None, origin_flags=None, obstructor=None,
                   c_field=None):
    if rho_0 is not None:
        try:
            volume_free = obstructor.volume_free
        except AttributeError:
            volume_free = np.product(L)
        n = int(round(rho_0 * volume_free))

    ds = directions.make_directions(n, dim, aligned_flag=aligned_flag, rng=rng)

    if spatial_flag:
        ps = positions.positions_factory(periodic_flag, n, dim, L,
                                         origin_flags, rng, obstructor)
    else:
        ps = None

    dc_dx_measurer = measurers.dc_dx_factory(temporal_chemo_flag,
                                             ds,
                                             ps, v_0, dt_mem, t_mem, p_0, Dr_0,
                                             time,
                                             c_field)

    rudder_sets = rudders.rudder_set_factory(onesided_flag, chi,
                                             dc_dx_measurer,
                                             tumble_chemo_flag, p_0,
                                             rotation_chemo_flag, Dr_0, dim)

    if spatial_flag:
        swims = swimmers.Swimmers(v_0, ds)
        return SpatialAgents(ds, ps, rudder_sets, swims)
    else:
        return Agents(ds, rudder_sets)
