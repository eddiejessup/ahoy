from __future__ import print_function, division
import numpy as np
from ahoy.rudders import TumbleRudders, RotationRudders, rudders_factory
from ahoy.utils.meta import make_repr_str
from ahoy.dc_dx_measurers import dc_dx_factory


class RudderSets(object):

    def __init__(self):
        self.sets = []

    @property
    def chi(self):
        for rudders in self.sets:
            if rudders.is_chemotactic:
                return rudders.chi

    @property
    def does_tumbling(self):
        for rs in self.sets:
            if isinstance(rs, TumbleRudders):
                return True
        return False

    @property
    def does_rotation(self):
        for rs in self.sets:
            if isinstance(rs, RotationRudders):
                return True
        return False

    @property
    def chemo_rudders(self):
        for rs in self.sets:
            if rs.is_chemotactic:
                return rs
        raise Exception

    def rotate(self, directions, dt, rng):
        for rudders in self.sets:
            directions = rudders.rotate(directions, dt, rng)
        return directions

    def __repr__(self):
        fs = [('sets', self.sets)]
        return make_repr_str(self, fs)


def rudder_set_factory(temporal_chemo_flag,
                       ds,
                       ps, v_0, dt_mem, t_mem, time,
                       c_field_flag, c_field,
                       onesided_flag, chi,
                       tumble_flag, p_0, tumble_chemo_flag,
                       rotation_flag, Dr_0, dim, rotation_chemo_flag):
    dc_dx_measurer = dc_dx_factory(temporal_chemo_flag,
                                   ds,
                                   ps, v_0, dt_mem, t_mem, p_0, Dr_0, time,
                                   c_field_flag, c_field)
    sets = RudderSets()
    if tumble_flag:
        sets.sets.append(rudders_factory(True, dim, tumble_chemo_flag,
                                         onesided_flag, p_0, chi,
                                         dc_dx_measurer))
    if rotation_flag:
        sets.sets.append(rudders_factory(False, dim, rotation_chemo_flag,
                                         onesided_flag, p_0, chi,
                                         dc_dx_measurer))
    return sets
