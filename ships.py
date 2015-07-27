from __future__ import print_function, division
import numpy as np
import ahoy
from ahoy import obstructors, agents, field
from ahoy.measurers import TemporalDcDxMeasurer
from ahoy.noise_measurers import (ChemoNoiseMeasurer,
                                  OneSidedChemoNoiseMeasurer)


class Ships(object):

    def __init__(self, time, ags, *args, **kwargs):
        self.time = time
        self.agents = ags

    def iterate(self):
        self.agents.iterate(self.time.dt)
        self.time.iterate()

    @property
    def dim(self):
        return self.agents.directions.dim

    @property
    def t(self):
        return self.time.t

    @property
    def dt(self):
        return self.time.dt

    @property
    def i(self):
        return self.time.i

    def __repr__(self):
        repr_str = 'Ships(time={},agents={})'
        return repr_str.format(self.time, self.agents)

    def get_output_dirname(self):
        align = self.agents.directions.aligned_flag
        spatial = isinstance(self.agents, ahoy.agents.SpatialAgents)
        s = 'Ships_{}D,dt={:g},n={},align={:d}'.format(self.dim, self.time.dt,
                                                       self.agents.n, align)

        if spatial:
            origin_flag = self.agents.positions.origin_flag
            v = self.agents.swimmers.v_0
            s += ',origin={:d},v={:g}'.format(origin_flag, v)
            periodic = isinstance(self.agents.positions,
                                  ahoy.positions.PeriodicPositions)
            if periodic:
                s += ',L={}'.format(tuple(self.agents.positions.L_repr()))

        for rs in self.agents.rudder_sets:
            if isinstance(rs, ahoy.rudders.TumbleRudders):
                tumb_nm = rs.noise_measurer
                p_0 = tumb_nm.noise_0
                s += ',p={:g}'.format(p_0)

                tumb_chemo = isinstance(tumb_nm,
                                        ChemoNoiseMeasurer)
                if tumb_chemo:
                    tumb_chi = tumb_nm.chi
                    tumb_sided = 1 + isinstance(tumb_nm,
                                                OneSidedChemoNoiseMeasurer)
                    tumb_chemo_temp = isinstance(tumb_nm.dc_dx_measurer,
                                                 TemporalDcDxMeasurer)
                    tumb_type_s = 'T' if tumb_chemo_temp else 'S'
                    s += ',pchi={:g},pside={:d},ptype={}'.format(tumb_chi,
                                                                 tumb_sided,
                                                                 tumb_type_s)

                    if tumb_chemo_temp:
                        tumb_chemo_dt_mem = tumb_nm.dc_dx_measurer.dt_mem
                        tumb_chemo_t_mem = tumb_nm.dc_dx_measurer.t_mem
                        s += ',dtmem={:g},ptmem={:g}'.format(tumb_chemo_dt_mem,
                                                             tumb_chemo_t_mem)

            elif isinstance(rs, ahoy.rudders.RotationRudders):
                rot_nm = rs.noise_measurer
                Dr_0 = rot_nm.noise_0
                rot_chemo = isinstance(rot_nm,
                                       ahoy.measurers.ChemoNoiseMeasurer)
                s += ',Dr={:g}'.format(Dr_0)
                if rot_chemo:
                    rot_chi = rot_nm.chi
                    rot_sided = 1 + isinstance(rot_nm,
                                               OneSidedChemoNoiseMeasurer)
                    rot_chemo_temp = isinstance(rot_nm.dc_dx_measurer,
                                                TemporalDcDxMeasurer)
                    rot_type_s = 'T' if rot_chemo_temp else 'S'
                    s += ',Dchi={:g},Dside={:d},Dtype={}'.format(rot_chi,
                                                                 rot_sided,
                                                                 rot_type_s)
                    if rot_chemo_temp:
                        rot_chemo_t_mem = rot_nm.dc_dx_measurer.t_mem
                        s += ',Dtmem={:g}'.format(rot_chemo_t_mem)
        return s


class SpatialShips(Ships):

    def __init__(self, time, ags, obstructor):
        super(SpatialShips, self).__init__(time, ags)
        self.obstructor = obstructor

    def iterate(self):
        self.agents.iterate(self.time.dt, self.obstructor)
        self.time.iterate()

    def __repr__(self):
        repr_str = 'SpatialShips(time={},agents={},obstructor={})'
        return repr_str.format(self.time, self.agents, self.obstructor)


class CFieldShips(SpatialShips):

    def __init__(self, time, ags, obstructor, c_field):
        super(CFieldShips, self).__init__(time, ags, obstructor)
        self.c_field = c_field

    def iterate(self):
        super(CFieldShips, self).iterate()
        self.c_field.iterate(self.agents.positions)

    def __repr__(self):
        repr_str = 'SpatialShips(time={},agents={},obstructor={},c_field={})'
        return repr_str.format(self.time, self.agents, self.obstructor,
                               self.c_field)


def ships_factory(rng, dim, dt, n, aligned_flag,
                  chi=None, onesided_flag=None,
                  p_0=None, tumble_chemo_flag=None,
                  Dr_0=None, rotation_chemo_flag=None):
    time = ahoy.stime.Time(dt)
    ags = agents.agents_factory(rng, dim, n, aligned_flag,
                                chi, onesided_flag,
                                p_0, tumble_chemo_flag,
                                Dr_0, rotation_chemo_flag)
    return Ships(time, ags)


def spatial_ships_factory(rng, dim, dt, n, aligned_flag,
                          v_0,
                          L=None, origin_flags=None, obstructor=None,
                          chi=None, onesided_flag=None,
                          p_0=None, tumble_chemo_flag=None,
                          Dr_0=None, rotation_chemo_flag=None,
                          spatial_chemo_flag=None, dt_mem=None, t_mem=None):
    time = ahoy.stime.Time(dt)
    if obstructor is None:
        obstructor = obstructors.NoneObstructor()
    ags = agents.spatial_agents_factory(rng, dim, n, aligned_flag,
                                        v_0,
                                        L, origin_flags, obstructor,
                                        chi, onesided_flag,
                                        p_0, tumble_chemo_flag,
                                        Dr_0, rotation_chemo_flag,
                                        spatial_chemo_flag, dt_mem, t_mem,
                                        time)
    return SpatialShips(time, ags, obstructor)


def c_field_ships_factory(rng, dim, dt, n, aligned_flag,
                          v_0,
                          L,
                          c_dx, c_D, c_delta, c_0,
                          origin_flags=None, obstructor=None,
                          chi=None, onesided_flag=None,
                          p_0=None, tumble_chemo_flag=None,
                          Dr_0=None, rotation_chemo_flag=None,
                          spatial_chemo_flag=None, dt_mem=None, t_mem=None):
    time = ahoy.stime.Time(dt)
    if obstructor is None:
        obstructor = ahoy.obstructors.NoneObstructor()
    mesh = obstructor.get_mesh(L, c_dx)
    c_field = field.FoodField(dim, mesh, dt, c_D, c_delta, c_0, rng)
    ags = agents.spatial_agents_factory(rng, dim, n, aligned_flag,
                                        v_0,
                                        L, origin_flags, obstructor,
                                        chi, onesided_flag,
                                        p_0, tumble_chemo_flag,
                                        Dr_0, rotation_chemo_flag,
                                        spatial_chemo_flag, dt_mem, t_mem,
                                        time,
                                        c_field)
    return CFieldShips(time, ags, obstructor, c_field)
