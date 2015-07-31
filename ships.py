from __future__ import print_function, division
import numpy as np
from ahoy.utils.meta import make_repr_str
import ahoy
from ahoy import obstructors, agents, field, turners


class Ships(object):

    def __init__(self, rng, time, ags, *args, **kwargs):
        self.rng = rng
        self.time = time
        self.agents = ags

    def iterate(self):
        self.agents.iterate(self.time.dt, self.rng)
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
        fs = [('rng', self.rng), ('time', self.time), ('agents', self.agents)]
        return make_repr_str(self, fs)

    def _get_output_dirname_agent_part(self):
        ags = self.agents

        s = 'n={},align={:d}'.format(ags.positions.n,
                                     ags.directions.aligned_flag)

        # Space and swimming
        if isinstance(ags, ahoy.agents.SpatialAgents):
            s += ',origin={:d},v={:g}'.format(ags.positions.origin_flag,
                                              ags.swimmers.v_0)
            if isinstance(ags.positions, ahoy.positions.PeriodicPositions):
                s += ',L={}'.format(tuple(ags.positions.L_repr()))

        # Rudders
        for rs in ags.rudder_sets:
            nm = rs.noise_measurer
            if isinstance(rs, ahoy.rudders.TumbleRudders):
                noise_str = 'p'
            elif isinstance(rs, ahoy.rudders.RotationRudders):
                noise_str = 'Dr'
            s += ',{}={:g}'.format(noise_str, nm.noise_0)
            if rs.is_chemotactic():
                chemo_temp = nm.is_temporal()
                type_s = 'T' if chemo_temp else 'S'
                s += ',chi={:g},side={:d},type={}'.format(nm.chi,
                                                          2 - rs.is_onesided(),
                                                          type_s)
                if chemo_temp:
                    measurer = nm.dc_dx_measurer
                    s += ',dtMem={:g},tMem={:g}'.format(measurer.dt_mem,
                                                        measurer.t_mem)
        return s

    def get_output_dirname(self):
        s = 'ships_{}D,dt={:g},rng={}'.format(self.dim, self.time.dt,
                                              hash(self.rng))
        s += ',{}'.format(self._get_output_dirname_agent_part())
        return s


class SpatialShips(Ships):

    def __init__(self, rng, time, ags, obstructor):
        super(SpatialShips, self).__init__(rng, time, ags)
        self.obstructor = obstructor

    def iterate(self):
        self.agents.iterate(self.time.dt, self.rng, self.obstructor)
        self.time.iterate()

    def __repr__(self):
        fs = [('time', self.time), ('agents', self.agents),
              ('obstructor', self.obstructor)]
        return make_repr_str(self, fs)

    def _get_output_dirname_obstruction_part(self):
        obs = self.obstructor
        s = ''

        if obs.__class__ is obstructors.NoneObstructor:
            return 'noObs'

        if obs.turner.__class__ is turners.Turner:
            s_turner = 'stall'
        elif obs.turner.__class__ is turners.BounceBackTurner:
            s_turner = 'bback'
        elif obs.turner.__class__ is turners.ReflectTurner:
            s_turner = 'reflect'
        elif obs.turner.__class__ is turners.AlignTurner:
            s_turner = 'align'
        s += 'turn={}'.format(s_turner)

        if obs.__class__ is obstructors.SingleSphereObstructor2D:
            s += ',ss_R={:g}'.format(obs.R)
        elif obs.__class__ is obstructors.PorousObstructor:
            pf = obs.fraction_occupied
            s += ',pore_R={:g},pf={:g},periodic={:d}'.format(obs.R, pf,
                                                             obs.periodic_flag)
        return s

    def get_output_dirname(self):
        s = super(SpatialShips, self).get_output_dirname()
        s += ',{}'.format(self._get_output_dirname_obstruction_part())
        return s


class CFieldShips(SpatialShips):

    def __init__(self, rng, time, ags, obstructor, c_field):
        super(CFieldShips, self).__init__(rng, time, ags, obstructor)
        self.c_field = c_field

    def iterate(self):
        super(CFieldShips, self).iterate()
        self.c_field.iterate(self.agents.positions)

    def __repr__(self):
        fs = [('time', self.time), ('agents', self.agents),
              ('obstructor', self.obstructor), ('c_field', self.c_field)]
        return make_repr_str(self, fs)

    def _get_output_dirname_field_part(self):
        c_field = self.c_field
        s = 'c0={:g},cD={:g},cDelta={:g}'.format(c_field.c_0, c_field.D,
                                                 c_field.delta)
        return s

    def get_output_dirname(self):
        s = super(CFieldShips, self).get_output_dirname()
        s += ',{}'.format(self._get_output_dirname_field_part())
        return s


def ships_factory(rng, dim, dt, n, aligned_flag,
                  chi=None, onesided_flag=None,
                  p_0=None, tumble_chemo_flag=None,
                  Dr_0=None, rotation_chemo_flag=None):
    time = ahoy.stime.Time(dt)
    ags = agents.agents_factory(rng, dim, n, aligned_flag,
                                chi, onesided_flag,
                                p_0, tumble_chemo_flag,
                                Dr_0, rotation_chemo_flag)
    return Ships(rng, time, ags)


def spatial_ships_factory(rng, dim, dt, n, aligned_flag,
                          v_0,
                          L=None, origin_flags=None, obstructor=None,
                          chi=None, onesided_flag=None,
                          p_0=None, tumble_chemo_flag=None,
                          Dr_0=None, rotation_chemo_flag=None,
                          temporal_chemo_flag=None, dt_mem=None, t_mem=None):
    time = ahoy.stime.Time(dt)
    if obstructor is None:
        obstructor = obstructors.NoneObstructor()
    ags = agents.spatial_agents_factory(rng, dim, n, aligned_flag,
                                        v_0,
                                        L, origin_flags, obstructor,
                                        chi, onesided_flag,
                                        p_0, tumble_chemo_flag,
                                        Dr_0, rotation_chemo_flag,
                                        temporal_chemo_flag, dt_mem, t_mem,
                                        time)
    return SpatialShips(rng, time, ags, obstructor)


def c_field_ships_factory(rng, dim, dt, n, aligned_flag,
                          v_0,
                          L,
                          c_dx, c_D, c_delta, c_0,
                          origin_flags=None, obstructor=None,
                          chi=None, onesided_flag=None,
                          p_0=None, tumble_chemo_flag=None,
                          Dr_0=None, rotation_chemo_flag=None,
                          temporal_chemo_flag=None, dt_mem=None, t_mem=None):
    time = ahoy.stime.Time(dt)
    if obstructor is None:
        obstructor = ahoy.obstructors.NoneObstructor()
    mesh = obstructor.get_mesh(L, c_dx)
    c_field = field.FoodField(dim, mesh, dt, c_D, c_delta, c_0)
    ags = agents.spatial_agents_factory(rng, dim, n, aligned_flag,
                                        v_0,
                                        L, origin_flags, obstructor,
                                        chi, onesided_flag,
                                        p_0, tumble_chemo_flag,
                                        Dr_0, rotation_chemo_flag,
                                        temporal_chemo_flag, dt_mem, t_mem,
                                        time,
                                        c_field)
    return CFieldShips(rng, time, ags, obstructor, c_field)
