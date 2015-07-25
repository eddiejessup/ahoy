from __future__ import print_function, division
import ships
from ships.measurers import OneSidedChemoNoiseMeasurer, TemporalDcDxMeasurer


class Ships(object):

    def __init__(self, time, agents, *args, **kwargs):
        self.time = time
        self.agents = agents

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
        return 'Ships(dim={},dt={:g},ags={})'.format(self.dim, self.time.dt,
                                                     self.agents)

    def get_output_dirname(self):
        align = self.agents.directions.aligned_flag
        spatial = isinstance(self.agents, ships.agents.SpatialAgents)
        s = 'Ships_{}D,dt={:g},n={},align={:d}'.format(self.dim, self.time.dt,
                                                       self.agents.n, align)

        if spatial:
            origin_flag = self.agents.positions.origin_flag
            v = self.agents.swimmers.v_0
            s += ',origin={:d},v={:g}'.format(origin_flag, v)
            periodic = isinstance(self.agents.positions,
                                  ships.positions.PeriodicPositions)
            if periodic:
                s += ',L={}'.format(tuple(self.agents.positions.L_repr()))

        for rs in self.agents.rudder_sets:
            if isinstance(rs, ships.rudders.TumbleRudders):
                tumb_nm = rs.noise_measurer
                p_0 = tumb_nm.noise_0
                s += ',p={:g}'.format(p_0)

                tumb_chemo = isinstance(tumb_nm,
                                        ships.measurers.ChemoNoiseMeasurer)
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

            elif isinstance(rs, ships.rudders.RotationRudders):
                rot_nm = rs.noise_measurer
                Dr_0 = rot_nm.noise_0
                rot_chemo = isinstance(rot_nm,
                                       ships.measurers.ChemoNoiseMeasurer)
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

    def __init__(self, time, agents, obstructer, food_field):
        super(SpatialShips, self).__init__(time, agents)
        self.obstructer = obstructer
        self.food_field = food_field

    def iterate(self):
        self.agents.iterate(self.time.dt, self.obstructer)
        self.food_field.iterate(self.agents.positions)
        self.time.iterate()


def ships_factory(spatial_flag, *args, **kwargs):
    if spatial_flag:
        return ships.model.SpatialShips(*args, **kwargs)
    else:
        return ships.model.Ships(*args, **kwargs)
