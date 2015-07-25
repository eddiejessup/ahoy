from __future__ import print_function, division
import numpy as np
import ships
from ships.rudders import TumbleRudders, rotation_rudders_factory
from ships import measurers
from ships.measurers import noise_measurer_factory


def model_factory(seed, dim, dt, n, aligned_flag, spatial_flag,
                  v_0=None, L=None, obstructer=None,
                  chi=None, onesided_flag=None,
                  p_0=None, tumble_chemo_flag=None,
                  Dr_0=None, rotation_chemo_flag=None,
                  spatial_chemo_flag=None, dt_mem=None, t_mem=None):
    rng = np.random.RandomState(seed)

    time = ships.stime.Time(dt)

    ds = ships.directions.directions_factory(dim, aligned_flag=aligned_flag,
                                             n=n, rng=rng)

    if spatial_flag:
        r_0 = ships.positions.get_uniform_points(n, dim, L, rng, obstructer)
        ps = ships.positions.positions_factory(L, r_0)
    else:
        ps = None

    dc_dx_measurer = dc_dx_factory(spatial_chemo_flag,
                                   ds,
                                   ps, v_0, dt_mem, t_mem, p_0, Dr_0, time)

    rudder_sets = []
    if p_0:
        tumble_noise_measurer = noise_measurer_factory(tumble_chemo_flag,
                                                       onesided_flag, p_0, chi,
                                                       dc_dx_measurer)
        rudder_sets.append(TumbleRudders(tumble_noise_measurer, rng))

    if Dr_0 and dim > 1:
        rotation_noise_measurer = noise_measurer_factory(rotation_chemo_flag,
                                                         onesided_flag, Dr_0,
                                                         chi, dc_dx_measurer)
        rudder_sets.append(rotation_rudders_factory(dim,
                                                    rotation_noise_measurer,
                                                    rng))

    agents = agents_factory(spatial_flag, ds, rudder_sets, v_0, ps)
    model = ships.model.ships_factory(spatial_flag, time, agents, obstructer)
    return model


def dc_dx_factory(spatial_chemo_flag,
                  ds=None,
                  ps=None, v_0=None, dt_mem=None, t_mem=None, p_0=None,
                  Dr_0=None, time=None):
    if spatial_chemo_flag:
        n, dim = ds.n, ds.dim
        grad_c_measurer = measurers.ConstantGradCMeasurer(n, dim)
        return measurers.SpatialDcDxMeasurer(ds, grad_c_measurer)
    else:
        if ps is None:
            raise ValueError('Must have spatial agents to do temporal '
                             'chemotaxis')
        c_measurer = measurers.LinearCMeasurer(ps)
        D_rot_0 = p_0 + Dr_0
        t_rot_0 = 1.0 / D_rot_0
        return measurers.TemporalDcDxMeasurer(c_measurer, v_0, dt_mem, t_mem,
                                              t_rot_0, time)


def agents_factory(spatial_flag, ds, rudder_sets,
                   v_0=None, ps=None):
    if spatial_flag:
        swimmers = ships.swimmers.Swimmers(v_0, ds)
        return ships.agents.SpatialAgents(ds, ps, rudder_sets, swimmers)
    else:
        return ships.agents.Agents(ds, rudder_sets)
