from __future__ import print_function, division
from itertools import product
import numpy as np
from agaro import run_utils
import ahoy.turners
from ahoy import ships
from ahoy.utils.defaults import (default_ship_kwargs, default_pore_ship_kwargs,
                                 default_field_ship_kwargs, combo_to_chi,
                                 pore_L)


def run_spatial():
    extra_ship_kwargs = {
        'pore_turner': ahoy.turners.AlignTurner(),
        'pore_pf': 0.0707,
    }
    ship_kwargs = dict(default_pore_ship_kwargs, **extra_ship_kwargs)

    shps = ships.spatial_ships_factory(**ship_kwargs)

    t_output_every = 100.0
    t_upto = 5000.0
    output_dir = None
    force_resume = None

    run_utils.run_model(t_output_every, output_dir, m=shps,
                        force_resume=force_resume, t_upto=t_upto)


def run_Dr_scan():
    extra_ship_kwargs = {
        'pore_turner': ahoy.turners.BounceBackTurner(),
        'pore_pf': 0.8,
    }
    ship_kwargs = dict(default_pore_ship_kwargs, **extra_ship_kwargs)

    t_output_every = 5.0
    t_upto = 300.0
    Dr_0s = np.logspace(-2, 2, 9)
    force_resume = True
    parallel = True

    run_utils.run_field_scan(ships.spatial_ships_factory, ship_kwargs,
                             t_output_every, t_upto, 'Dr_0', Dr_0s,
                             force_resume=force_resume, parallel=parallel)


def run_pf_scan():
    extra_ship_kwargs = {
        'pore_turner': ahoy.turners.ReflectTurner(),
    }
    ship_kwargs = dict(default_pore_ship_kwargs, **extra_ship_kwargs)

    t_output_every = 100.0
    t_upto = 500.0
    pfs = np.linspace(0.0, 0.8, 9)
    force_resume = True
    parallel = True

    noise_vars = ['Dr_0', 'p_0']
    turners = [ahoy.turners.Turner(), ahoy.turners.BounceBackTurner(),
               ahoy.turners.ReflectTurner(), ahoy.turners.AlignTurner()]
    combos = product(noise_vars, turners)
    for noise_var, turner in combos:
        if noise_var == 'Dr_0':
            ship_kwargs['Dr_0'] = 1.0
            ship_kwargs['p_0'] = 0.0
        else:
            ship_kwargs['Dr_0'] = 0.0
            ship_kwargs['p_0'] = 1.0
        ship_kwargs['pore_turner'] = turner

        run_utils.run_field_scan(ships.spatial_ships_factory, ship_kwargs,
                                 t_output_every, t_upto,
                                 'pore_pf', pfs,
                                 force_resume=force_resume, parallel=parallel)


def run_chi_scan():
    ship_kwargs = default_ship_kwargs.copy()

    t_output_every = 1.0
    t_upto = 10.0
    chis = np.linspace(0.0, 0.95, 3)
    force_resume = True
    parallel = True

    dims = [1, 2]
    noise_vars = ['Dr_0', 'p_0']
    onesided_flags = [True, False]
    temporal_chemo_flags = [True, False]

    combos = product(noise_vars, dims, onesided_flags, temporal_chemo_flags)
    for noise_var, dim, onesided_flag, temporal_chemo_flag in combos:
        if noise_var == 'Dr_0':
            if dim == 1:
                continue
            ship_kwargs['Dr_0'] = 1.0
            ship_kwargs['rotation_chemo_flag'] = True
            ship_kwargs['p_0'] = 0.0
            ship_kwargs['tumble_chemo_flag'] = False
        else:
            ship_kwargs['Dr_0'] = 0.0
            ship_kwargs['rotation_chemo_flag'] = False
            ship_kwargs['p_0'] = 1.0
            ship_kwargs['tumble_chemo_flag'] = True
        ship_kwargs['dim'] = dim
        ship_kwargs['onesided_flag'] = onesided_flag
        ship_kwargs['temporal_chemo_flag'] = temporal_chemo_flag

        run_utils.run_field_scan(ships.spatial_ships_factory, ship_kwargs,
                                 t_output_every, t_upto, 'chi', chis,
                                 force_resume=force_resume, parallel=parallel)


def run_pf_scan_drift():
    extra_ship_kwargs = {
        'pore_turner': ahoy.turners.AlignTurner(),
    }
    ship_kwargs = dict(default_pore_ship_kwargs, **extra_ship_kwargs)

    t_output_every = 100.0
    t_upto = 500.0
    pore_pfs = np.linspace(0.0, 0.8, 11)
    force_resume = True
    parallel = True

    noise_vars = ['Dr_0', 'p_0']
    onesided_flags = [True, False]
    temporal_chemo_flags = [True, False]

    combos = product(noise_vars, onesided_flags, temporal_chemo_flags)

    for combo in combos:
        noise_var, onesided_flag, temporal_chemo_flag = combo
        if noise_var == 'Dr_0':
            ship_kwargs['Dr_0'] = 1.0
            ship_kwargs['rotation_chemo_flag'] = True
            ship_kwargs['p_0'] = 0.0
            ship_kwargs['tumble_chemo_flag'] = False
        else:
            ship_kwargs['Dr_0'] = 0.0
            ship_kwargs['rotation_chemo_flag'] = False
            ship_kwargs['p_0'] = 1.0
            ship_kwargs['tumble_chemo_flag'] = True
        ship_kwargs['onesided_flag'] = onesided_flag
        ship_kwargs['temporal_chemo_flag'] = temporal_chemo_flag
        ship_kwargs['chi'] = combo_to_chi[combo]

        run_utils.run_field_scan(ships.spatial_ships_factory, ship_kwargs,
                                 t_output_every, t_upto,
                                 'pore_pf', pore_pfs,
                                 force_resume=force_resume, parallel=parallel)


def run_field():
    rho_0 = 0.04
    c_delta_0 = 0.1
    c_delta = c_delta_0 / rho_0

    extra_ship_kwargs = {
        'rho_0': rho_0,
        'L': np.array([250.0, 200.0]),
        'origin_flags': np.array([True, False]),
        'aligned_flag': False,

        'pore_flag': True,
        'pore_turner': ahoy.turners.AlignTurner(),
        'pore_pf': 0.4,
        'pore_R': 20.0,

        'c_dx': 20.0,
        'c_D': 10.0,
        'c_delta': c_delta,
        'c_0': 1.0,

        'chi': 15.0,
        'p_0': 1.0,
        'tumble_chemo_flag': True,
    }
    ship_kwargs = dict(default_field_ship_kwargs, **extra_ship_kwargs)

    shps = ships.c_field_ships_factory(**ship_kwargs)

    t_output_every = 0.1
    t_upto = 5.0
    output_dir = None
    force_resume = None

    run_utils.run_model(t_output_every, output_dir, m=shps,
                        force_resume=force_resume, t_upto=t_upto)
