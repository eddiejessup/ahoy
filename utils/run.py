from __future__ import print_function, division
from itertools import product
import numpy as np
from agaro import run_utils
from ahoy.obstructors import PorousObstructor
import ahoy.turners
from ahoy import ships
from ahoy.utils.defaults import (default_ship_kwargs, porous_obstructor_kwargs,
                                 rng, combo_to_chi)


def run_spatial():
    extra_obstructor_kwargs = {
        'turner': ahoy.turners.AlignTurner(),
        'pf': 0.0707,
    }
    obstructor_kwargs = dict(porous_obstructor_kwargs,
                             **extra_obstructor_kwargs)
    extra_ship_kwargs = {
        'dim': 2,
        'L': obstructor_kwargs['L'],
        'obstructor': PorousObstructor(**obstructor_kwargs),
    }
    ship_kwargs = dict(default_ship_kwargs, **extra_ship_kwargs)
    shps = ships.spatial_ships_factory(**ship_kwargs)

    t_output_every = 100.0
    t_upto = 5000.0
    output_dir = None
    force_resume = None

    run_utils.run_model(t_output_every, output_dir, m=shps,
                        force_resume=force_resume, t_upto=t_upto)


def run_Dr_scan():
    extra_obstructor_kwargs = {
        'turner': ahoy.turners.BounceBackTurner(),
        'pf': 0.8,
    }
    obstructor_kwargs = dict(porous_obstructor_kwargs,
                             **extra_obstructor_kwargs)
    extra_ship_kwargs = {
        'dim': 2,
        'L': obstructor_kwargs['L'],
        'obstructor': PorousObstructor(**obstructor_kwargs),
    }
    ship_kwargs = dict(default_ship_kwargs, **extra_ship_kwargs)

    t_output_every = 5.0
    t_upto = 300.0
    Dr_0s = np.logspace(-2, 2, 9)
    force_resume = True
    parallel = True

    run_utils.run_field_scan(ships.spatial_ships_factory, ship_kwargs,
                             t_output_every, t_upto, 'Dr_0', Dr_0s,
                             force_resume=force_resume, parallel=parallel)


def run_pf_scan():
    extra_obstructor_kwargs = {
        'turner': ahoy.turners.ReflectTurner(),
    }
    obstructor_kwargs = dict(porous_obstructor_kwargs,
                             **extra_obstructor_kwargs)
    extra_ship_kwargs = {
        'dim': 2,
        'L': obstructor_kwargs['L'],
    }
    ship_kwargs = dict(default_ship_kwargs, **extra_ship_kwargs)

    t_output_every = 100.0
    t_upto = 500.0
    pfs = np.linspace(0.0, 0.8, 9)
    obstructors = [PorousObstructor(pf=pf, **obstructor_kwargs) for pf in pfs]
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
        for obs in obstructors:
            obs.turner = turner

        run_utils.run_field_scan(ships.spatial_ships_factory, ship_kwargs,
                                 t_output_every, t_upto,
                                 'obstructor', obstructors,
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
    extra_obstructor_kwargs = {
        'turner': ahoy.turners.AlignTurner(),
    }
    obstructor_kwargs = dict(porous_obstructor_kwargs,
                             **extra_obstructor_kwargs)
    extra_ship_kwargs = {
        'dim': 2,
        'L': obstructor_kwargs['L'],
    }
    ship_kwargs = dict(default_ship_kwargs, **extra_ship_kwargs)

    t_output_every = 100.0
    t_upto = 400.0
    pfs = np.linspace(0.0, 0.8, 22)
    obstructors = [PorousObstructor(pf=pf, **obstructor_kwargs) for pf in pfs]
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
                                 'obstructor', obstructors,
                                 force_resume=force_resume, parallel=parallel)


def run_field():
    L = np.array([250.0, 200.0])
    extra_obstructor_kwargs = {
        'turner': ahoy.turners.AlignTurner(),
        'pf': 0.4,
        'L': L,
        'periodic': False,
    }
    obstructor_kwargs = dict(porous_obstructor_kwargs,
                             **extra_obstructor_kwargs)
    extra_ship_kwargs = {
        'n': 1000,
        'dim': 2,
        'L': L,
        'obstructor': PorousObstructor(**obstructor_kwargs),
        # 'obstructor': None,
        'origin_flags': np.array([True, False]),
        'aligned_flag': False,
        'c_dx': 20.0,
        'c_D': 10.0,
        'c_delta': 100.0,
        'c_0': 1.0,

        'chi': 50.0,
        'p_0': 1.0,
        'tumble_chemo_flag': True,
    }
    ship_kwargs = dict(default_ship_kwargs, **extra_ship_kwargs)

    shps = ships.c_field_ships_factory(**ship_kwargs)

    t_output_every = 0.1
    t_upto = 5.0
    output_dir = None
    force_resume = None

    run_utils.run_model(t_output_every, output_dir, m=shps,
                        force_resume=force_resume, t_upto=t_upto)
