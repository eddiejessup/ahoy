from __future__ import print_function, division
from itertools import product
import numpy as np
from agaro import run_utils
from ahoy.obstructors import PorousObstructor
import ahoy.turners
from ahoy import ships
from ahoy.utils.defaults import (default_ship_kwargs, porous_obstructor_kwargs,
                                 rng)


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

        'chi': 0.858,
        'onesided_flag': True,

        'Dr_0': 1.0,
        'rotation_chemo_flag': True,

        'spatial_chemo_flag': False,
    }
    ship_kwargs = dict(default_ship_kwargs, **extra_ship_kwargs)
    shps = ships.spatial_ships_factory(**ship_kwargs)

    t_output_every = 100.0
    t_upto = 5000.0
    output_dir = None
    force_resume = None

    run_utils.run_model(t_output_every, output_dir, m=shps,
                        force_resume=force_resume, t_upto=t_upto)


def run_field():
    L = np.array([1000.0, 1000.0])
    obstructor_kwargs = {
        'turner': ahoy.turners.AlignTurner(),
        'R': 100.0,
        'pf': 0.1,
        'L': L,
        'periodic': False,
        'rng': rng,
    }
    extra_ship_kwargs = {
        'dim': 2,
        'L': L,
        'obstructor': PorousObstructor(**obstructor_kwargs),

        'aligned_flag': True,

        'chi': 0.9,

        'p_0': 1.0,
        'tumble_chemo_flag': True,

        'spatial_chemo_flag': True,

        'c_dx': np.array([50.0, 50.0]),
        'c_D': 10.0,
        'c_delta': 10.0,
        'c_0': 10.0,
    }
    ship_kwargs = dict(default_ship_kwargs, **extra_ship_kwargs)

    shps = ships.c_field_ships_factory(**ship_kwargs)

    t_output_every = 0.1
    t_upto = 5.0
    output_dir = None
    force_resume = None

    run_utils.run_model(t_output_every, output_dir, m=shps,
                        force_resume=force_resume, t_upto=t_upto)


def run_chi_scan():
    ship_kwargs = default_ship_kwargs.copy()

    t_output_every = 100.0
    t_upto = 1000.0
    chis = np.linspace(0.0, 0.95, 9)
    force_resume = True
    parallel = True

    dims = [1, 2]
    noise_vars = ['Dr_0', 'p_0']
    onesided_flags = [True, False]
    spatial_chemo_flags = [True, False]

    combos = product(noise_vars, dims, onesided_flags, spatial_chemo_flags)
    for noise_var, dim, onesided_flag, spatial_chemo_flag in combos:
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
        ship_kwargs['spatial_chemo_flag'] = spatial_chemo_flag

        run_utils.run_field_scan(ships.spatial_ships_factory, ship_kwargs,
                                 t_output_every, t_upto, 'chi', chis,
                                 force_resume=force_resume, parallel=parallel)


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

    t_output_every = 50.0
    t_upto = 500.0
    pfs = np.linspace(0.0, 0.8, 9)
    obstructors = [PorousObstructor(pf=pf, **obstructor_kwargs) for pf in pfs]
    force_resume = True
    parallel = True

    noise_vars = ['Dr_0', 'p_0']
    onesided_flags = [True, False]
    spatial_chemo_flags = [True, False]

    combos = product(noise_vars, onesided_flags, spatial_chemo_flags)

    # Values of chi that give equivalent drift speeds in empty space.
    combo_to_chi = {
        ('p_0', True, True): 0.330,
        ('p_0', True, False): 0.802,
        ('p_0', False, True): 0.492,
        ('p_0', False, False): 0.844,
        ('Dr_0', True, True): 0.333,
        ('Dr_0', True, False): 0.858,
        ('Dr_0', False, True): 0.492,
        ('Dr_0', False, False): 0.909,
    }

    for combo in combos:
        noise_var, onesided_flag, spatial_chemo_flag = combo
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
        ship_kwargs['spatial_chemo_flag'] = spatial_chemo_flag
        ship_kwargs['chi'] = combo_to_chi[combo]

        run_utils.run_field_scan(ships.spatial_ships_factory, ship_kwargs,
                                 t_output_every, t_upto,
                                 'obstructor', obstructors,
                                 force_resume=force_resume, parallel=parallel)


if __name__ == '__main__':
    # run_field()
    run_spatial()
    # run_chi_scan()
