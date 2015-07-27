from __future__ import print_function, division
import numpy as np
from agaro import run_utils
from ahoy.obstructors import SingleSphereObstructor, NoneObstructor, PorousObstructor
from ahoy import model, turners


def run_spatial():
    rng = np.random.RandomState(1)
    L = np.array([2.0, 2.0])
    obstructor_kwargs = {
        'turner': turners.AlignTurner(),
        'R': 0.1,
        'pf': 0.1,
        'L': L,
        'rng': rng,
    }
    obstructor = PorousObstructor(**obstructor_kwargs)
    model_kwargs = {
        'rng': rng,
        'dim': 2,
        'dt': 0.01,
        'n': 1000,
        'aligned_flag': False,
        'v_0': 1.0,
        'L': L,
        'origin_flags': np.array([False, False]),

        'obstructor': obstructor,

        'chi': 0.9,
        'onesided_flag': False,
        'p_0': 1.0,
        'tumble_chemo_flag': True,
        'Dr_0': 0.0,
        'rotation_chemo_flag': False,
        'spatial_chemo_flag': True,
    }
    m = model.spatial_ships_factory(**model_kwargs)

    output_every = 1
    t_upto = 0.5
    output_dir = None
    force_resume = None

    run_utils.run_model(output_every, output_dir, m=m,
                        force_resume=force_resume, t_upto=t_upto)


def run_chi_scan():
    extra_model_kwargs = {
        'dim': 1,
        'aligned_flag': False,
        'spatial_flag': True,
        'origin_flag': True,
        'L': None,
        'onesided_flag': False,
        'tumble_chemo_flag': True,
        'rotation_chemo_flag': False,
        'spatial_chemo_flag': False,
        'p_0': 1.0,
        'Dr_0': 0.0,
    }
    obs = None
    extra_model_kwargs['obstructor'] = obs
    model_kwargs = dict(defaults.default_model_kwargs, **extra_model_kwargs)

    output_every = 4000
    t_upto = 2000.0
    chis = np.linspace(0.0, 0.95, 9)
    force_resume = True
    parallel = True

    run_utils.run_field_scan(model.ships_factory, model_kwargs,
                             output_every, t_upto, 'chi', chis,
                             force_resume=force_resume, parallel=parallel)


def run_field():
    model_kwargs = {
        'dim': 2,
        'dt': 0.1,
        'n': 1000,
        'aligned_flag': True,
        'v_0': 20.0,
        # 'L': np.array([1000.0, 100.0]),
        'L': np.array([1000.0, 1000.0]),
        # 'origin_flags': np.array([True, False]),
        'origin_flags': np.array([False, False]),

        # 'c_dx': np.array([20, 10.0]),
        'c_dx': np.array([50.0, 50.0]),
        'c_D': 10.0,
        'c_delta': 10.0,
        'c_0': 10.0,

        'chi': 0.9,
        'onesided_flag': False,
        'p_0': 1.0,
        'tumble_chemo_flag': True,
        'Dr_0': 0.0,
        'rotation_chemo_flag': False,
        'spatial_chemo_flag': True,
        # 'dt_mem': 0.05,
        # 't_mem': 5.0,
    }

    turner = turners.AlignTurner()
    R = 100.0
    # obs = SingleSphereObstructor(turner, R)
    # obs = NoneObstructor()
    obs = PorousObstructor(turner, R, model_kwargs['L'], pf=0.1, rng=None)
    model_kwargs['obstructor'] = obs

    rng_seed = 1

    m = model.c_field_ships_factory(rng_seed, **model_kwargs)

    output_every = 1
    t_upto = 5.0
    output_dir = None
    force_resume = None

    run_utils.run_model(output_every, output_dir, m=m,
                        force_resume=force_resume, t_upto=t_upto)


if __name__ == '__main__':
    # run_field()
    run_spatial()
    # run_chi_scan()
