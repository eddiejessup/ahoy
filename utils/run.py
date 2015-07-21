from __future__ import print_function, division
import numpy as np
from ciabatta import run_utils as crun_utils
import defaults
import run_utils


def run_1d():
    extra_model_kwargs = {
        'origin_flag': True,
        'aligned_flag': False,
        'p_0': 0.0,
    }
    model_kwargs = dict(defaults.default_model_1d_kwargs, **extra_model_kwargs)

    model = run_utils.model_factory(**model_kwargs)

    output_every = 100
    t_upto = 500.0
    output_dir = None
    force_resume = None

    crun_utils.run_model(output_every, output_dir, m=model,
                         force_resume=force_resume, t_upto=t_upto)


def run_2d():
    extra_model_kwargs = {
        'origin_flag': True,
        'aligned_flag': False,
        'p_0': 0.0,
        'D_rot_0': 1.0,
    }
    model_kwargs = dict(defaults.default_model_2d_kwargs, **extra_model_kwargs)

    model = run_utils.model_factory(**model_kwargs)

    output_every = 100
    t_upto = 500.0
    output_dir = 'test_2d'
    force_resume = None

    crun_utils.run_model(output_every, output_dir, m=model,
                         force_resume=force_resume, t_upto=t_upto)


def run_chi_scan_2d():
    extra_model_kwargs = {
        'origin_flag': True,
        'aligned_flag': False,
        'p_0': 0.0,
        'D_rot_0': 1.0,
    }
    model_kwargs = dict(defaults.default_model_2d_kwargs, **extra_model_kwargs)

    model = run_utils.model_factory(**model_kwargs)

    output_every = 100
    t_upto = 500.0
    chis = np.linspace(0.0, 800.0, 22)
    force_resume = True
    parallel = True

    brun_utils.run_field_scan(model.Model2D, model_kwargs, output_every, t_upto,
                              'chi', chis, force_resume, parallel)


if __name__ == '__main__':
    run_1d()
    # run_2d()
