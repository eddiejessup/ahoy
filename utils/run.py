from __future__ import print_function, division
import numpy as np
from ciabatta import run_utils as crun_utils
import defaults
from ships.utils import factories, run_utils


def run():
    extra_model_kwargs = {
        'dim': 1,
        'aligned_flag': False,
        'spatial_flag': True,
        'origin_flag': True,
        'L': None,
        'chi': 0.9,
        'onesided_flag': False,
        'tumble_chemo_flag': True,
        'rotation_chemo_flag': False,
        'spatial_chemo_flag': True,
    }
    model_kwargs = dict(defaults.default_model_kwargs, **extra_model_kwargs)

    model = factories.model_factory(**model_kwargs)

    output_every = 100
    t_upto = 500.0
    output_dir = run_utils.get_output_dirname(model)
    force_resume = None

    crun_utils.run_model(output_every, output_dir, m=model,
                         force_resume=force_resume, t_upto=t_upto)


def run_chi_scan():
    extra_model_kwargs = {
        'dim': 1,
        'aligned_flag': False,
        'spatial_flag': True,
        'origin_flag': True,
        'L': None,
        'onesided_flag': True,
        'tumble_chemo_flag': True,
        'rotation_chemo_flag': False,
        'spatial_chemo_flag': False,
    }
    model_kwargs = dict(defaults.default_model_kwargs, **extra_model_kwargs)

    output_every = 1000
    t_upto = 500.0
    chis = np.linspace(0.0, 0.95, 9)
    force_resume = True
    parallel = True

    crun_utils.run_field_scan(factories.model_factory, model_kwargs,
                              output_every, t_upto, 'chi', chis,
                              force_resume=force_resume, parallel=parallel)


if __name__ == '__main__':
    # run()
    run_chi_scan()
