from __future__ import print_function, division
import numpy as np
from ciabatta import run_utils as crun_utils
import defaults
from ships.utils import factories
from ships import obstructers


def run():
    extra_model_kwargs = {
        'dim': 2,
        'aligned_flag': False,
        'spatial_flag': True,
        'origin_flag': True,
        'L': np.array([2.0, 2.0]),
        'chi': 0.9,
        'onesided_flag': False,
        'tumble_chemo_flag': True,
        'rotation_chemo_flag': False,
        'spatial_chemo_flag': True,
        'p_0': 1.0,
        'Dr_0': 0.0,
    }
    # turner = obstructers.BounceBackTurner()
    # turner = obstructers.ReflectTurner()
    turner = obstructers.AlignTurner()
    R = 0.4
    obs = obstructers.SingleSphereObstructer(turner, R)
    extra_model_kwargs['obstructer'] = obs

    model_kwargs = dict(defaults.default_model_kwargs, **extra_model_kwargs)

    model = factories.model_factory(**model_kwargs)

    output_every = 1
    t_upto = 2.0
    output_dir = None
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
        'onesided_flag': False,
        'tumble_chemo_flag': True,
        'rotation_chemo_flag': False,
        'spatial_chemo_flag': False,
        'p_0': 1.0,
        'Dr_0': 0.0,
    }
    obs = None
    extra_model_kwargs['obstructer'] = obs
    model_kwargs = dict(defaults.default_model_kwargs, **extra_model_kwargs)

    output_every = 4000
    t_upto = 2000.0
    chis = np.linspace(0.0, 0.95, 9)
    force_resume = True
    parallel = True

    crun_utils.run_field_scan(factories.model_factory, model_kwargs,
                              output_every, t_upto, 'chi', chis,
                              force_resume=force_resume, parallel=parallel)


if __name__ == '__main__':
    run()
    # run_chi_scan()
