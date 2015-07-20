from __future__ import print_function, division
import numpy as np
from ciabatta import run_utils as crun_utils
import defaults
import run_utils


def run_1d():
    model_1d_kwargs = defaults.default_model_1d_kwargs.copy()
    extra_model_kwargs = {
        'chemo_flag': False,
        'chi': 0.0,
        'origin_flag': True,
        'aligned_flag': False,
        'n': 5000,
    }
    model_1d_kwargs.update(extra_model_kwargs)

    model = run_utils.model_factory_1d(**model_1d_kwargs)

    output_every = 200
    t_upto = 500.0
    output_dir = 'test_1'
    force_resume = None

    crun_utils.run_model(output_every, output_dir, m=model,
                         force_resume=force_resume, t_upto=t_upto)

if __name__ == '__main__':
    run_1d()
