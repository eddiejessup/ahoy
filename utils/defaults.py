from __future__ import print_function, division
import numpy as np

# Make 1d default model args
default_model_1d_kwargs = {
    'seed': 1,
    'dt': 0.01,
    'L': np.array([np.inf]),
    'n': 1000,
    'v_0': 1.0,
    'p_0': 1.0,
}

# Make 2d default model args
default_extra_model_2d_kwargs = {
    'Dr_0': 0.2,
    'L': np.array([1.0, 1.0]),
}
default_model_2d_kwargs = dict(default_model_1d_kwargs,
                               **default_extra_model_2d_kwargs)
