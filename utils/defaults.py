from __future__ import print_function, division
import numpy as np

default_model_kwargs = {
    'seed': 1,
    'dim': 1,
    'dt': 0.01,
    'n': 1000,
    'v_0': 1.0,
    'L': np.array([np.inf]),
    'p_0': 1.0,
    't_mem': 5.0,
    'dt_mem': 0.05,
    'Dr_0': 1.0,
}
