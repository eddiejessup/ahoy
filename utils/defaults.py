import numpy as np

L_porous = np.array([200.0, 200.0])
R_porous = 30.0
rng = np.random.RandomState(1)

default_ship_kwargs = {
    'rng': rng,
    'dt': 0.01,
    'aligned_flag': False,

    'n': 5000,
    'v_0': 20.0,

    'L': None,
    'origin_flags': None,
    'obstructor': None,

    'chi': 0.0,
    'onesided_flag': False,

    'p_0': 0.0,
    'tumble_chemo_flag': False,

    'Dr_0': 0.0,
    'rotation_chemo_flag': False,

    'temporal_chemo_flag': False,
    'dt_mem': 0.1,
    't_mem': 5.0,
}

porous_obstructor_kwargs = {
    'rng': rng,
    'periodic': True,
    'L': L_porous,
    'R': R_porous,
}
