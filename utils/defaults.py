import numpy as np

pore_L = np.array([200.0, 200.0])
pore_R = 30.0

default_ship_kwargs = {
    'seed': 1,
    'dim': 2,
    'dt': 0.01,
    'n': 5000,

    'aligned_flag': False,
    'v_0': 20.0,

    'L': None,
    'origin_flags': None,

    'chi': 0.0,
    'onesided_flag': False,

    'p_0': 0.0,
    'tumble_chemo_flag': False,

    'Dr_0': 0.0,
    'rotation_chemo_flag': False,

    'temporal_chemo_flag': False,
    'dt_mem': 0.1,
    't_mem': 5.0,

    'pore_flag': False,
}

default_pore_ship_kwargs = default_ship_kwargs.copy()
default_pore_ship_kwargs['pore_flag'] = True
default_pore_ship_kwargs['L'] = pore_L
default_pore_ship_kwargs['pore_R'] = pore_R

default_field_ship_kwargs = default_pore_ship_kwargs.copy()
del default_field_ship_kwargs['n']

combo_to_chi = {
    ('Dr_0', False, False): 0.38747846573137146,
    ('Dr_0', False, True): 0.91610356364005729,
    ('Dr_0', True, False): 0.55171912452935623,
    ('Dr_0', True, True): 0.95031324074045198,
    ('p_0', False, False): 0.38527303561393739,
    ('p_0', False, True): 0.85519056757936063,
    ('p_0', True, False): 0.54788034865582758,
    ('p_0', True, True): 0.88731933438050015
}
