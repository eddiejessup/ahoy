from __future__ import print_function, division
import numpy as np
from ships.estimators import LinearTemporalCDotEstimators
from test_estimators import MockPositions


def run():
    v_0 = 2.2
    dt = 0.01
    t_mem = 10.0
    t_rot_0 = 1.0
    n = 2000
    dim = 2

    u_0 = np.zeros([n, dim])
    u_0[:, 0] = 1.0
    ps = MockPositions(dim, dt, n, v_0, u_0)
    estimators = LinearTemporalCDotEstimators(dt, n, t_mem, t_rot_0)
    t = 0.0
    while t < t_mem:
        ps.iterate()
        cdots = estimators.get_cdots(ps, directions=None)
        t += dt
    assert np.allclose(cdots, v_0)
