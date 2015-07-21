from __future__ import print_function, division
import numpy as np
from ships import positions, directions
from ships.estimators import (LinearSpatialCDotEstimators,
                              LinearTemporalCDotEstimators)
from ships.utils import run_utils
import test


class TestLinearSpatialCDotEstimators(test.TestBase):

    def setUp(self):
        super(TestLinearSpatialCDotEstimators, self).setUp()
        self.n = 1000
        self.v_0 = 2.2
        self.estimators = LinearSpatialCDotEstimators(self.v_0)

    def run_parallel_nd(self, dim):
        L = dim * [1.0]
        r_0 = run_utils.get_uniform_points(self.n, dim, L, self.rng)
        ps = positions.Positions(r_0, L)

        u_0 = np.zeros_like(ps.r)
        u_0[:, 0] = 1.0
        ds = directions.directions_factory(u_0, dim)
        cdots = self.estimators.get_cdots(ps, ds)
        self.assertTrue(np.allclose(cdots, self.v_0))

    def run_antiparallel_nd(self, dim):
        L = dim * [1.0]
        r_0 = run_utils.get_uniform_points(self.n, dim, L, self.rng)
        ps = positions.Positions(r_0, L)

        u_0 = np.zeros_like(ps.r)
        u_0[:, 0] = -1.0
        ds = directions.directions_factory(u_0, dim)
        cdots = self.estimators.get_cdots(ps, ds)
        self.assertTrue(np.allclose(cdots, -self.v_0))

    def run_perp_nd(self, dim):
        L = dim * [1.0]
        r_0 = run_utils.get_uniform_points(self.n, dim, L, self.rng)
        ps = positions.Positions(r_0, L)

        u_0 = np.zeros_like(ps.r)
        u_0[:, 1] = -1.0
        ds = directions.directions_factory(u_0, dim)
        cdots = self.estimators.get_cdots(ps, ds)
        self.assertTrue(np.allclose(cdots, 0.0))

    def test_parallel_2d(self):
        self.run_parallel_nd(2)

    def test_parallel_1d(self):
        self.run_parallel_nd(1)

    def test_antiparallel_2d(self):
        self.run_antiparallel_nd(2)

    def test_antiparallel_1d(self):
        self.run_antiparallel_nd(1)

    def test_perp_2d(self):
        self.run_perp_nd(2)


class MockPositions(object):
    def __init__(self, dim, dt, n, v_0, u_0):
        self.dt = dt
        self.r = np.zeros([n, dim])
        self.v_0 = v_0
        self.u_0 = u_0

    def iterate(self):
        self.r += self.v_0 * self.u_0 * self.dt

    def get_unwrapped_dr(self):
        return self.r


class TestLinearTemporalCDotEstimators(test.TestBase):

    def setUp(self):
        super(TestLinearTemporalCDotEstimators, self).setUp()
        self.v_0 = 2.2
        self.dt = 0.0005
        self.t_mem = 5.0
        self.t_rot_0 = 1.0

    def run_parallel_nd(self, dim):
        u_0 = np.zeros([self.n, dim])
        u_0[:, 0] = 1.0
        ps = MockPositions(dim, self.dt, self.n, self.v_0, u_0)
        self.estimators = LinearTemporalCDotEstimators(self.dt, self.n,
                                                       self.t_mem,
                                                       self.t_rot_0)
        t = 0.0
        while t < self.t_mem:
            ps.iterate()
            cdots = self.estimators.get_cdots(ps, directions=None)
            print(cdots[0])
            t += self.dt
        self.assertTrue(np.allclose(cdots, self.v_0))

    def test_parallel_1d(self):
        self.run_parallel_nd(1)
