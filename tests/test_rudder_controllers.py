from __future__ import print_function, division
import numpy as np
from ships import positions, directions, rudders, estimators
from ships.rudder_controllers import (ChemoRudderControllers,
                                      OneSidedChemoRudderControllers)
import test
from ships.utils import run_utils


class TestChemoRudderControllers1D(test.TestBase):
    dim = 1
    L = np.array([0.7])

    def setUp(self):
        super(TestChemoRudderControllers1D, self).setUp()
        self.n = 1000
        self.v_0 = 2.2
        self.estimators = estimators.LinearSpatialCDotEstimators(self.v_0)
        self.noise_0 = 1.5
        self.dt = 0.1
        self.ps = self.get_ps()
        self.chemo_rud_conts_cls = ChemoRudderControllers

    def run_valchemo(self, ps, ds, chi, noise_expected):
        ruds = rudders.TumbleRudders(self.dt, self.rng)
        rud_conts = self.chemo_rud_conts_cls(ruds, self.noise_0, self.v_0, chi,
                                             self.estimators)
        noise = rud_conts.get_noise(ps, ds)
        self.assertTrue(np.allclose(noise, noise_expected))

    def get_ps(self):
        r_0 = run_utils.get_uniform_points(self.n, self.dim, self.L, self.rng)
        return positions.Positions(r_0, self.L)

    def get_ds_parallel(self):
        u_0 = np.zeros_like(self.ps.r)
        u_0[:, 0] = 1.0
        return directions.directions_factory(u_0, self.dim)

    def get_ds_antiparallel(self):
        u_0 = np.zeros_like(self.ps.r)
        u_0[:, 0] = -1.0
        return directions.directions_factory(u_0, self.dim)

    def test_parallel_nochemo(self):
        ds = self.get_ds_parallel()
        self.run_valchemo(self.ps, ds, 0.0, self.noise_0)

    def test_antiparallel_nochemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(self.ps, ds, 0.0, self.noise_0)

    def test_parallel_maxchemo(self):
        ds = self.get_ds_parallel()
        self.run_valchemo(self.ps, ds, 1.0, 0.0)

    def test_antiparallel_maxchemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(self.ps, ds, 1.0, 2.0 * self.noise_0)

    def test_parallel_halfchemo(self):
        ds = self.get_ds_parallel()
        self.run_valchemo(self.ps, ds, 0.5, self.noise_0 / 2.0)

    def test_antiparallel_halfchemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(self.ps, ds, 0.5, 1.5 * self.noise_0)


class TestChemoRudderControllers2D(TestChemoRudderControllers1D):
    dim = 2
    L = np.array([0.7, 1.1])

    def get_ds_perp(self):
        u_0 = np.zeros_like(self.ps.r)
        u_0[:, 1] = 1.0
        return directions.directions_factory(u_0, self.dim)

    def test_perp_nochemo(self):
        ds = self.get_ds_perp()
        self.run_valchemo(self.ps, ds, 0.0, self.noise_0)

    def test_perp_maxchemo(self):
        ds = self.get_ds_perp()
        self.run_valchemo(self.ps, ds, 1.0, self.noise_0)

    def test_perp_halfchemo(self):
        ds = self.get_ds_perp()
        self.run_valchemo(self.ps, ds, 0.5, self.noise_0)


class TestOneSidedChemoRudderControllers1D(TestChemoRudderControllers1D):

    def setUp(self):
        super(TestOneSidedChemoRudderControllers1D, self).setUp()
        self.chemo_rud_conts_cls = OneSidedChemoRudderControllers

    def test_antiparallel_maxchemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(self.ps, ds, 1.0, self.noise_0)

    def test_antiparallel_halfchemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(self.ps, ds, 0.5, self.noise_0)


class TestOneSidedChemoRudderControllers2D(TestChemoRudderControllers2D):

    def setUp(self):
        super(TestOneSidedChemoRudderControllers2D, self).setUp()
        self.chemo_rud_conts_cls = OneSidedChemoRudderControllers

    def test_antiparallel_maxchemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(self.ps, ds, 1.0, self.noise_0)

    def test_antiparallel_halfchemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(self.ps, ds, 0.5, self.noise_0)
