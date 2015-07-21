import numpy as np
from scipy.optimize import curve_fit
from ships import directions, rudders
from ships.utils import run_utils
import test


class TestRudders(test.TestBase):

    def setUp(self):
        super(TestRudders, self).setUp()
        self.dt = 0.005
        self.noise_0 = 2.0

    def run_rudder_autocorrelation(self, ruds, dim, Dr_expect):
        n = 10000
        t_max = 0.5

        u_0 = run_utils.get_aligned_directions(n, dim)
        ds = directions.directions_factory(u_0.copy(), dim)
        u_0 = ds.u()
        ts = np.arange(0.0, t_max, self.dt)
        mean_dots = []
        for t in ts:
            ds = ruds.rotate(ds, Dr_expect)
            mean_dot = np.mean(np.sum(u_0 * ds.u(), axis=-1))
            mean_dots.append(mean_dot)

        def exp(x, a):
            return np.exp(-a * x)

        popt, pcov = curve_fit(exp, ts, mean_dots)
        Dr_actual = popt[0]
        self.assertAlmostEqual(Dr_actual, Dr_expect, 1)

    def run_random_seeding(self, rudder_cls, dim):
        n = 10000
        u_0 = run_utils.get_aligned_directions(n, dim)
        num_iterations = 100
        rng_seed = 1

        rng = np.random.RandomState(rng_seed)
        np.random.seed(2)
        ds_1 = directions.directions_factory(u_0.copy(), dim)
        ruds_1 = rudder_cls(self.dt, rng)
        for _ in range(num_iterations):
            ds_1 = ruds_1.rotate(ds_1, self.noise_0)

        rng = np.random.RandomState(rng_seed)
        np.random.seed(3)
        ds_2 = directions.directions_factory(u_0.copy(), dim)
        ruds_2 = rudder_cls(self.dt, rng)
        for _ in range(num_iterations):
            ds_2 = ruds_2.rotate(ds_2, self.noise_0)

        self.assertTrue(np.allclose(ds_1.u(), ds_2.u()))


class TestTumbleRudders(TestRudders):

    def test_tumble_rudders_autocorrelation_1d(self):
        ruds = rudders.TumbleRudders(self.dt, self.rng)
        self.run_rudder_autocorrelation(ruds, 1, 2.0)

    def test_tumble_rudders_autocorrelation_2d(self):
        ruds = rudders.TumbleRudders(self.dt, self.rng)
        self.run_rudder_autocorrelation(ruds, 2, 2.0)

    def test_tumble_rudders_random_seeding_1d(self):
        self.run_random_seeding(rudders.TumbleRudders, 1)

    def test_tumble_rudders_random_seeding_2d(self):
        self.run_random_seeding(rudders.TumbleRudders, 2)

    def run_tumble_rate_nd(self, dim):
        n = 2000
        dt = 1.0
        p = self.rng.uniform(0.0, 1.0, size=n)

        n_expected = (p * dt).sum() / 2.0
        u_0 = run_utils.get_uniform_directions(n, dim, self.rng)
        ds = directions.directions_factory(u_0, dim)
        u_0 = ds.u()
        ruds = rudders.TumbleRudders(dt, self.rng)
        ds_rot = ruds.rotate(ds, p)
        u_change = np.not_equal(u_0, ds_rot.u())
        n_actual = np.any(u_change, axis=-1).sum()
        err = np.abs(n_expected - n_actual) / n_expected
        self.assertTrue(err < 0.05)

    def test_tumble_rate_1d(self):
        self.run_tumble_rate_nd(1)

    def test_tumble_rate_2d(self):
        self.run_tumble_rate_nd(1)


class TestRotationRudders(TestRudders):

    def test_rotation_rudders_autocorrelation(self):
        ruds = rudders.RotationRudders2D(self.dt, self.rng)
        self.run_rudder_autocorrelation(ruds, 2, 2.0)

    def test_rotation_rudders_random_seeding(self):
        self.run_random_seeding(rudders.RotationRudders2D, 2)
