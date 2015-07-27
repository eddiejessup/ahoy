import numpy as np
from scipy.optimize import curve_fit
from ahoy import directions, rudders, noise_measurers
import test


class TestRotationRudders2D(test.TestBase):
    rudders_cls = rudders.RotationRudders2D

    def setUp(self):
        super(TestRotationRudders2D, self).setUp()
        self.dt = 0.005
        self.noise_0 = 2.0
        self.noise_measurer = noise_measurers.NoiseMeasurer(self.noise_0)

    def run_rudder_autocorrelation(self, dim, t_rot_expect):
        n = 10000
        t_max = 0.5

        ds = directions.make_directions(n, dim, aligned_flag=True)
        rudders = self.rudders_cls(self.noise_measurer, self.rng)

        u_0 = ds.u()
        ts = np.arange(0.0, t_max, self.dt)
        mean_dots = []
        for t in ts:
            ds = rudders.rotate(ds, self.dt)
            mean_dot = np.mean(np.sum(u_0 * ds.u(), axis=-1))
            mean_dots.append(mean_dot)

        def exp(t, t_rot):
            return np.exp(-t / t_rot)

        popt, pcov = curve_fit(exp, ts, mean_dots)
        t_rot_actual = popt[0]
        self.assertAlmostEqual(t_rot_actual, t_rot_expect, 1)

    def run_random_seeding(self, dim):
        n = 1000
        num_iterations = 100
        rng_seed = 1

        rng = np.random.RandomState(rng_seed)
        np.random.seed(2)
        ds_1 = directions.make_directions(n, dim, aligned_flag=True)
        ruds_1 = self.rudders_cls(self.noise_measurer, rng)
        for _ in range(num_iterations):
            ds_1 = ruds_1.rotate(ds_1, self.dt)

        rng = np.random.RandomState(rng_seed)
        np.random.seed(3)
        ds_2 = directions.make_directions(n, dim, aligned_flag=True)
        ruds_2 = self.rudders_cls(self.noise_measurer, rng)
        for _ in range(num_iterations):
            ds_2 = ruds_2.rotate(ds_2, self.dt)

        self.assertTrue(np.allclose(ds_1.u(), ds_2.u()))

    def test_rudders_autocorrelation_2d(self):
        self.run_rudder_autocorrelation(2, 1.0 / self.noise_0)

    def test_rudders_random_seeding_2d(self):
        self.run_random_seeding(2)


class TestTumbleRudders(TestRotationRudders2D):
    rudders_cls = rudders.TumbleRudders

    def run_tumble_rate_nd(self, dim):
        n = 1000
        dt = 1.0
        p = self.rng.uniform(0.0, 1.0, size=n)
        noise_measurer = noise_measurers.NoiseMeasurer(p)

        n_expected = (p * dt).sum() / 2.0
        ds = directions.make_directions(n, dim, aligned_flag=False,
                                        rng=self.rng)
        u_0 = ds.u()
        ruds = rudders.TumbleRudders(noise_measurer, self.rng)
        ds_rot = ruds.rotate(ds, dt)
        u_change = np.not_equal(u_0, ds_rot.u())
        n_actual = np.any(u_change, axis=-1).sum()
        err = np.abs(n_expected - n_actual) / n_expected
        self.assertTrue(err < 0.05)

    def test_tumble_rate_1d(self):
        self.run_tumble_rate_nd(1)

    def test_tumble_rate_2d(self):
        self.run_tumble_rate_nd(1)

    def test_rudders_autocorrelation_1d(self):
        self.run_rudder_autocorrelation(1, 1.0 / self.noise_0)

    def test_rudders_random_seeding_1d(self):
        self.run_random_seeding(1)
