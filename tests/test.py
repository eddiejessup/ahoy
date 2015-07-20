from __future__ import print_function, division
import unittest
import numpy as np
from scipy.optimize import curve_fit
from ships import positions, directions, rudders, estimators
from ships.utils import run_utils


class TestBase(unittest.TestCase):

    def setUp(self):
        seed = 1
        self.rng = np.random.RandomState(seed)
        self.n = 100


class TestPositions2D(TestBase):

    def setUp(self):
        super(TestPositions2D, self).setUp()
        self.dim = 2
        self.L = np.array([0.5, 1.5])
        r_0 = run_utils.get_uniform_points(self.n, self.dim, self.L, self.rng)
        self.ps = positions.Positions(r_0, self.L)

    def test_wrapping_down(self):
        dr = np.zeros_like(self.ps.r)
        self.ps.r[0, 0] = self.L[0] / 2.0
        dr[0, 0] = 0.9 * self.L[0]
        self.ps.displace(np.full(self.ps.r.shape, dr))
        self.assertTrue(np.abs(self.ps.r[:, 0]).max() < self.L[0] / 2.0)

    def test_wrapping_up(self):
        dr = np.zeros_like(self.ps.r)
        self.ps.r[-1, -1] = -self.L[-1] / 2.0
        dr[-1, -1] = -0.9 * self.L[-1]
        self.ps.displace(np.full(self.ps.r.shape, dr))
        self.assertTrue(np.abs(self.ps.r[:, -1]).max() < self.L[-1] / 2.0)

    def test_infinite_boundaries(self):
        self.L = np.array([np.inf, 1.7])
        r_0 = np.zeros_like(self.ps.r_0)
        ps = positions.Positions(r_0.copy(), self.L)
        dr = self.rng.uniform(-1.0, 1.0, size=r_0.shape)
        ps.displace(dr)
        r_inf_naive = r_0[:, 0] + dr[:, 0]
        # Check no wrapping along infinite axis
        self.assertTrue(np.allclose(r_inf_naive, ps.r[:, 0]))
        # Check ndone wrapping along finite axis
        self.assertTrue(np.all(np.abs(ps.r[:, 1]) < self.L[1] / 2.0))


class TestPositions1D(TestBase):

    def setUp(self):
        super(TestPositions1D, self).setUp()
        self.dim = 1
        self.L = np.array([1.7])
        r_0 = run_utils.get_uniform_points(self.n, self.dim, self.L, self.rng)
        self.ps = positions.Positions(r_0, self.L)

    def test_wrapping_down(self):
        dr = np.zeros_like(self.ps.r)
        self.ps.r[0, 0] = self.L[0] / 2.0
        dr[0, 0] = 0.9 * self.L[0]
        self.ps.displace(np.full(self.ps.r.shape, dr))
        self.assertTrue(np.abs(self.ps.r[:, 0]).max() < self.L[0] / 2.0)

    def test_wrapping_up(self):
        dr = np.zeros_like(self.ps.r)
        self.ps.r[-1, 0] = -self.L[0] / 2.0
        dr[-1, 0] = -0.9 * self.L[0]
        self.ps.displace(np.full(self.ps.r.shape, dr))
        self.assertTrue(np.abs(self.ps.r[:, 0]).max() < self.L[0] / 2.0)

    def test_infinite_boundaries(self):
        self.L = np.array([np.inf])
        r_0 = np.zeros_like(self.ps.r_0)
        ps = positions.Positions(r_0.copy(), self.L)
        dr = self.rng.uniform(-10.0, 10.0, size=r_0.shape)
        ps.displace(dr)
        r_inf_naive = r_0[:, 0] + dr[:, 0]
        # Check no wrapping along infinite axis
        self.assertTrue(np.allclose(r_inf_naive, ps.r[:, 0]))


class TestDirections1D(TestBase):
    dim = 1

    def setUp(self):
        super(TestDirections1D, self).setUp()
        self.n = 1000
        u_0 = run_utils.get_uniform_directions(self.n, self.dim, self.rng)
        self.ds = directions.directions_factory(u_0, self.dim)

    def test_tumble_identity(self):
        u_0 = self.ds.u()
        tumblers = np.zeros([u_0.shape[0]], dtype=np.bool)
        ds_rot = self.ds.tumble(tumblers)
        u_rot = ds_rot.u()
        self.assertTrue(np.allclose(u_0, u_rot))

    def test_tumble_magnitude_conservation(self):
        mags_0 = np.sum(np.square(self.ds.u()))
        tumblers = np.random.choice([True, False], size=self.ds.n)
        ds_rot = self.ds.tumble(tumblers)
        mags_rot = np.sum(np.square(ds_rot.u()))
        self.assertTrue(np.allclose(mags_0, mags_rot))

    def test_tumble_coverage(self):
        u_0 = self.ds.u()
        tumblers = np.ones([self.ds.n], dtype=np.bool)
        ds_rot = self.ds.tumble(tumblers)
        u_rot = ds_rot.u()
        frac_close = np.isclose(u_0, u_rot).sum() / float(self.ds.n)
        self.assertAlmostEqual(frac_close, 0.5, 1)


class TestDirections2D(TestDirections1D):
    dim = 2

    def test_rotate_identity(self):
        u_0 = self.ds.u()
        dth = np.zeros([u_0.shape[0]])
        ds_rot = self.ds.rotate(dth)
        u_rot = ds_rot.u()
        self.assertTrue(np.allclose(u_0, u_rot))

    def test_rotate_idempotence(self):
        u_0 = self.ds.u()
        dth = self.rng.uniform(-np.pi, np.pi, size=self.n)
        ds_rot = self.ds.rotate(dth)
        ds_rot = ds_rot.rotate(-dth)
        u_rot = ds_rot.u()
        self.assertTrue(np.allclose(u_0, u_rot))

    def test_rotate_periodicity(self):
        u_0 = self.ds.u()
        dth = np.full(u_0.shape[0], np.pi / 2.0)
        ds_rot = self.ds
        for i in range(4):
            ds_rot = ds_rot.rotate(dth)
        u_rot = ds_rot.u()
        self.assertTrue(np.allclose(u_0, u_rot))

    def test_rotate_magnitude_conservation(self):
        mags_0 = np.sum(np.square(self.ds.u()))
        dth = self.rng.uniform(-np.pi, np.pi, size=self.n)
        ds_rot = self.ds.rotate(dth)
        mags_rot = np.sum(np.square(ds_rot.u()))
        self.assertTrue(np.allclose(mags_0, mags_rot))

    def test_tumble_coverage(self):
        u_0 = self.ds.u()
        tumblers = np.ones([self.ds.n], dtype=np.bool)
        ds_rot = self.ds.tumble(tumblers)
        u_rot = ds_rot.u()
        self.assertFalse(np.any(np.isclose(u_0, u_rot)))


class TestRudders(TestBase):

    def setUp(self):
        super(TestRudders, self).setUp()
        self.dt = 0.005
        self.noise_0 = 2.0

    def run_rudder_autocorrelation(self, ruds, dim, Dr_expect):
        n = 10000
        t_max = 5.0

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
        num_iterations = 1000
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


class TestLinearSpatialCDotEstimators(TestBase):

    def setUp(self):
        super(TestLinearSpatialCDotEstimators, self).setUp()
        self.n = 1000
        self.v_0 = 2.2
        self.estimators = estimators.LinearSpatialCDotEstimators(self.v_0)

    def run_parallel_nd(self, dim, L):
        r_0 = run_utils.get_uniform_points(self.n, dim, L, self.rng)
        ps = positions.Positions(r_0, L)

        u_0 = np.zeros_like(ps.r)
        u_0[:, 0] = 1.0
        ds = directions.directions_factory(u_0, dim)
        cdots = self.estimators.get_cdots(ps, ds)
        self.assertTrue(np.allclose(cdots, self.v_0))

    def run_antiparallel_nd(self, dim, L):
        r_0 = run_utils.get_uniform_points(self.n, dim, L, self.rng)
        ps = positions.Positions(r_0, L)

        u_0 = np.zeros_like(ps.r)
        u_0[:, 0] = -1.0
        ds = directions.directions_factory(u_0, dim)
        cdots = self.estimators.get_cdots(ps, ds)
        self.assertTrue(np.allclose(cdots, -self.v_0))

    def run_perp_nd(self, dim, L):
        r_0 = run_utils.get_uniform_points(self.n, dim, L, self.rng)
        ps = positions.Positions(r_0, L)

        u_0 = np.zeros_like(ps.r)
        u_0[:, 1] = -1.0
        ds = directions.directions_factory(u_0, dim)
        cdots = self.estimators.get_cdots(ps, ds)
        self.assertTrue(np.allclose(cdots, 0.0))

    def test_parallel_2d(self):
        self.run_parallel_nd(2, np.array([1.1, 0.7]))

    def test_parallel_1d(self):
        self.run_parallel_nd(1, np.array([1.1]))

    def test_antiparallel_2d(self):
        self.run_antiparallel_nd(2, np.array([1.1, 0.7]))

    def test_antiparallel_1d(self):
        self.run_antiparallel_nd(1, np.array([1.1]))

    def test_perp_2d(self):
        self.run_perp_nd(2, np.array([1.1, 0.7]))


class TestRunUtils(TestBase):
    def test_model_factory_random_seeding(self):
        model_kwargs = {
            'dim': 2,
            'dt': 0.01,
            # Must have at least one length finite to test uniform points
            # function.
            'L': np.array([1.7, np.inf]),
            'n': 1000,
            'v_0': 1.5,
            'p_0': 1.3,
            # Must have origin flag False to test uniform points function.
            'origin_flag': False,
            # Must aligned flag False to test uniform directions function.
            'aligned_flag': False,
            'chi': 0.1,
            'tumble_chemo_flag': True,
            'D_rot_0': 1.3,
            'D_rot_chemo_flag': True,
        }

        num_iterations = 1000
        rng_seed = 1

        rng = np.random.RandomState(rng_seed)
        np.random.seed(2)
        model_1 = run_utils.model_factory(rng=rng, **model_kwargs)
        for _ in range(num_iterations):
            model_1.iterate()

        rng = np.random.RandomState(rng_seed)
        np.random.seed(3)
        model_2 = run_utils.model_factory(rng=rng, **model_kwargs)
        for _ in range(num_iterations):
            model_2.iterate()

        self.assertTrue(np.all(model_1.agents.positions.r ==
                               model_2.agents.positions.r))
        self.assertTrue(np.all(model_1.agents.directions.u() ==
                               model_2.agents.directions.u()))

    def test_uniform_directions_isotropy(self):
        n = 1e5
        dim = 1
        u_0 = run_utils.get_uniform_directions(n, dim, self.rng)
        u_net = np.mean(u_0, axis=0)
        u_net_mag = np.sqrt(np.sum(np.square(u_net)))
        self.assertTrue(u_net_mag < 1e-2)

    def test_uniform_directions_magnitude(self):
        n = 1e5
        dim = 1
        u_0 = run_utils.get_uniform_directions(n, dim, self.rng)
        u_mags = np.sum(np.square(u_0), axis=-1)
        self.assertTrue(np.allclose(u_mags, 1.0))

if __name__ == '__main__':
    unittest.main()
