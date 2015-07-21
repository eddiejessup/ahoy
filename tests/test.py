from __future__ import print_function, division
import unittest
import numpy as np
from ships import positions, directions
from ships.utils import run_utils


class TestBase(unittest.TestCase):
    n = 5
    seed = 1

    def setUp(self):
        self.rng = np.random.RandomState(self.seed)


class TestPositions1D(TestBase):
    dim = 1
    L = np.array([1.7])

    def setUp(self):
        super(TestPositions1D, self).setUp()
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
        L = np.array([np.inf])
        r_0 = np.zeros_like(self.ps.r_0)
        ps = positions.Positions(r_0.copy(), L)
        dr = self.rng.uniform(-10.0, 10.0, size=r_0.shape)
        ps.displace(dr)
        r_inf_naive = r_0[:, 0] + dr[:, 0]
        # Check no wrapping along infinite axis
        self.assertTrue(np.allclose(r_inf_naive, ps.r[:, 0]))


class TestPositions2D(TestPositions1D):
    dim = 2
    L = np.array([0.5, 1.5])

    def test_wrapping_up(self):
        dr = np.zeros_like(self.ps.r)
        self.ps.r[-1, -1] = -self.L[-1] / 2.0
        dr[-1, -1] = -0.9 * self.L[-1]
        self.ps.displace(np.full(self.ps.r.shape, dr))
        self.assertTrue(np.abs(self.ps.r[:, -1]).max() < self.L[-1] / 2.0)

    def test_infinite_boundaries(self):
        L = np.array([np.inf, 1.7])
        r_0 = np.zeros_like(self.ps.r_0)
        ps = positions.Positions(r_0.copy(), L)
        dr = self.rng.uniform(-1.0, 1.0, size=r_0.shape)
        ps.displace(dr)
        r_inf_naive = r_0[:, 0] + dr[:, 0]
        # Check no wrapping along infinite axis
        self.assertTrue(np.allclose(r_inf_naive, ps.r[:, 0]))
        # Check done wrapping along finite axis
        self.assertTrue(np.all(np.abs(ps.r[:, 1]) < L[1] / 2.0))


class TestDirections1D(TestBase):
    dim = 1

    def setUp(self):
        super(TestDirections1D, self).setUp()
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
        n = 1000
        u_0 = run_utils.get_uniform_directions(n, self.dim, self.rng)
        ds = directions.directions_factory(u_0, self.dim)
        tumblers = np.ones([n], dtype=np.bool)
        ds_rot = ds.tumble(tumblers)
        u_rot = ds_rot.u()
        frac_close = np.isclose(u_0, u_rot).sum() / float(n)
        self.assertAlmostEqual(frac_close, 0.5, 1)


class TestDirections2D(TestDirections1D):
    dim = 2

    def test_rotate_identity(self):
        u_0 = self.ds.u()
        dth = np.zeros([u_0.shape[0]])
        ds_rot = self.ds.rotate(dth)
        u_rot = ds_rot.u()
        self.assertTrue(np.allclose(u_0, u_rot))

    def test_rotate_right_angle(self):
        u_0 = np.zeros([self.n, self.dim])
        u_0[:, 0] = 1.0
        dth = np.full([self.n], np.pi / 2.0)
        ds = directions.directions_factory(u_0, self.dim)
        ds_rot = ds.rotate(dth)
        u_rot = ds_rot.u()
        u_rot_expected = np.zeros([self.n, self.dim])
        u_rot_expected[:, 1] = 1.0
        self.assertTrue(np.allclose(u_rot, u_rot_expected))

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

        num_iterations = 100
        rng_seed = 1

        np.random.seed(2)
        model_1 = run_utils.model_factory(rng_seed, **model_kwargs)
        for _ in range(num_iterations):
            model_1.iterate()

        np.random.seed(3)
        model_2 = run_utils.model_factory(rng_seed, **model_kwargs)
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
