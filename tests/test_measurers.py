from __future__ import print_function, division
import numpy as np
from ships import stime, directions, measurers
import test


class TestLinearSpatialDcDxMeasurer(test.TestBase):

    def setUp(self):
        super(TestLinearSpatialDcDxMeasurer, self).setUp()

    def run_nd(self, dim, u_0, dc_dxs_expected):
        ds = directions.directions_factory(dim, u_0)
        grad_c_measurer = measurers.ConstantGradCMeasurer(self.n, dim)
        dc_dx_measurer = measurers.SpatialDcDxMeasurer(ds, grad_c_measurer)
        dc_dxs = dc_dx_measurer.get_dc_dxs()
        self.assertTrue(np.allclose(dc_dxs, dc_dxs_expected))

    def run_parallel_nd(self, dim):
        u_0 = np.zeros([self.n, dim])
        u_0[:, 0] = 1.0
        self.run_nd(dim, u_0, 1.0)

    def run_antiparallel_nd(self, dim):
        u_0 = np.zeros([self.n, dim])
        u_0[:, 0] = -1.0
        self.run_nd(dim, u_0, -1.0)

    def run_perp_nd(self, dim):
        u_0 = np.zeros([self.n, dim])
        u_0[:, 1] = -1.0
        self.run_nd(dim, u_0, 0.0)

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
    def __init__(self, dim, n, v_0, u_0):
        self.r = np.zeros([n, dim])
        self.v_0 = v_0
        self.u_0 = u_0

    def iterate(self, dt):
        self.r += self.v_0 * self.u_0 * dt


class TestLinearTemporalDcDxMeasurer(TestLinearSpatialDcDxMeasurer):

    def setUp(self):
        super(TestLinearTemporalDcDxMeasurer, self).setUp()
        self.v_0 = 2.2
        self.dt = 0.01
        self.dt_mem = 0.05
        self.t_mem = 5.0
        self.t_rot_0 = 1.0

    def run_nd(self, dim, u_0, dc_dxs_expected):
        time = stime.Time(self.dt)
        ps = MockPositions(dim, self.n, self.v_0, u_0)
        c_measurer = measurers.LinearCMeasurer(ps)
        dc_dx_measurer = measurers.TemporalDcDxMeasurer(c_measurer, self.v_0,
                                                        self.dt_mem,
                                                        self.t_mem,
                                                        self.t_rot_0, time)
        while time.t < 2.0 * self.t_mem:
            ps.iterate(time.dt)
            dc_dxs = dc_dx_measurer.get_dc_dxs()
            time.iterate()
        self.assertTrue(np.allclose(dc_dxs, dc_dxs_expected))


class TestChemoNoiseMeasurer1D(test.TestBase):
    dim = 1
    L = np.array([0.7])
    noise_measurer_cls = measurers.ChemoNoiseMeasurer

    def setUp(self):
        super(TestChemoNoiseMeasurer1D, self).setUp()
        self.noise_0 = 1.5
        self.v_0 = 2.2

    def run_valchemo(self, ds, chi, noise_expected):
        grad_c_measurer = measurers.ConstantGradCMeasurer(self.n, self.dim)
        dc_dx_measurer = measurers.SpatialDcDxMeasurer(ds, grad_c_measurer)
        noise_measurer = self.noise_measurer_cls(self.noise_0, chi,
                                                 dc_dx_measurer)
        noise = noise_measurer.get_noise()
        self.assertTrue(np.allclose(noise, noise_expected))

    def get_ds_parallel(self):
        u_0 = np.zeros([self.n, self.dim])
        u_0[:, 0] = 1.0
        return directions.directions_factory(self.dim, u_0)

    def get_ds_antiparallel(self):
        u_0 = np.zeros([self.n, self.dim])
        u_0[:, 0] = -1.0
        return directions.directions_factory(self.dim, u_0)

    def test_parallel_nochemo(self):
        ds = self.get_ds_parallel()
        self.run_valchemo(ds, 0.0, self.noise_0)

    def test_antiparallel_nochemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(ds, 0.0, self.noise_0)

    def test_parallel_maxchemo(self):
        ds = self.get_ds_parallel()
        self.run_valchemo(ds, 1.0, 0.0)

    def test_antiparallel_maxchemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(ds, 1.0, 2.0 * self.noise_0)

    def test_parallel_halfchemo(self):
        ds = self.get_ds_parallel()
        self.run_valchemo(ds, 0.5, self.noise_0 / 2.0)

    def test_antiparallel_halfchemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(ds, 0.5, 1.5 * self.noise_0)


class TestChemoNoiseMeasurer2D(TestChemoNoiseMeasurer1D):
    dim = 2
    L = np.array([0.7, 1.1])

    def get_ds_perp(self):
        u_0 = np.zeros([self.n, self.dim])
        u_0[:, 1] = 1.0
        return directions.directions_factory(self.dim, u_0)

    def test_perp_nochemo(self):
        ds = self.get_ds_perp()
        self.run_valchemo(ds, 0.0, self.noise_0)

    def test_perp_maxchemo(self):
        ds = self.get_ds_perp()
        self.run_valchemo(ds, 1.0, self.noise_0)

    def test_perp_halfchemo(self):
        ds = self.get_ds_perp()
        self.run_valchemo(ds, 0.5, self.noise_0)


class TestOneSidedChemoNoiseMeasurer1D(TestChemoNoiseMeasurer1D):
    noise_measurer_cls = measurers.OneSidedChemoNoiseMeasurer

    def test_antiparallel_maxchemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(ds, 1.0, self.noise_0)

    def test_antiparallel_halfchemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(ds, 0.5, self.noise_0)


class TestOneSidedChemoRudderControllers2D(TestChemoNoiseMeasurer2D):
    noise_measurer_cls = measurers.OneSidedChemoNoiseMeasurer

    def test_antiparallel_maxchemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(ds, 1.0, self.noise_0)

    def test_antiparallel_halfchemo(self):
        ds = self.get_ds_antiparallel()
        self.run_valchemo(ds, 0.5, self.noise_0)
