from __future__ import print_function, division
from itertools import product
import numpy as np
from scipy.stats import multivariate_normal
from ahoy import positions, field
from ahoy.mesh import uniform_mesh_factory
import test


class TestMesh(test.TestBase):

    def test_uniform_mesh_1d(self):
        L = np.array([1.8])
        dx = np.array([0.1])

        mesh = uniform_mesh_factory(L, dx)
        self.assertTrue(mesh.cellCenters.value.max() < L[0] / 2.0)
        self.assertTrue(mesh.cellCenters.value.min() > -L[0] / 2.0)

    def test_uniform_mesh_2d(self):
        dim = 2
        L = np.array([1.7, 3.0])
        dx = np.array([0.1, 0.2])

        mesh = uniform_mesh_factory(L, dx)
        for i_dim in range(dim):
            self.assertTrue(mesh.cellCenters[i_dim, :].value.max() <
                            L[i_dim] / 2.0)
            self.assertTrue(mesh.cellCenters[i_dim, :].value.min()
                            > -L[i_dim] / 2.0)


def get_nearest_cell_ids_manual(f, ps):
    rs = ps.r_w().T
    ccs = f.mesh.cellCenters.value
    drs = np.abs(ccs[:, np.newaxis, :] - rs[:, :, np.newaxis])
    for i_dim in range(ps.dim):
        drs[i_dim] = np.minimum(drs[i_dim], ps.L[i_dim] - drs[i_dim])
    dr_mags = np.sum(np.square(drs), axis=0)
    return np.argmin(dr_mags, axis=1)


class TestField(test.TestBase):

    def test_field_1d(self):
        L = np.array([1.6])
        dim = L.shape[0]
        dx = np.array([0.1])

        mesh = uniform_mesh_factory(L, dx)
        f = field.Field(dim, mesh)

        x_vals = [-L[0] / 2.012, -0.312, 0.01, 0.121, L[0] / 1.976]
        rs_special = np.array(list(product(x_vals)))
        rs_random = positions.get_uniform_points(1, dim, L, rng=self.rng)
        rs = np.append(rs_random, rs_special, axis=0)
        ps = positions.PeriodicPositions(L, rs)
        cids = f.get_nearest_cell_ids(ps)
        cids_manual = get_nearest_cell_ids_manual(f, ps)
        self.assertTrue(np.allclose(cids, cids_manual))

    def test_field_2d(self):
        dim = 2
        L = np.array([1.6, 3.0])
        dx = np.array([0.1, 0.2])

        mesh = uniform_mesh_factory(L, dx)
        f = field.Field(dim, mesh)

        x_vals = [-L[0] / 2.012, -0.312, 0.01, 0.121, L[0] / 1.976]
        y_vals = [-L[1] / 1.99632, -0.312, 0.01, 0.121, L[1] / 2.0021]
        rs_special = np.array(list(product(x_vals, y_vals)))
        rs = np.append(self.rng.uniform(-0.5, 0.5, size=(100, dim)),
                       rs_special, axis=0)
        ps = positions.PeriodicPositions(L, rs)
        cids = f.get_nearest_cell_ids(ps)
        cids_manual = get_nearest_cell_ids_manual(f, ps)
        self.assertTrue(np.allclose(cids, cids_manual))


class TestFoodField(TestField):

    def do_rho_array_uniform(self, L, dx):
        dim = len(L)
        rho_expected = np.product(dx)

        mesh = uniform_mesh_factory(L, dx)
        r_centers = mesh.cellCenters.value.T
        ps_centers = positions.PeriodicPositions(L, r_centers)

        dt = 1.0
        D = 1.0
        delta = 1.0
        c_0 = 1.0
        f = field.FoodField(dim, mesh, dt, D, delta, c_0)
        rho_array = f._get_rho_array(ps_centers)
        self.assertTrue(np.allclose(rho_array, rho_expected))

    def test_get_rho_array_1d(self):
        L = np.array([1.6])
        dx = np.array([0.1])
        self.do_rho_array_uniform(L, dx)

    def test_get_rho_array_2d(self):
        L = np.array([1.3, 3.1])
        dx = np.array([0.1, 0.05])
        self.do_rho_array_uniform(L, dx)

    def do_decay_term(self, L, dx):
        dim = len(L)
        rho_expected = np.product(dx)

        mesh = uniform_mesh_factory(L, dx)
        r_centers = mesh.cellCenters.value.T
        ps_centers = positions.PeriodicPositions(L, r_centers)

        dt = 0.01
        D = 0.0
        delta = 1.0
        c_0 = 1.0
        f = field.FoodField(dim, mesh, dt, D, delta, c_0)
        f.iterate(ps_centers)
        c_expected = c_0 * np.exp(-delta * rho_expected * dt)
        self.assertTrue(np.allclose(f.c, c_expected))

    def test_decay_term_2d(self):
        L = np.array([1.3, 3.1])
        dx = np.array([0.1, 0.05])
        self.do_decay_term(L, dx)

    def test_decay_term_1d(self):
        L = np.array([1.2])
        dx = np.array([0.3])
        self.do_decay_term(L, dx)

    def do_diff_term(self, L, dx):
        dim = len(L)

        dt = 0.0025
        t_max = 0.2
        D = 1.0
        delta = 0.0
        c_0 = 1.0

        mesh = uniform_mesh_factory(L, dx)
        r_centers = mesh.cellCenters.value.T
        r_centers_mag = np.sqrt(np.sum(np.square(r_centers), axis=-1))
        i_center = np.argmin(r_centers_mag)
        c_0_array = np.zeros(mesh.cellCenters.shape[1])
        c_0_array[i_center] = c_0
        ps = positions.PeriodicPositions(L, np.zeros((1, dim)))

        f = field.FoodField(dim, mesh, dt, D, delta, c_0_array)
        for t in np.arange(0.0, t_max, dt):
            f.iterate(ps)
        variance = 2.0 * D * t_max
        mean = dim * [0.0]
        cov = np.identity(dim) * variance
        mn = multivariate_normal(mean, cov)
        c_expected = (c_0 * np.product(dx)) * mn.pdf(r_centers)
        # print(c_expected)
        # print(f.c.value)
        # import matplotlib.pyplot as plt
        # plt.plot(r_centers_mag, f.c.value, label='Got')
        # plt.plot(r_centers_mag, c_expected, label='Expected')
        # plt.legend()
        # plt.show()
        self.assertTrue(np.allclose(f.c, c_expected, atol=1e-4))

    def test_diff_term_1d(self):
        L = np.array([4.0])
        dx = np.array([0.005])
        self.do_diff_term(L, dx)

    def test_diff_term_2d(self):
        L = np.array(2 * [4.0])
        dx = np.array(2 * [0.05])
        self.do_diff_term(L, dx)
