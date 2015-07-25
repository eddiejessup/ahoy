from __future__ import print_function, division
from itertools import product
import numpy as np
from scipy.stats import multivariate_normal
from ships import positions, field
import test


class TestMesh(test.TestBase):

    def test_uniform_mesh_1d(self):
        dim = 1
        L = np.array([1.7])
        dx = np.array([0.1])

        mesh = field.uniform_mesh_factory(dim, L, dx)
        print(mesh.cellCenters.value)
        self.assertTrue(mesh.cellCenters.value.max() < L[0])
        self.assertTrue(np.isclose(mesh.cellCenters.value.min(),
                        dx[0] / 2.0))

    def test_uniform_mesh_2d(self):
        dim = 2
        L = np.array([1.7, 3.0])
        dx = np.array([0.1, 0.2])

        mesh = field.uniform_mesh_factory(dim, L, dx)
        for i_dim in range(dim):
            self.assertTrue(mesh.cellCenters[i_dim, :].value.max() <
                            L[i_dim])
            self.assertTrue(np.isclose(mesh.cellCenters[i_dim, :].value.min(),
                            dx[i_dim] / 2.0))


def get_nearest_cell_ids_manual(f, ps):
    rs = (ps.r + ps.L / 2.0).T
    ccs = f.mesh.cellCenters.value
    drs = ccs[:, np.newaxis, :] - rs[:, :, np.newaxis]
    dr_mags = np.sum(np.square(drs), axis=0)
    cids = np.argmin(dr_mags, axis=1)
    return cids


class TestField(test.TestBase):

    def test_field_1d(self):
        dim = 1
        L = np.array([1.6])
        dx = np.array([0.1])

        mesh = field.uniform_mesh_factory(dim, L, dx)
        f = field.Field(dim, mesh)

        x_vals = [-L[0] / 2.0, -0.312, 0.01, 0.121, L[0] / 2.0]
        rs_special = np.array(list(product(x_vals)))
        rs = np.append(np.random.uniform(-0.5, 0.5, size=(100, dim)),
                       rs_special, axis=0)
        ps = positions.PeriodicPositions(L, rs)
        cids = f.get_nearest_cell_ids(ps)
        cids_manual = get_nearest_cell_ids_manual(f, ps)
        print(cids)
        print(cids_manual)
        self.assertTrue(np.allclose(cids, cids_manual))

    def test_field_2d(self):
        dim = 2
        L = np.array([1.6, 3.0])
        dx = np.array([0.1, 0.2])

        mesh = field.uniform_mesh_factory(dim, L, dx)
        f = field.Field(dim, mesh)

        x_vals = [-L[0] / 2.0, -0.312, 0.01, 0.121, L[0] / 2.0]
        y_vals = [-L[1] / 2.0, -0.312, 0.01, 0.121, L[1] / 2.0]
        rs_special = np.array(list(product(x_vals, y_vals)))
        rs = np.append(np.random.uniform(-0.5, 0.5, size=(100, dim)),
                       rs_special, axis=0)
        ps = positions.PeriodicPositions(L, rs)
        cids = f.get_nearest_cell_ids(ps)
        cids_manual = get_nearest_cell_ids_manual(f, ps)
        print(cids)
        print(cids_manual)
        self.assertTrue(np.allclose(cids, cids_manual))


class TestFoodField(TestField):

    def do_rho_array_uniform(self, L, dx):
        dim = len(L)
        rho_expected = np.product(dx)

        mesh = field.uniform_mesh_factory(dim, L, dx)
        r_centers = mesh.cellCenters.value.T - L / 2.0
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

        mesh = field.uniform_mesh_factory(dim, L, dx)
        r_centers = mesh.cellCenters.value.T - L / 2.0
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

        mesh = field.uniform_mesh_factory(dim, L, dx)
        r_centers = mesh.cellCenters.value.T - L / 2.0
        r_centers_mag = np.sqrt(np.sum(np.square(r_centers), axis=-1))
        i_center = np.argmin(r_centers_mag)
        c_0_array = np.zeros(mesh.cellCenters.shape[1])
        c_0_array[i_center] = c_0
        ps = positions.PeriodicPositions(L, np.zeros((1, 1)))

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
