from __future__ import print_function, division
import numpy as np
import fipy
from fipy.terms import TransientTerm, DiffusionTerm, ImplicitSourceTerm
from ahoy.utils.meta import make_repr_str


class Field(object):
    def __init__(self, mesh, c_0):
        self.mesh = mesh
        self.c_0 = c_0
        self.c = fipy.CellVariable(mesh=self.mesh, value=self.c_0)

    @property
    def dim(self):
        return self.mesh.dim

    def _ps_to_rs(self, ps):
        return ps.r_w().T

    def _get_nearest_cell_ids(self, rs):
        return self.mesh._getNearestCellID(rs)

    def get_nearest_cell_ids(self, ps):
        return self._get_nearest_cell_ids(self._ps_to_rs(ps))

    def _get_grad_i(self, rs):
        grad = self.c.grad
        grad_i = np.empty(rs.T.shape)
        near_cell_ids = self._get_nearest_cell_ids(rs)
        for i in range(rs.shape[1]):
            grad_i[i] = grad[:, near_cell_ids[i]]
        return grad_i

    def _get_val_i(self, rs):
        val = self.c
        val_i = np.empty(rs.shape[1])
        near_cell_ids = self._get_nearest_cell_ids(rs)
        for i in range(rs.shape[1]):
            val_i[i] = val[near_cell_ids[i]]
        return val_i

    def get_grad_i(self, ps):
        return self._get_grad_i(self._ps_to_rs(ps))

    def get_val_i(self, ps):
        return self._get_val_i(self._ps_to_rs(ps))

    def __repr__(self):
        fs = [('dim', self.dim), ('mesh', self.mesh), ('c_0', self.c_0)]
        return make_repr_str(self, fs)


class FoodField(Field):

    def __init__(self, mesh, D, delta, c_0):
        super(FoodField, self).__init__(mesh, c_0)
        self.D = D
        self.delta = delta

        self.rho = fipy.CellVariable(mesh=self.mesh, value=0.0)

        self.eq = (TransientTerm(var=self.c) ==
                   DiffusionTerm(coeff=self.D, var=self.c) -
                   ImplicitSourceTerm(coeff=self.delta * self.rho, var=self.c))

    def _get_rho_array(self, ps):
        rho_array = np.zeros(self.rho.shape)
        cids = self.get_nearest_cell_ids(ps)
        for i in cids:
            rho_array[i] += 1.0 / self.rho.mesh.cellVolumes[i]
        return rho_array

    def iterate(self, ps, dt):
        rho_array = self._get_rho_array(ps)
        self.rho.setValue(rho_array)
        self.eq.solve(dt=dt)

    def __repr__(self):
        fs = [('dim', self.dim), ('mesh', self.mesh), ('c_0', self.c_0),
              ('D', self.D), ('delta', self.delta)]
        return make_repr_str(self, fs)


def food_field_factory(L, c_dx, c_D, c_delta, c_0, obstructor):
    mesh = obstructor.get_mesh(L, c_dx)
    return FoodField(mesh, c_D, c_delta, c_0)
