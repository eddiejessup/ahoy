from __future__ import print_function, division
import numpy as np
import fipy
from fipy.terms import TransientTerm, DiffusionTerm, ImplicitSourceTerm


class Field(object):
    def __init__(self, dim, mesh, c_0=0.0):
        self.dim = dim
        self.mesh = mesh
        self.c_0 = c_0
        self.c = fipy.CellVariable(mesh=self.mesh, value=self.c_0)

    def _ps_to_rs(self, ps):
        return (ps.r + ps.L / 2.0).T

    def _get_nearest_cell_ids(self, rs):
        return self.mesh._getNearestCellID(rs)

    def get_nearest_cell_ids(self, ps):
        return self._get_nearest_cell_ids(self._ps_to_rs(ps))

    def _get_grad_i(self, rs):
        grad = self.c.grad
        grad_i = np.empty_like(rs)
        near_cell_ids = self._get_nearest_cell_ids(rs)
        for i in range(rs.shape[0]):
            grad_i[i] = grad[near_cell_ids[i]]
        return grad_i

    def _get_val_i(self, rs):
        val = self.c
        val_i = np.empty(rs.shape[0])
        near_cell_ids = self._get_nearest_cell_ids(rs)
        for i in range(rs.shape[0]):
            val_i[i] = val[near_cell_ids[i]]
        return val_i

    def get_grad_i(self, ps):
        return self._get_grad_i(self._ps_to_rs(ps))

    def get_val_i(self, ps):
        return self._get_val_i(self._ps_to_rs(ps))


class FoodField(Field):

    def __init__(self, dim, mesh, dt, D, delta, c_0=1.0):
        super(FoodField, self).__init__(dim, mesh, c_0)
        self.dt = dt
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
            rho_array[i] += self.rho.mesh.cellVolumes[i]
        return rho_array

    def iterate(self, ps):
        rho_array = self._get_rho_array(ps)
        self.rho.setValue(rho_array)
        self.eq.solve(dt=self.dt)


