from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import numpy as np
from ahoy.ring_buffer import CylinderBuffer


def get_K(t, dt, t_rot_0):
    A = 0.5
    ts = np.arange(0.0, t, dt)
    gs = ts / t_rot_0
    K = np.exp(-gs) * (1.0 - A * (gs + (gs ** 2) / 2.0))
    K[K < 0.0] *= np.abs(K[K >= 0.0].sum() / K[K < 0.0].sum())
    K /= np.sum(K * -ts * dt)
    return K


class Measurer(object):
    __metaclass__ = ABCMeta


class CMeasurer(Measurer):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_cs(self):
        return


class FieldCMeasurer(CMeasurer):

    def __init__(self, c_field, positions):
        self.c_field = c_field
        self.positions = positions

    def get_cs(self):
        return self.c_field.get_val_i(self.positions)

    def __repr__(self):
        dct = {'c_field': self.c_field}
        return '{}({})' % (self.__class__, dct)


class LinearCMeasurer(CMeasurer):

    def __init__(self, positions):
        self.positions = positions

    def get_cs(self):
        return self.positions.r[:, 0]

    def __repr__(self):
        dct = {}
        return '{}({})' % (self.__class__, dct)


class GradCMeasurer(Measurer):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_grad_cs(self):
        return


class FieldGradCMeasurer(GradCMeasurer):

    def __init__(self, c_field, positions):
        self.c_field = c_field
        self.positions = positions

    def get_grad_cs(self):
        return self.c_field.get_grad_i(self.positions)

    def __repr__(self):
        dct = {'c_field': self.c_field}
        return '{}({})' % (self.__class__, dct)


class ConstantGradCMeasurer(GradCMeasurer):

    def __init__(self, n, dim):
        self.grad_c = np.zeros([n, dim])
        self.grad_c[:, 0] = 1.0

    @property
    def n(self):
        return self.grad_c.shape[0]

    @property
    def dim(self):
        return self.grad_c.shape[1]

    def get_grad_cs(self):
        return self.grad_c

    def __repr__(self):
        dct = {}
        return '{}({})' % (self.__class__, dct)


class DcDxMeasurer(Measurer):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_dc_dxs(self):
        return


class SpatialDcDxMeasurer(DcDxMeasurer):

    def __init__(self, directions, grad_c_measurer):
        self.directions = directions
        self.grad_c_measurer = grad_c_measurer

    def get_dc_dxs(self):
        grad_c = self.grad_c_measurer.get_grad_cs()
        return np.sum(self.directions.u() * grad_c, axis=-1)

    def __repr__(self):
        dct = {'grad_c_measurer': self.grad_c_measurer}
        return '{}({})' % (self.__class__, dct)


class TemporalDcDxMeasurer(DcDxMeasurer):

    def __init__(self, c_measurer, v_0, dt_mem, t_mem, t_rot_0,
                 time):
        self.c_measurer = c_measurer
        self.v_0 = v_0
        self.dt_mem = dt_mem
        self.t_mem = t_mem
        cs = self.c_measurer.get_cs()
        n = cs.shape[0]
        self.K_dt = get_K(self.t_mem, self.dt_mem, t_rot_0) * self.dt_mem
        self.c_mem = CylinderBuffer(n, self.K_dt.shape[0])
        self.time = time

        # Optimisation, only calculate dc_dx when c memory is updated.
        self.dc_dx_cache = np.zeros([n])
        self.t_last_update = 0.0

    def _iterate(self):
        cs = self.c_measurer.get_cs()
        self.c_mem.update(cs)

    def _get_dc_dxs(self):
        return self.c_mem.integral_transform(self.K_dt) / self.v_0

    def iterate(self):
        t_now = self.time.t
        if t_now - self.t_last_update > 0.99 * self.dt_mem:
            self._iterate()
            self.dc_dx_cache = self._get_dc_dxs()
            self.t_last_update = t_now

    def get_dc_dxs(self):
        self.iterate()
        return self.dc_dx_cache

    def __repr__(self):
        dct = {'c_measurer': self.c_measurer, 'v_0': self.v_0,
               'dt_mem': self.dt_mem, 't_mem': self.t_mem,
               't_last_update': self.t_last_update}
        return '{}({})' % (self.__class__, dct)


def dc_dx_factory(spatial_chemo_flag,
                  ds=None,
                  ps=None, v_0=None, dt_mem=None, t_mem=None, p_0=None,
                  Dr_0=None, time=None,
                  c_field=None):
    if spatial_chemo_flag:
        return spatial_dc_dx_factory(ds, c_field, ps)
    else:
        return temporal_dc_dx_factory(ps, v_0, dt_mem, t_mem, p_0, Dr_0, time,
                                      c_field)


def spatial_dc_dx_factory(ds, c_field=None, ps=None):
    if c_field is None:
        grad_c_measurer = ConstantGradCMeasurer(ds.n, ds.dim)
    else:
        grad_c_measurer = FieldGradCMeasurer(c_field, ps)
    return SpatialDcDxMeasurer(ds, grad_c_measurer)


def temporal_dc_dx_factory(ps, v_0, dt_mem, t_mem, p_0, Dr_0, time,
                           c_field=None):
    if c_field is None:
        c_measurer = LinearCMeasurer(ps)
    else:
        c_measurer = FieldCMeasurer(c_field, ps)
    D_rot_0 = p_0 + Dr_0
    t_rot_0 = 1.0 / D_rot_0
    return TemporalDcDxMeasurer(c_measurer, v_0, dt_mem, t_mem, t_rot_0, time)
