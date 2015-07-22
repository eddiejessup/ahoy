from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import numpy as np
from ships.ring_buffer import CylinderBuffer


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


class TimeMeasurer(Measurer):

    def __init__(self, time):
        self.time = time

    def get_time(self):
        return self.time


class PositionMeasurer(Measurer):

    def __init__(self, positions):
        self.positions = positions

    def get_positions(self):
        return self.positions


class DirectionMeasurer(Measurer):

    def __init__(self, directions):
        self.directions = directions

    def get_directions(self):
        return self.directions


class CMeasurer(Measurer):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_cs(self):
        return


class LinearCMeasurer(CMeasurer):

    def __init__(self, position_measurer):
        self.position_measurer = position_measurer

    def get_cs(self):
        ps = self.position_measurer.get_positions()
        return ps.r[:, 0]


class GradCMeasurer(Measurer):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_grad_cs(self):
        return


class ConstantGradCMeasurer(GradCMeasurer):

    def __init__(self, n, dim):
        self.grad_c = np.zeros([n, dim])
        self.grad_c[:, 0] = 1.0

    def get_grad_cs(self):
        return self.grad_c


class DcDxMeasurer(Measurer):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_dc_dxs(self):
        return


class SpatialDcDxMeasurer(DcDxMeasurer):

    def __init__(self, direction_measurer, grad_c_measurer):
        self.direction_measurer = direction_measurer
        self.grad_c_measurer = grad_c_measurer

    def get_dc_dxs(self):
        ds = self.direction_measurer.get_directions()
        grad_c = self.grad_c_measurer.get_grad_cs()
        return np.sum(ds.u() * grad_c, axis=-1)

    def __repr__(self):
        return 'SpatialDcDxMeasurer()'


class TemporalDcDxMeasurer(DcDxMeasurer):

    def __init__(self, c_measurer, v_0, dt_mem, t_mem, t_rot_0,
                 time_measurer):
        self.c_measurer = c_measurer
        self.v_0 = v_0
        self.dt_mem = dt_mem
        self.t_mem = t_mem
        n = self.c_measurer.get_cs().shape[0]
        self.K_dt = get_K(self.t_mem, self.dt_mem, t_rot_0) * self.dt_mem
        self.c_mem = CylinderBuffer(n, self.K_dt.shape[0])

        self.time_measurer = time_measurer

        # Optimisation, only calculate dc_dx when c memory is updated.
        self.dc_dx_cache = np.zeros([n])
        self.t_last_update = 0.0

    def _iterate(self):
        self.c_mem.update(self.c_measurer.get_cs())

    def _get_dc_dxs(self):
        return self.c_mem.integral_transform(self.K_dt) / self.v_0

    def iterate(self):
        t_now = self.time_measurer.get_time().t
        if t_now - self.t_last_update > 0.99 * self.dt_mem:
            self._iterate()
            self.dc_dx_cache = self._get_dc_dxs()
            self.t_last_update = t_now

    def get_dc_dxs(self):
        self.iterate()
        return self.dc_dx_cache

    def __repr__(self):
        return 'TemporalDcDxMeasurer(dtmem={:g},tmem={:g})'.format(self.dt_mem,
                                                                   self.t_mem)


class NoiseMeasurer(Measurer):

    def __init__(self, noise_0, *args, **kwargs):
        self.noise_0 = noise_0

    def get_noise(self):
        return self.noise_0

    def __repr__(self):
        return 'Noise(n0={:g})'.format(self.noise_0)


class ChemoNoiseMeasurer(NoiseMeasurer):

    def __init__(self, noise_0, chi, dc_dx_measurer):
        NoiseMeasurer.__init__(self, noise_0)
        self.chi = chi
        self.dc_dx_measurer = dc_dx_measurer

    def get_noise(self):
        dc_dxs = self.dc_dx_measurer.get_dc_dxs()
        return self.noise_0 * (1.0 - self.chi * dc_dxs)

    def __repr__(self):
        return 'CNoise2(n0={:g},chi={:g},meas={})'.format(self.noise_0,
                                                          self.chi,
                                                          self.dc_dx_measurer)


class OneSidedChemoNoiseMeasurer(ChemoNoiseMeasurer):

    def get_noise(self):
        noise_two_sided = super(OneSidedChemoNoiseMeasurer, self).get_noise()
        return np.minimum(self.noise_0, noise_two_sided)

    def __repr__(self):
        return 'CNoise1(n0={:g},chi={:g},meas={})'.format(self.noise_0,
                                                          self.chi,
                                                          self.dc_dx_measurer)


def chemo_noise_measurer_factory(onesided_flag, *args, **kwargs):
    if onesided_flag:
        return ChemoNoiseMeasurer(*args, **kwargs)
    else:
        return OneSidedChemoNoiseMeasurer(*args, **kwargs)


def noise_measurer_factory(chemo_flag, onesided_flag, *args, **kwargs):
    if chemo_flag:
        return chemo_noise_measurer_factory(onesided_flag, *args, **kwargs)
    else:
        return NoiseMeasurer(*args, **kwargs)
