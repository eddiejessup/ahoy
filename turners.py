from __future__ import print_function, division
import numpy as np
from ciabatta import vector, crandom
from ciabatta.vector import smallest_signed_angle as angle_dist
from ahoy.utils.meta import make_repr_str


class Turner(object):

    def get_angle(self, th_in, th_normal):
        return th_in

    def get_norm_angle(self, th_in, th_normal):
        return vector.normalise_angle(self.get_angle(th_in, th_normal))

    def turn(self, obs, ds, th_normals):
        ds.th[obs] = self.get_norm_angle(ds.th[obs], th_normals)

    def __repr__(self):
        fs = []
        return make_repr_str(self, fs)


class BounceBackTurner(Turner):

    def get_angle(self, th_in, th_normal):
        return th_in + np.pi


class ReflectTurner(Turner):

    def get_angle(self, th_in, th_normal):
        th_rel = th_in - th_normal
        return th_normal + np.where(th_rel == 0.0,
                                    np.pi,
                                    np.sign(th_rel) * np.pi - th_rel)


class AlignTurner(Turner):

    def __init__(self, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

    def get_angle(self, th_in, th_normal):
        th_rel = vector.normalise_angle(th_in - th_normal)
        antiparallels = np.isclose(np.abs(angle_dist(th_in, th_normal)), np.pi)
        signs = np.where(antiparallels,
                         crandom.randbool(antiparallels.shape[0], self.rng),
                         np.sign(th_rel))
        th_rel = signs * (np.pi / 2.0)
        return th_normal + th_rel
