from __future__ import print_function, division
import numpy as np
import numpy.ma as ma
from ciabatta import vector, crandom
from ciabatta.vector import smallest_signed_angle as angle_dist


class Obstructer(object):

    def __init__(self, turner):
        self.turner = turner

    def get_obstructeds(self, rs):
        return np.zeros([rs.shape[0]], dtype=np.bool)

    def _push(self, obs, rs, drs):
        rs[obs] -= drs[obs]

    def _get_normals(self, rs):
        for r in rs:
            raise NotImplementedError('This should not happen.')

    def obstruct(self, ps, drs, ds):
        obs = self.get_obstructeds(ps.r_w())
        self._push(obs, ps.r, drs)
        normals = self._get_normals(ps.r_w()[obs])
        self.turner.turn(obs, ds, normals)


class SingleSphereObstructer(Obstructer):

    def __init__(self, turner, R):
        super(SingleSphereObstructer, self).__init__(turner)
        self.R = R

    def get_obstructeds(self, rs):
        obs = vector.vector_mag_sq(rs) < self.R ** 2.0
        return obs

    def _get_normals(self, rs):
        return np.arctan2(rs[:, 1], rs[:, 0])


class Turner(object):

    def get_angle(self, th_in, th_normal):
        return th_in

    def get_norm_angle(self, th_in, th_normal):
        return vector.normalise_angle(self.get_angle(th_in, th_normal))

    def turn(self, obs, ds, th_normals):
        ds.th[obs] = self.get_norm_angle(ds.th[obs], th_normals)


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
                         crandom.randbool(antiparallels.shape[0]),
                         np.sign(th_rel))
        th_rel = signs * (np.pi / 2.0)
        return th_normal + th_rel
