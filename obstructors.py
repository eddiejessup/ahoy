from __future__ import print_function, division
from abc import ABCMeta, abstractmethod
import numpy as np
from ciabatta import vector, pack, distance, geom
from ahoy import mesh


class NoneObstructor(object):

    def get_obstructeds(self, rs):
        return np.zeros([rs.shape[0]], dtype=np.bool)

    def obstruct(self, ps, drs, ds):
        return

    def get_mesh(self, L, dx):
        return mesh.uniform_mesh_factory(L, dx)

    def __repr__(self):
        dct = {}
        return '{}({})'.format(self.__class__, dct)


class BaseObstructor(NoneObstructor):
    __metaclass__ = ABCMeta

    def __init__(self, turner):
        self.turner = turner

    def _push(self, obs, rs, drs):
        rs[obs] -= drs[obs]

    def get_obstructeds(self, rs):
        seps = self.get_seps(rs)
        return self._is_obstructed(seps)


class SphereObstructor(BaseObstructor):
    __metaclass__ = ABCMeta

    def __init__(self, turner, R, *args, **kwargs):
        super(SphereObstructor, self).__init__(turner)
        self.R = R

    def _is_obstructed(self, seps):
        return vector.vector_mag_sq(seps) < self.R ** 2.0

    @property
    def volume_sphere(self):
        return geom.sphere_volume(self.R, self.dim)

    @abstractmethod
    def _get_normals(self, seps):
        return

    @abstractmethod
    def get_seps(self, rs):
        return rs

    def obstruct(self, ps, drs, ds):
        seps = self.get_seps(ps.r_w())
        obs = self._is_obstructed(seps)
        self._push(obs, ps.r, drs)
        normals = self._get_normals(seps[obs])
        self.turner.turn(obs, ds, normals)

    def __repr__(self):
        dct = {'turner': self.turner, 'R': self.R}
        return '{}({})'.format(self.__class__, dct)


class SphereObstructor2D(SphereObstructor):

    def _get_normals(self, seps):
        return np.arctan2(seps[:, 1], seps[:, 0])


class SingleSphereObstructor2D(SphereObstructor2D):

    def get_seps(self, rs):
        return rs

    def get_mesh(self, L, dx):
        return mesh.single_sphere_mesh_factory(np.zeros_like(L), self.R, dx[0],
                                               L)


class PorousObstructor(SphereObstructor2D):

    def __init__(self, turner, R, L, pf, rng):
        super(PorousObstructor, self).__init__(turner, R)
        self.L = L
        self.rs, self.R = pack.pack_simple(self.R, self.L, pf=pf, rng=rng)

    @property
    def volume(self):
        return np.product(self.L)

    @property
    def dim(self):
        return self.rs.shape[1]

    @property
    def n(self):
        return self.rs.shape[0]

    @property
    def volume_occupied(self):
        return self.n * self.volume_sphere

    @property
    def fraction_occupied(self):
        return self.volume_occupied / self.volume

    def get_seps(self, rs):
        return distance.csep_periodic_close(rs, self.rs, self.L)[0]

    def get_mesh(self, L, dx):
        return mesh.porous_mesh_factory(self.rs, self.R, dx[0], L)

    def __repr__(self):
        dct = {'turner': self.turner, 'R': self.R, 'L': self.L,
               'fraction_occupied': self.fraction_occupied, 'rng': self.rng}
        return '{}({})' % (self.__class__, dct)
