from __future__ import print_function, division
import numpy as np
from ciabatta import vector
import ships


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

    def get_mesh(self, L, dx):
        return ships.mesh.uniform_mesh_factory(L, dx)


class SingleSphereObstructer(Obstructer):

    def __init__(self, turner, R):
        super(SingleSphereObstructer, self).__init__(turner)
        self.R = R

    def get_obstructeds(self, rs):
        obs = vector.vector_mag_sq(rs) < self.R ** 2.0
        return obs

    def _get_normals(self, rs):
        return np.arctan2(rs[:, 1], rs[:, 0])

    def get_mesh(self, L, dx):
        dim = len(L)
        return get_single_sphere_mesh(np.array(dim * [0.0], self.R, dx[0], L))
