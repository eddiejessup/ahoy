import numpy as np
from ahoy import mesh
import test


class TestMesh(test.TestBase):

    def test_single_sphere_random_seeding(self):
        L = np.array([2.0, 2.0])
        R = 0.1
        dx = 0.1

        rng_seed = 1

        np.random.seed(2)
        rng = np.random.RandomState(rng_seed)
        mesh_1 = mesh.single_sphere_mesh_factory(np.zeros_like(L), R, dx, L)

        np.random.seed(3)
        rng = np.random.RandomState(rng_seed)
        mesh_2 = mesh.single_sphere_mesh_factory(np.zeros_like(L), R, dx, L)

        print(mesh_1.cellCenters)
        print(mesh_2.cellCenters)

        self.assertTrue(np.allclose(mesh_1.cellCenters.value,
                                    mesh_2.cellCenters.value))
