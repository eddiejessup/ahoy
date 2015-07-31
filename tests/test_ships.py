import numpy as np
from ahoy import ships, turners
import test


class TestShips(test.TestBase):
    def test_ships_random_seeding(self):
        ships_kwargs = {
            'seed': 1,
            'dim': 2,
            'dt': 0.01,
            'n': 10,
            # Must have aligned flag False to test uniform directions function.
            'aligned_flag': False,
            'chi': 0.1,
            'onesided_flag': True,
            # Must have tumbling to test tumbling function.
            'p_0': 1.3,
            'tumble_chemo_flag': True,
            # Must have rotational diffusion to test rot diff function.
            'Dr_0': 1.3,
            'rotation_chemo_flag': True,
        }

        num_iterations = 100
        rng_seed = 1

        def get_ships(npy_seed):
            np.random.seed(npy_seed)
            shps = ships.ships_factory(**ships_kwargs)
            for _ in range(num_iterations):
                shps.iterate()
            return shps

        ships_1 = get_ships(2)
        ships_2 = get_ships(3)

        self.assertTrue(np.allclose(ships_1.agents.directions.u(),
                                    ships_2.agents.directions.u()))

    def test_spatial_ships_random_seeding(self):
        ships_kwargs = {
            'seed': 1,
            'dim': 2,
            'dt': 0.01,
            'n': 10,
            # Must have aligned flag False to test uniform directions function.
            'aligned_flag': False,
            'v_0': 1.5,
            # Must have at least one periodic axis to test uniform points
            # function.
            'L': np.array([2.0, 2.2]),
            'origin_flags': np.array([False, False]),

            'pore_turner': turners.AlignTurner(),
            'pore_R': 0.1,
            'pore_pf': 0.1,

            'chi': 0.1,
            'onesided_flag': True,
            # Must have tumbling to test tumbling function.
            'p_0': 1.3,
            'tumble_chemo_flag': True,
            # Must have rotational diffusion to test rot diff function.
            'Dr_0': 1.3,
            'rotation_chemo_flag': True,
            'temporal_chemo_flag': True,
            'dt_mem': 0.05,
            't_mem': 5.0,
        }

        num_iterations = 100

        def get_ships(npy_seed):
            np.random.seed(npy_seed)
            shps = ships.spatial_ships_factory(**ships_kwargs)
            for _ in range(num_iterations):
                shps.iterate()
            return shps

        ships_1 = get_ships(2)
        ships_2 = get_ships(3)

        self.assertTrue(np.allclose(ships_1.agents.positions.r,
                                    ships_2.agents.positions.r))
        self.assertTrue(np.allclose(ships_1.agents.directions.u(),
                                    ships_2.agents.directions.u()))

    def test_c_field_ships_random_seeding(self):
        ships_kwargs = {
            'seed': 1,
            'dim': 2,
            'dt': 0.01,
            'rho_0': 10.0,
            # Must have aligned flag False to test uniform directions function.
            'aligned_flag': False,
            'v_0': 1.5,
            'L': np.array([2.0, 2.2]),

            'c_dx': 0.2,
            'c_D': 10.0,
            'c_delta': 1000.0,
            'c_0': 1.3,

            'origin_flags': np.array([False, False]),

            'pore_turner': turners.AlignTurner(),
            'pore_R': 0.2,
            'pore_pf': 0.1,

            'chi': 0.3,
            'onesided_flag': True,
            # Must have tumbling to test tumbling function.
            'p_0': 1.3,
            'tumble_chemo_flag': True,
            # Must have rotational diffusion to test rot diff function.
            'Dr_0': 1.3,
            'rotation_chemo_flag': True,

            'temporal_chemo_flag': True,
            'dt_mem': 0.1,
            't_mem': 5.0
        }

        num_iterations = 100

        def get_ships(npy_seed):
            np.random.seed(npy_seed)
            shps = ships.c_field_ships_factory(**ships_kwargs)
            for _ in range(num_iterations):
                shps.iterate()
            return shps

        ships_1 = get_ships(2)
        ships_2 = get_ships(3)

        self.assertTrue(np.allclose(ships_1.agents.positions.r,
                                    ships_2.agents.positions.r))
        self.assertTrue(np.allclose(ships_1.agents.directions.u(),
                                    ships_2.agents.directions.u()))
