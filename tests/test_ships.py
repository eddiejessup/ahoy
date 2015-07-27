import numpy as np
from ahoy.obstructors import PorousObstructor, NoneObstructor, SingleSphereObstructor2D
from ahoy import ships, turners
import test


class TestShips(test.TestBase):
    def test_ships_random_seeding(self):
        model_kwargs = {
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

        num_iterations = 1000
        rng_seed = 1

        np.random.seed(2)
        rng = np.random.RandomState(rng_seed)
        model_1 = ships.ships_factory(rng, **model_kwargs)
        for _ in range(num_iterations):
            model_1.iterate()

        np.random.seed(3)
        rng = np.random.RandomState(rng_seed)
        model_2 = ships.ships_factory(rng, **model_kwargs)
        for _ in range(num_iterations):
            model_2.iterate()
        self.assertTrue(np.all(model_1.agents.directions.u() ==
                               model_2.agents.directions.u()))

    def test_spatial_ships_random_seeding(self):
        L = np.array([2.0, 2.0])
        obstructor_kwargs = {
            'turner': turners.AlignTurner(),
            'R': 0.1,
            'pf': 0.1,
            'L': L,
        }
        model_kwargs = {
            'dim': 2,
            'dt': 0.01,
            'n': 10,
            # Must have aligned flag False to test uniform directions function.
            'aligned_flag': False,
            'v_0': 1.5,
            # Must have at least one periodic axis to test uniform points
            # function.
            'L': np.array([1.5, np.inf]),
            'origin_flags': np.array([False, False]),

            'chi': 0.1,
            'onesided_flag': True,
            # Must have tumbling to test tumbling function.
            'p_0': 1.3,
            'tumble_chemo_flag': True,
            # Must have rotational diffusion to test rot diff function.
            'Dr_0': 1.3,
            'rotation_chemo_flag': True,
            'spatial_chemo_flag': False,
            'dt_mem': 0.05,
            't_mem': 5.0,
        }

        num_iterations = 1000
        rng_seed = 1

        np.random.seed(2)
        rng = np.random.RandomState(rng_seed)
        obstructor = PorousObstructor(rng=rng, **obstructor_kwargs)
        model_1 = ships.spatial_ships_factory(rng, obstructor=obstructor,
                                              **model_kwargs)
        for _ in range(num_iterations):
            model_1.iterate()

        np.random.seed(3)
        rng = np.random.RandomState(rng_seed)
        obstructor = PorousObstructor(rng=rng, **obstructor_kwargs)
        model_2 = ships.spatial_ships_factory(rng, obstructor=obstructor,
                                              **model_kwargs)
        for _ in range(num_iterations):
            model_2.iterate()
        self.assertTrue(np.all(model_1.agents.positions.r ==
                               model_2.agents.positions.r))
        self.assertTrue(np.all(model_1.agents.directions.u() ==
                               model_2.agents.directions.u()))

    def test_c_field_ships_random_seeding(self):
        L = np.array([2.0, 2.0])
        obstructor_kwargs = {
            'turner': turners.AlignTurner(),
            'R': 0.1,
            'pf': 0.1,
            'L': L,
        }
        model_kwargs = {
            'dim': 2,
            'dt': 0.01,
            'n': 10,
            # Must have aligned flag False to test uniform directions function.
            'aligned_flag': False,
            'v_0': 1.5,
            'L': np.array([1.5, 3.0]),

            'c_dx': np.array([0.1, 0.1]),
            'c_D': 10.0,
            'c_delta': 000.0,
            'c_0': 1.3,

            'origin_flags': np.array([False, False]),

            'chi': 0.3,
            'onesided_flag': True,
            # Must have tumbling to test tumbling function.
            'p_0': 1.3,
            'tumble_chemo_flag': True,
            # Must have rotational diffusion to test rot diff function.
            'Dr_0': 1.3,
            'rotation_chemo_flag': True,

            'spatial_chemo_flag': True,
            'dt_mem': 0.05,
            't_mem': 5.0,
        }

        num_iterations = 100
        rng_seed = 1

        np.random.seed(2)
        rng = np.random.RandomState(rng_seed)
        # obstructor = NoneObstructor()
        obstructor = SingleSphereObstructor2D(rng=rng, **obstructor_kwargs)
        model_1 = ships.c_field_ships_factory(rng, obstructor=obstructor,
                                              **model_kwargs)
        for _ in range(num_iterations):
            model_1.iterate()

        np.random.seed(3)
        rng = np.random.RandomState(rng_seed)
        # obstructor = NoneObstructor()
        obstructor = SingleSphereObstructor2D(rng=rng, **obstructor_kwargs)
        model_2 = ships.c_field_ships_factory(rng, obstructor=obstructor,
                                              **model_kwargs)
        for _ in range(num_iterations):
            model_2.iterate()

        self.assertTrue(np.all(model_1.agents.positions.r ==
                               model_2.agents.positions.r))
        self.assertTrue(np.all(model_1.agents.directions.u() ==
                               model_2.agents.directions.u()))
