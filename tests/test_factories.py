import numpy as np
from ships.utils import factories
from ships import obstructers
import test


class TestModel(test.TestBase):
    def test_model_factory_random_seeding(self):
        model_kwargs = {
            'dim': 2,
            'dt': 0.01,
            'n': 10,
            # Must have aligned flag False to test uniform directions function.
            'aligned_flag': False,

            'spatial_flag': True,
            'v_0': 1.5,
            # Must have origin flag False to test uniform points function.
            'origin_flag': False,
            # Must have at least one periodic axis to test uniform points
            # function.
            'L': np.array([1.7, np.inf]),

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

        turner = obstructers.Turner()
        R = 0.4
        model_kwargs['obstructer'] = obstructers.SingleSphereObstructer(turner,
                                                                        R)

        num_iterations = 1000
        rng_seed = 1

        np.random.seed(2)
        model_1 = factories.model_factory(rng_seed, **model_kwargs)
        for _ in range(num_iterations):
            model_1.iterate()

        np.random.seed(3)
        model_2 = factories.model_factory(rng_seed, **model_kwargs)
        for _ in range(num_iterations):
            model_2.iterate()
        print(model_1.agents.directions.th)
        print(model_2.agents.directions.th)
        self.assertTrue(np.all(model_1.agents.positions.r ==
                               model_2.agents.positions.r))
        self.assertTrue(np.all(model_1.agents.directions.u() ==
                               model_2.agents.directions.u()))
