from __future__ import print_function, division
import numpy as np
from ciabatta import vector
from ciabatta.run_utils import run_model
from ciabatta.parallel import run_func
import ships
from ships.rudder_controllers import RudderControllers, chemo_rud_conts_factory
import utils


class _TaskRunner(object):
    """Replacement for a closure, which I would use if
    the multiprocessing module supported them.

    Imagine `__init__` is the captured outside state,
    and `__call__` is the closure body.
    """

    def __init__(self, ModelClass, model_kwargs,
                 output_every, t_upto, force_resume=True):
        self.ModelClass = ModelClass
        self.model_kwargs = model_kwargs.copy()
        self.output_every = output_every
        self.t_upto = t_upto
        self.force_resume = force_resume

    def __call__(self, extra_model_kwargs):
        model_kwargs = self.model_kwargs.copy()
        model_kwargs.update(extra_model_kwargs)
        m = self.ModelClass(**model_kwargs)
        r = run_model(self.output_every, m=m, force_resume=self.force_resume,
                      t_upto=self.t_upto)
        print(extra_model_kwargs, 'k: {}'.format(utils.get_k(r.model)))


def run_field_scan(ModelClass, model_kwargs, output_every, t_upto, field, vals,
                   force_resume=True, parallel=False):
    """Run many models with the same parameters but variable `field`.

    For each `val` in `vals`, a new model will be made, and run up to a time.
    The output directory is automatically generated from the model arguments.

    Parameters
    ----------
    ModelClass: type
        A class that can be instantiated into a Model object by calling
        `ModelClass(model_kwargs)`
    model_kwargs: dict
        Arguments that can instantiate a `ModelClass` object when passed
        to the `__init__` method.
    output_every: int
        see :class:`Runner`.
    t_upto: float
        Run each model until the time is equal to this
    field: str
        The name of the field to be varied, whose values are in `vals`.
    vals: array_like
        Iterable of values to use to instantiate each Model object.
    parallel: bool
        Whether or not to run the models in parallel, using the Multiprocessing
        library. If `True`, the number of concurrent tasks will be equal to
        one less than the number of available cores detected.
     """
    task_runner = _TaskRunner(ModelClass, model_kwargs, output_every, t_upto,
                              force_resume)
    extra_model_kwarg_sets = [{field: val} for val in vals]
    run_func(task_runner, extra_model_kwarg_sets, parallel)


def get_uniform_points(n, dim, L, rng=None):
    if rng is None:
        rng = np.random
    r = np.zeros([n, dim])
    for i_dim in np.where(np.isfinite(L))[0]:
        r[:, i_dim] = rng.uniform(-L[i_dim] / 2.0, L[i_dim] / 2.0, size=n)
    return r


def get_uniform_directions(n, dim, rng=None):
    return vector.sphere_pick(n=n, d=dim, rng=rng)


def get_aligned_directions(n, dim):
    u = np.zeros([n, dim])
    u[:, 0] = 1.0
    return u


def positions_factory(n, dim, L, origin_flag, rng):
    if origin_flag:
        r_0 = np.zeros([n, dim])
    else:
        r_0 = get_uniform_points(n, dim, L, rng)
    return ships.positions.Positions(r_0, L)


def directions_factory(n, dim, aligned_flag, rng):
    if aligned_flag:
        u_0 = get_aligned_directions(n, dim)
    else:
        u_0 = get_uniform_directions(n, dim, rng)
    return ships.directions.directions_factory(u_0, dim)


def rud_conts_factory(chemo_flag, onesided_flag, ruds, noise_0, v_0, chi,
                      esters):
    if chemo_flag:
        rud_conts = chemo_rud_conts_factory(onesided_flag, ruds, noise_0, v_0,
                                            chi, esters)
    else:
        rud_conts = RudderControllers(ruds, noise_0)
    return rud_conts


def rud_cont_sets_factory(dim, dt, v_0, p_0, chi, onesided_flag,
                          tumble_chemo_flag, D_rot_0, D_rot_chemo_flag, rng):
    # If no base source of noise to modulate, then no way to do chemotaxis.
    if not p_0:
        tumble_chemo_flag = False
    if not D_rot_0:
        D_rot_chemo_flag = False
    # If chi is zero, no point doing chemotaxis
    if not chi:
        tumble_chemo_flag = D_rot_chemo_flag = False
    chemo_flag = tumble_chemo_flag or D_rot_chemo_flag

    # If chemotaxis is happening, will need estimators.
    if chemo_flag:
        esters = ships.estimators.LinearSpatialCDotEstimators(v_0)

    rudder_controller_sets = []
    if p_0:
        tumble_ruds = ships.rudders.TumbleRudders(dt, rng)
        tumble_rud_conts = rud_conts_factory(tumble_chemo_flag, onesided_flag,
                                             tumble_ruds, p_0, v_0, chi,
                                             esters)
        rudder_controller_sets.append(tumble_rud_conts)
    if D_rot_0:
        rotation_ruds = ships.rudders.rotation_rudders_factory(dt, dim, rng)
        rotation_rud_conts = rud_conts_factory(D_rot_chemo_flag, onesided_flag,
                                               rotation_ruds, D_rot_0, v_0,
                                               chi, esters)
        rudder_controller_sets.append(rotation_rud_conts)
    return rudder_controller_sets


def model_factory(seed, dim, dt, L, n, v_0, p_0, origin_flag=False,
                  aligned_flag=False,
                  chi=None, onesided_flag=False, tumble_chemo_flag=False,
                  D_rot_0=None, D_rot_chemo_flag=False):
    """D_rot* parameters only relevant in dim > 1"""
    rng = np.random.RandomState(seed)
    ps = positions_factory(n, dim, L, origin_flag, rng)
    ds = directions_factory(n, dim, aligned_flag, rng)
    rud_cont_sets = rud_cont_sets_factory(dim, dt, v_0, p_0, chi,
                                          onesided_flag, tumble_chemo_flag,
                                          D_rot_0, D_rot_chemo_flag, rng)
    swims = ships.swimmers.Swimmers(dt, v_0)
    agents = ships.agents.Agents(ps, ds, rud_cont_sets, swims)
    model = ships.model.Model(dt, agents)
    return model
