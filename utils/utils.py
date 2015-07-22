from __future__ import print_function, division
import numpy as np
from ciabatta import vector
from agaro.output_utils import get_filenames, filename_to_model
from agaro.measure_utils import measures


def get_vd_coeff(x, t):
    return x / t


def get_diff_coeff(x, t):
    return x ** 2 / (2.0 * t)


def get_ud_vector(m):
    dr = m.agents.positions.dr()
    return np.mean(get_vd_coeff(dr, m.t), axis=0) / m.agents.swimmers.v_0


def get_ud_scalar(m):
    dr = m.agents.positions.dr_mag()
    return np.mean(get_vd_coeff(dr, m.t), axis=0) / m.agents.swimmers.v_0


def get_D_vector(m):
    dr = m.agents.positions.dr()
    return np.mean(get_diff_coeff(dr, m.t), axis=0)


def get_D_scalar(m):
    dr = m.agents.positions.dr_mag()
    return np.mean(get_diff_coeff(dr, m.t), axis=0)


def get_r_vector(m):
    dr = m.agents.positions.dr()
    return np.mean(dr, axis=0)


def get_r_scalar(m):
    dr = m.agents.positions.dr_mag()
    return np.mean(dr, axis=0)


def get_u_net_vector(m):
    return np.mean(m.agents.directions.u(), axis=0)


def get_u_net_scalar(m):
    return vector.vector_mag(get_u_net_vector(m))


def get_chi(m):
    return m.agents.get_chi()


def _t_measures(dirname, measure_func):
    ts, measures = [], []
    for fname in get_filenames(dirname):
        m = filename_to_model(fname)
        ts.append(m.t)
        measures.append(measure_func(m))
    return np.array(ts), np.array(measures)


def t_uds_vector(dirname):
    """Calculate the particle drift speed over time along each axis
    for a model output directory.

    Parameters
    ----------
    dirname: str
        A model output directory path

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    uds: numpy.ndarray[dtype=float]
         Drift speeds, normalised by the swimmer speed.
    """
    return _t_measures(dirname, get_ud_vector)


def t_uds_scalar(dirname):
    """Calculate the overall particle drift speed over time
    for a model output directory.

    Parameters
    ----------
    dirname: str
        A model output directory path

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    uds: numpy.ndarray[dtype=float]
         Particle drift speeds.
    """
    return _t_measures(dirname, get_ud_scalar)


def t_Ds_scalar(dirname):
    """Calculate the overall particle diffusion constant over time
    for a model output directory.

    Parameters
    ----------
    dirname: str
        A model output directory path

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    Ds: numpy.ndarray[dtype=float]
         Particle diffusion constants.
    """
    return _t_measures(dirname, get_D_scalar)


def t_Ds_vector(dirname):
    """Calculate the particle diffusion constant over time along each axis
    for a model output directory.

    Parameters
    ----------
    dirname: str
        A model output directory path

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    Ds: numpy.ndarray[dtype=float]
         Particle diffusion constants.
    """
    return _t_measures(dirname, get_D_vector)


def t_rs_scalar(dirname):
    """Calculate the overall particle displacement over time
    for a model output directory.

    Parameters
    ----------
    dirname: str
        A model output directory path

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    rs: numpy.ndarray[dtype=float]
         Particle diffusion constants.
    """
    return _t_measures(dirname, get_r_scalar)


def t_rs_vector(dirname):
    """Calculate the particle displacement over time along each axis
    for a model output directory.

    Parameters
    ----------
    dirname: str
        A model output directory path

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    rs: numpy.ndarray[dtype=float]
         Particle diffusion constants.
    """
    return _t_measures(dirname, get_r_vector)


def t_u_nets_scalar(dirname):
    """Calculate the particles' overall centre-of-mass speed over time
    for a model output directory.

    Parameters
    ----------
    dirname: str
        A model output directory path

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    v_nets: numpy.ndarray[dtype=float]
         Centre-of-mass particle speeds.
    """
    return _t_measures(dirname, get_u_net_scalar)


def t_u_nets_vector(dirname):
    """Calculate the particle's centre-of-mass velocity over time
    for a model output directory.

    Parameters
    ----------
    dirname: str
        A model output directory path

    Returns
    -------
    ts: numpy.ndarray[dtype=float]
        Times.
    v_nets: numpy.ndarray[dtype=float]
         Centre-of-mass particle velocities.
    """
    return _t_measures(dirname, get_u_net_vector)


def chi_uds_scalar(dirnames, t_steady=None):
    """Calculate the drift speed of a set of
    model output directories, and their associated chis.

    Parameters
    ----------
    dirnames: list[str]
        Model output directory paths.
    t_steady: None or float
        Time to consider the model to be at steady-state.
        The measure will be averaged over all later times.
        `None` means just consider the latest time.

    Returns
    -------
    chis: numpy.ndarray[dtype=float]
        Chemotactic sensitivities
    uds: numpy.ndarray[dtype=float]
         Drift speeds, normalised by the swimmer speed.
    """
    chis = measures(dirnames, get_chi, t_steady)
    uds = measures(dirnames, get_ud_vector, t_steady)
    return chis, uds
