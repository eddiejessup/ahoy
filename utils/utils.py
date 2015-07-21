from __future__ import print_function, division
import numpy as np
from ciabatta import vector
from ciabatta.runner_utils import get_filenames, filename_to_model


def get_vd_coeff(x, t):
    return x / t


def get_diff_coeff(x, t):
    return x ** 2 / (2.0 * t)


def get_vd_vector(m):
    dr = m.agents.positions.get_unwrapped_dr()
    return np.mean(get_vd_coeff(dr, m.t), axis=0)


def get_vd_scalar(m):
    dr = m.agents.positions.get_unwrapped_dr_mag()
    return np.mean(get_vd_coeff(dr, m.t), axis=0)


def get_D_vector(m):
    dr = m.agents.positions.r
    return np.mean(get_diff_coeff(dr, m.t), axis=0)


def get_D_scalar(m):
    dr = m.agents.positions.get_unwrapped_dr_mag()
    return np.mean(get_diff_coeff(dr, m.t), axis=0)


def get_r_vector(m):
    dr = m.agents.positions.r
    return np.mean(dr, axis=0)


def get_r_scalar(m):
    dr = m.agents.positions.get_unwrapped_dr_mag()
    return np.mean(dr, axis=0)


def get_v_net_vector(m):
    v = m.agents.swimmers.v_0 * m.agents.directions.u()
    return np.mean(v, axis=0)


def get_v_net_scalar(m):
    return vector.vector_mag(get_v_net_vector(m))


def _t_measures(dirname, measure_func):
    ts, measures = [], []
    for fname in get_filenames(dirname):
        m = filename_to_model(fname)
        ts.append(m.t)
        measures.append(measure_func(m))
    return np.array(ts), np.array(measures)


def t_vds_vector(dirname):
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
    vds: numpy.ndarray[dtype=float]
         Particle drift speeds.
    """
    return _t_measures(dirname, get_vd_vector)


def t_vds_scalar(dirname):
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
    vds: numpy.ndarray[dtype=float]
         Particle drift speeds.
    """
    return _t_measures(dirname, get_vd_scalar)


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


def t_v_nets_scalar(dirname):
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
    return _t_measures(dirname, get_v_net_scalar)


def t_v_nets_vector(dirname):
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
    return _t_measures(dirname, get_v_net_vector)
