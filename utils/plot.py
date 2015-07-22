from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ciabatta import runner_utils
import utils


def plot_2d(dirname):
    fig = plt.figure()
    ax_vis = fig.add_subplot(111)

    fnames = runner_utils.get_filenames(dirname)
    m_0 = runner_utils.filename_to_model(fnames[0])

    L = m_0.agents.positions.L

    ax_vis.set_xlim(-L[0] / 2.0, L[0] / 2.0)
    ax_vis.set_ylim(-L[1] / 2.0, L[1] / 2.0)
    ax_vis.set_aspect('equal')

    plt.subplots_adjust(left=0.25, bottom=0.25)
    plot_p = ax_vis.quiver(m_0.agents.positions.r[:, 0],
                           m_0.agents.positions.r[:, 1],
                           m_0.agents.directions.u()[:, 0],
                           m_0.agents.directions.u()[:, 1])

    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    t_slider = Slider(ax_slide, 'Time', 0, len(fnames), valinit=0)

    t_time = fig.text(0.1, 0.5, '')

    def update(val):
        fname_i = int(round(val))
        if 0 <= fname_i < len(fnames):
            m = runner_utils.filename_to_model(fnames[fname_i])
            plot_p.set_offsets(m.agents.positions.r)
            plot_p.set_UVC(m.agents.directions.u()[:, 0],
                           m.agents.directions.u()[:, 1])
            t_time.set_text('Time: {:g}'.format(m.time.t))
            fig.canvas.draw_idle()

    t_slider.on_changed(update)

    plt.show()


def plot_1d(dirname):
    fig = plt.figure()
    ax_vis = fig.add_subplot(211)
    ax_d = fig.add_subplot(212)

    fnames = runner_utils.get_filenames(dirname)

    m_0 = runner_utils.filename_to_model(fnames[0])

    L = m_0.agents.positions.L
    dx = L / 100.0

    ax_vis.set_xlim(-L[0] / 2.0, L[0] / 2.0)
    ax_d.set_xlim(-L[0] / 2.0, L[0] / 2.0)

    plt.subplots_adjust(left=0.25, bottom=0.25)
    plot_p = ax_vis.scatter(m_0.agents.positions.r[:, 0],
                            np.zeros([m_0.agents.n]))

    d = m_0.agents.positions.get_density_field(dx)
    x = np.linspace(-L[0] / 2.0, L[0] / 2.0, d.shape[0])

    plot_d = ax_d.bar(x, d, width=x[1] - x[0])

    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    t_slider = Slider(ax_slide, 'Index', 0, len(fnames), valinit=0)

    def update(val):
        fname_i = int(round(val))
        if 0 <= fname_i < len(fnames):
            m = runner_utils.filename_to_model(fnames[fname_i])
            plot_p.set_offsets(np.array([m.agents.positions.r[:, 0],
                                         np.zeros([m.agents.n])]).T)
            ds = m.agents.positions.get_density_field(dx) / m.agent_density
            for rect, d in zip(plot_d, ds):
                rect.set_height(d)
            ax_d.set_ylim(0.0, 1.05 * ds.max())
            fig.canvas.draw_idle()

    t_slider.on_changed(update)

    plt.show()


def plot_vis(dirname):
    dim = runner_utils.get_recent_model(dirname).dim
    if dim == 1:
        plot_1d(dirname)
    elif dim == 2:
        plot_2d(dirname)


def plot_t_uds_scalar(dirname):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ts, uds = utils.t_uds_scalar(dirname)
    ax.plot(ts, uds)

    plt.show()


def plot_t_uds_vector(dirname):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ts, uds = utils.t_uds_vector(dirname)
    for vd_set in uds.T:
        ax.plot(ts, vd_set)

    plt.show()


def plot_t_Ds_scalar(dirname):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ts, Ds = utils.t_Ds_scalar(dirname)
    ax.plot(ts, Ds)

    plt.show()


def plot_t_Ds_vector(dirname):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ts, Ds = utils.t_Ds_vector(dirname)
    for D_set in Ds.T:
        ax.plot(ts, D_set)

    plt.show()


def plot_t_rs_scalar(dirname):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ts, rs = utils.t_rs_scalar(dirname)
    ax.plot(ts, rs)

    plt.show()


def plot_t_rs_vector(dirname):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ts, rs = utils.t_rs_vector(dirname)
    for r_set in rs.T:
        ax.plot(ts, r_set)

    plt.show()


def plot_t_u_nets_scalar(dirname):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ts, u_nets = utils.t_u_nets_scalar(dirname)
    ax.plot(ts, u_nets)

    plt.show()


def plot_t_u_nets_vector(dirname):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ts, u_nets = utils.t_u_nets_vector(dirname)
    for u_net_set in u_nets.T:
        ax.plot(ts, u_net_set)

    plt.show()


def plot_chi_uds_scalar(dirnames):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    chis, uds = utils.chi_uds_scalar(dirnames)
    i_sort = np.argsort(chis)
    chis, uds = chis[i_sort], uds[i_sort]
    ax.plot(chis, uds)
    ax.set_ylim(0.0, 1.1)

    plt.show()
