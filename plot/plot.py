from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from ciabatta.ejm_rcparams import reds_cmap
from agaro import output_utils
from ahoy.utils import utils
from ahoy.plot.var_plot import VarPlot


def plot_2d(dirname):
    fig = plt.figure()
    ax_vis = fig.add_subplot(111)

    fnames = output_utils.get_filenames(dirname)
    m_0 = output_utils.filename_to_model(fnames[0])

    L = m_0.agents.positions.L

    ax_vis.set_xlim(-L[0] / 2.0, L[0] / 2.0)
    ax_vis.set_ylim(-L[1] / 2.0, L[1] / 2.0)
    ax_vis.set_aspect('equal')

    plt.subplots_adjust(left=0.25, bottom=0.25)
    has_c_field = hasattr(m_0, 'c_field')
    if has_c_field:
        plot_c = VarPlot(m_0.c_field.c, cmap=reds_cmap, axes=ax_vis)
    plot_p = ax_vis.quiver(m_0.agents.positions.r_w()[:, 0],
                           m_0.agents.positions.r_w()[:, 1],
                           m_0.agents.directions.u()[:, 0],
                           m_0.agents.directions.u()[:, 1])

    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    t_slider = Slider(ax_slide, 'Time', 0, len(fnames), valinit=0)

    t_time = fig.text(0.1, 0.5, '')

    def update(val):
        fname_i = int(round(val))
        if 0 <= fname_i < len(fnames):
            m = output_utils.filename_to_model(fnames[fname_i])
            if has_c_field:
                plot_c.update(m.c_field.c)
            plot_p.set_offsets(m.agents.positions.r_w())
            plot_p.set_UVC(m.agents.directions.u()[:, 0],
                           m.agents.directions.u()[:, 1])
            t_time.set_text('Time: {:g}'.format(m.time.t))

            fig.canvas.draw_idle()

    t_slider.on_changed(update)

    plt.show()


def plot_linear_density(dirname):
    fig = plt.figure()
    ax_d = fig.add_subplot(211)
    ax_c = fig.add_subplot(212)

    fnames = output_utils.get_filenames(dirname)
    m_0 = output_utils.filename_to_model(fnames[0])

    L = m_0.agents.positions.L

    dx = L[0] / 100.0

    plt.subplots_adjust(left=0.25, bottom=0.25)

    ds, xbs = utils.get_linear_density(m_0, dx)

    plot_d = ax_d.bar(xbs[:-1], ds, width=xbs[1] - xbs[0])
    c_field = m_0.c_field.c
    plot_c = ax_c.scatter(c_field.mesh.cellCenters[0, :], c_field.value)
    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    t_slider = Slider(ax_slide, 'Index', 0, len(fnames), valinit=0)

    ax_d.set_xlim(-L[0] / 2.0, L[0] / 2.0)
    ax_c.set_xlim(-L[0] / 2.0, L[0] / 2.0)
    ax_c.set_ylim(0.0, m_0.c_field.c_0)

    def update(val):
        fname_i = int(round(val))
        if 0 <= fname_i < len(fnames):
            m = output_utils.filename_to_model(fnames[fname_i])
            ds, xbs = utils.get_linear_density(m, dx)
            for rect, d in zip(plot_d, ds):
                rect.set_height(d)
            c_field = m.c_field.c
            plot_c.set_offsets(np.array([c_field.mesh.cellCenters[0, :],
                                         c_field.value]).T)
            fig.canvas.draw_idle()

    t_slider.on_changed(update)

    plt.show()


def plot_1d(dirname):
    fig = plt.figure()
    ax_vis = fig.add_subplot(211)
    ax_d = fig.add_subplot(212)

    fnames = output_utils.get_filenames(dirname)
    m_0 = output_utils.filename_to_model(fnames[0])

    L = m_0.agents.positions.L

    ax_vis.set_xlim(-L[0] / 2.0, L[0] / 2.0)
    ax_d.set_xlim(-L[0] / 2.0, L[0] / 2.0)

    dx = L / 100.0

    plt.subplots_adjust(left=0.25, bottom=0.25)
    plot_p = ax_vis.scatter(m_0.agents.positions.r_w()[:, 0],
                            np.zeros([m_0.agents.n]))

    d = m_0.agents.positions.get_density_field(dx)
    x = np.linspace(-L[0] / 2.0, L[0] / 2.0, d.shape[0])

    plot_d = ax_d.bar(x, d, width=x[1] - x[0])

    ax_slide = plt.axes([0.25, 0.1, 0.65, 0.03])
    t_slider = Slider(ax_slide, 'Index', 0, len(fnames), valinit=0)

    def update(val):
        fname_i = int(round(val))
        if 0 <= fname_i < len(fnames):
            m = output_utils.filename_to_model(fnames[fname_i])
            plot_p.set_offsets(np.array([m.agents.positions.r_w()[:, 0],
                                         np.zeros([m.agents.n])]).T)
            ds = m.agents.positions.get_density_field(dx) / m.agent_density
            for rect, d in zip(plot_d, ds):
                rect.set_height(d)
            ax_d.set_ylim(0.0, 1.05 * ds.max())
            fig.canvas.draw_idle()

    t_slider.on_changed(update)

    plt.show()


def plot_vis(dirname):
    dim = output_utils.get_recent_model(dirname).dim
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


def plot_chi_uds_x(dirnames):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    chis, uds = utils.chi_uds_x(dirnames)
    i_sort = np.argsort(chis)
    chis, uds = chis[i_sort], uds[i_sort]
    ax.scatter(chis, uds)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.0, 1.1)

    plt.show()


def plot_pf_Ds_scalar(dirnames):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    pfs, Ds = utils.pf_Ds_scalar(dirnames)
    i_sort = np.argsort(pfs)
    pfs, Ds = pfs[i_sort], Ds[i_sort]
    ax.scatter(pfs, Ds)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.0, 410.0)

    plt.show()


def plot_pf_uds_x(dirnames):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    pfs, uds = utils.pf_uds_x(dirnames)
    i_sort = np.argsort(pfs)
    pfs, uds = pfs[i_sort], uds[i_sort]
    ax.scatter(pfs, uds)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.0, 1.0)

    plt.show()


def plot_Dr_0_Ds_scalar(dirnames):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    Dr_0s, Ds = utils.Dr_0_Ds_scalar(dirnames)
    i_sort = np.argsort(Dr_0s)
    Dr_0s, Ds = Dr_0s[i_sort], Ds[i_sort]
    ax.scatter(Dr_0s, Ds)
    # ax.set_xlim(-0.02, 1.0)
    # ax.set_ylim(0.0, 410.0)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()


def plot_p_0_Ds_scalar(dirnames):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    p_0s, Ds = utils.p_0_Ds_scalar(dirnames)
    i_sort = np.argsort(p_0s)
    p_0s, Ds = p_0s[i_sort], Ds[i_sort]
    ax.scatter(p_0s, Ds)
    # ax.set_xlim(-0.02, 1.0)
    # ax.set_ylim(0.0, 410.0)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()
