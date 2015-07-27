import numpy as np
import fipy
from fipy.meshes import gmshMesh
import gmshpy

# gmshpy.GmshSetOption('Mesh', 'ChacoSeed', 1.0)


def get_rectangle_loop(m, L, dx):
    p_nw = m.addVertex(-L[0] / 2, L[1] / 2, 0.0, dx)
    p_ne = m.addVertex(L[0] / 2, L[1] / 2, 0.0, dx)
    p_se = m.addVertex(L[0] / 2, -L[1] / 2, 0.0, dx)
    p_sw = m.addVertex(-L[0] / 2, -L[1] / 2, 0.0, dx)
    l_nw_ne = m.addLine(p_nw, p_ne)
    l_ne_se = m.addLine(p_ne, p_se)
    l_se_sw = m.addLine(p_se, p_sw)
    l_sw_nw = m.addLine(p_sw, p_nw)

    line_loop = gmshpy.GEdgeVector()
    line_loop.append(l_nw_ne)
    line_loop.append(l_ne_se)
    line_loop.append(l_se_sw)
    line_loop.append(l_sw_nw)
    return line_loop


def get_circle_loop(m, r, R, dx):
    p_w = m.addVertex(r[0] - R, r[1], 0, dx)
    p_n = m.addVertex(r[0], r[1] + R, 0, dx)
    p_e = m.addVertex(r[0] + R, r[1], 0, dx)
    p_s = m.addVertex(r[0], r[1] - R, 0, dx)

    c_w_n = m.addCircleArcCenter(r[0], r[1], 0.0, p_w, p_n)
    c_n_e = m.addCircleArcCenter(r[0], r[1], 0.0, p_n, p_e)
    c_e_s = m.addCircleArcCenter(r[0], r[1], 0.0, p_e, p_s)
    c_s_w = m.addCircleArcCenter(r[0], r[1], 0.0, p_s, p_w)

    line_loop = gmshpy.GEdgeVector()
    line_loop.append(c_w_n)
    line_loop.append(c_n_e)
    line_loop.append(c_e_s)
    line_loop.append(c_s_w)
    return line_loop


def single_sphere_mesh_factory(r, R, dx, L):
    m = gmshpy.GModel()
    outer_loop = get_rectangle_loop(m, L, dx)
    circle_loop = get_circle_loop(m, r, R, dx)
    m.addPlanarFace([outer_loop, circle_loop])
    return gmodel_to_fipy_mesh(m)


def porous_gmodel_factory(rs, R, dx, L):
    m = gmshpy.GModel()
    outer_loop = get_rectangle_loop(m, L, dx)
    edges = [outer_loop]
    for r in rs:
        circle_loop = get_circle_loop(m, r, R, dx)
        edges.append(circle_loop)
    m.addPlanarFace(edges)
    return m


def porous_mesh_factory(rs, R, dx, L):
    m = porous_gmodel_factory(rs, R, dx, L)
    return gmodel_to_fipy_mesh(m)


def uniform_mesh_factory(L, dx):
    dim = len(L)
    if dim == 1:
        return fipy.Grid1D(dx=dx[0], Lx=L[0],
                           origin=(-L[0] / 2.0,))
    elif dim == 2:
        return fipy.Grid2D(dx=dx[0], dy=dx[1], Lx=L[0], Ly=L[1],
                           origin=((-L[0] / 2.0,), (-L[1] / 2.0,)))


def gmodel_to_fipy_mesh(m, temp_fname='temp.geo'):
    m.writeGEO(temp_fname)
    return gmshMesh.Gmsh2D(temp_fname)


if __name__ == '__main__':
    r = np.array([0.0, 0.0])
    R = 0.1
    dx = 0.05
    L = np.array([1.0, 1.0])
    m = single_sphere_mesh_factory(r, R, dx, L)

    phi = fipy.CellVariable(m)
    v = fipy.Viewer(vars=phi, xmin=-L[0] / 2.0, xmax=L[0] / 2.0)
    v.plotMesh()
    raw_input()
