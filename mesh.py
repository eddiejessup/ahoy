import fipy


def uniform_mesh_factory(dim, L, dx):
    if dim == 1:
        return fipy.Grid1D(dx=dx[0], Lx=L[0])
    elif dim == 2:
        return fipy.Grid2D(dx=dx[0], dy=dx[1], Lx=L[0], Ly=L[1])
