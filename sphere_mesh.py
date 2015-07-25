import fipy
import numpy as np

# All arguments on the left hand side are indices to the various constructions.
gmsh_text = '''
// Define the square that acts as the system boundary.

dx = %(dx)g;
Lx = %(Lx)g;
Ly = %(Ly)g;
R = %(R)g;

// Define each corner of the square
// Arguments are (x, y, z, dx); dx is the desired cell size near that point.
Point(1) = {Lx / 2, Ly / 2, 0, dx};
Point(2) = {-Lx / 2, Ly / 2, 0, dx};
Point(3) = {-Lx / 2, -Ly / 2, 0, dx};
Point(4) = {Lx / 2, -Ly / 2, 0, dx};

// Line is a straight line between points.
// Arguments are indices of points as defined above.
Line(1) = {1, 4};
Line(2) = {4, 3};
Line(3) = {3, 2};
Line(4) = {2, 1};

// Loop is a closed loop of lines.
// Arguments are indices of lines as defined above.
Line Loop(1) = {1, 2, 3, 4};

// Circle center coordinates
x = %(x)g;
y = %(y)g;

// Define the center and compass points of the circle.
Point(5) = {x, y, 0, dx};
Point(6) = {x - R, y, 0, dx};
Point(7) = {x, y + R, 0, dx};
Point(8) = {x + R, y, 0, dx};
Point(9) = {x, y - R, 0, dx};

// Circle is confusingly actually an arc line between points.
// Arguments are indices of: starting point; center of curvature; end point.
Circle(5) = {6, 5, 7};
Circle(6) = {7, 5, 8};
Circle(7) = {8, 5, 9};
Circle(8) = {9, 5, 6};

Line Loop(2) = {5, 6, 7, 8};

// The first argument is the outer loop boundary.
// The remainder are holes in it.
Plane Surface(1) = {1, 2};
'''


def make_single_sphere_mesh(r, R, dx, L):
    args = {'dx': dx, 'Lx': L[0], 'Ly': L[1], 'R': R, 'x': r[0], 'y': r[1]}
    return fipy.Gmsh2D(gmsh_text % args)


if __name__ == '__main__':
    r = np.array([0.0, 0.0])
    R = 0.1
    dx = 0.05
    L = np.array([1.0, 1.0])
    m = make_single_sphere_mesh(r, R, dx, L)

    phi = fipy.CellVariable(m)
    v = fipy.Viewer(vars=phi, xmin=-L[0] / 2.0, xmax=L[0] / 2.0)
    v.plotMesh()
    raw_input()
