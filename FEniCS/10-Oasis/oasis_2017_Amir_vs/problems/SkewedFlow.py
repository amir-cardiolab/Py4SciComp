__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-10"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import BoxMesh, Point

# Create a mesh
h = 0.5
L = 1.


def mesh(N=20, **params):
    m = BoxMesh(Point(0, 0, 0), Point(L, 1, 1), N, N, N)
    return m


tol = 1e-8


# Specify boundary conditions
def inlet(x, on_bnd):
    return x[0] < tol and x[1] < h + tol and x[2] > (1 - h - tol)


def outlet(x, on_bnd):
    return x[0] > L - tol and x[1] > 1 - h - tol and x[2] < h + tol


def walls(x, on_bnd):
    return (abs(x[1] * (1 - x[1]) * x[2] * (1 - x[2])) < tol
            or ((x[0] < tol and (x[1] > h - tol or x[2] < (1 - h + tol)))
                or (x[0] > L - tol and (x[1] < 1 - h + tol or x[2] > h - tol))))
