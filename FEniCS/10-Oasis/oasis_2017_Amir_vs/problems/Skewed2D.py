__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-10"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import Mesh
import os

if not os.path.isfile("mesh/Skewed2D.xml"):
    try:
        os.system("gmsh mesh/Skewed2D.geo -2 -o mesh/Skewed2D.msh")
        os.system("dolfin-convert mesh/Skewed2D.msh mesh/Skewed2D.xml")
        os.system("rm mesh/Skewed2D.msh")
    except RuntimeError:
        raise "Gmsh is required to run this demo"

# Create a mesh
mesh = Mesh("mesh/Skewed2D.xml")

# Specify boundary conditions
tol = 1e-8
L = 1.0


def inlet(x, on_bnd):
    return x[0] < tol and on_bnd


def outlet(x, on_bnd):
    return x[0] > L - tol and on_bnd


def walls(x, on_bnd):
    return on_bnd and (x[1] < tol or x[1] > 1 - tol or (x[1] > 0.2 - tol and x[0] < 0.5) or (x[1] < 0.8 + tol and x[0] > 0.5))
