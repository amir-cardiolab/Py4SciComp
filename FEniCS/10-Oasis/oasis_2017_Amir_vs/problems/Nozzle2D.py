from dolfin import Mesh, AutoSubDomain, near, DOLFIN_EPS
import os

if not os.path.isfile("mesh/nozzle_2d.xml"):
    try:
        os.system("gmsh mesh/nozzle_2d.geo -2 -o mesh/nozzle_2d.msh")
        os.system("dolfin-convert mesh/nozzle_2d.msh mesh/nozzle_2d.xml")
        os.system("rm mesh/nozzle_2d.msh")
    except RuntimeError:
        raise "Gmsh is required to run this demo"

mesh = Mesh("mesh/nozzle_2d.xml")


# walls = 0
def walls(x, on_boundary):
    return on_boundary and (x[1] > 0.006 - DOLFIN_EPS or
                            (x[1] > 0.002 - DOLFIN_EPS and x[0] < 0.1 and x[0] > -0.1))


# inlet = 1
def inlet(x, on_boundary):
    return on_boundary and x[0] < -0.18269 + DOLFIN_EPS


# outlet = 2
def outlet(x, on_boundary):
    return on_boundary and x[0] > 0.32 - DOLFIN_EPS


def centerline(x, on_boundary):
    return on_boundary and x[1] < DOLFIN_EPS
