from __future__ import print_function
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
from ..Skewed2D import *

# Override some problem specific parameters
def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        nu=0.1,
        T=10.0,
        dt=0.05,
        use_krylov_solvers=True,
        print_velocity_pressure_convergence=True)


def create_bcs(V, Q, mesh, **NS_namespace):
    u_inlet = Expression("10*x[1]*(0.2-x[1])", element=V.ufl_element())
    bc0 = DirichletBC(V, 0, walls)
    bc1 = DirichletBC(V, u_inlet, inlet)
    bc2 = DirichletBC(V, 0, inlet)
    return dict(u0=[bc1, bc0],
                u1=[bc2, bc0],
                p=[DirichletBC(Q, 0, outlet)])


def pre_solve_hook(mesh, u_, AssignedVectorFunction, **NS_namespace):
    return dict(uv=AssignedVectorFunction(u_, "Velocity"), n=FacetNormal(mesh))

def temporal_hook(u_, p_, mesh, tstep, print_intermediate_info,
                  uv, n, plot_interval, **NS_namespace):
    if tstep % print_intermediate_info == 0:
        print("Continuity ", assemble(dot(u_, n) * ds()))

    if tstep % plot_interval == 0:
        uv()
        plot(uv, title='Velocity')
        plot(p_, title='Pressure')


def theend_hook(uv, p_, **NS_namespace):
    uv()
    plot(uv, title='Velocity')
    plot(p_, title='Pressure')
