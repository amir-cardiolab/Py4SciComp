__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
from numpy import cos, pi, cosh
from os import getcwd
import pickle


# Create a mesh
def mesh(**params):
    m = Mesh('/home/mikael/MySoftware/Oasis/mymesh/boxwithsphererefined.xml')
    return m


def problem_parameters(commandline_kwargs, NS_parameters, **NS_namespace):
    if "restart_folder" in commandline_kwargs.keys():
        restart_folder = commandline_kwargs["restart_folder"]
        restart_folder = path.join(getcwd(), restart_folder)
        f = open(path.join(restart_folder, 'params.dat'), 'r')
        NS_parameters.update(pickle.load(f))
        NS_parameters['T'] = NS_parameters['T'] + 10 * NS_parameters['dt']
        NS_parameters['restart_folder'] = restart_folder
        globals().update(NS_parameters)

    else:
        # Override some problem specific parameters
        NS_parameters.update(
            nu=0.1,
            T=5.0,
            dt=0.01,
            h=0.75,
            sol=40,
            dpdx=0.05,
            velocity_degree=2,
            plot_interval=10,
            print_intermediate_info=10,
            use_krylov_solvers=True)
        NS_parameters['krylov_solvers']['monitor_convergence'] = True


def create_bcs(V, Q, mesh, **NS_namespace):
    # Specify boundary conditions
    walls = "on_boundary && std::abs((x[1]-3)*(x[1]+3)*(x[2]-3)*(x[2]+3))<1e-8"
    inners = "on_boundary && std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]) < 1.5*{}".format(h)
    inlet = "x[0] < -3+1e-8 && on_boundary"
    outlet = "x[0] > 6-1e-8 && on_boundary"

    bmesh = BoundaryMesh(mesh, 'exterior')
    cc = CellFunction('size_t', bmesh, 0)
    ii = AutoSubDomain(lambda x, on_bnd: near(x[0], -3))
    ii.mark(cc, 1)
    smesh = SubMesh(bmesh, cc, 1)
    Vu = FunctionSpace(smesh, 'CG', 1)
    su = Function(Vu)
    us = TrialFunction(Vu)
    vs = TestFunction(Vu)
    solve(inner(grad(us), grad(vs)) * dx == Constant(0.1) * vs * dx, su,
          bcs=[DirichletBC(Vu, Constant(0), DomainBoundary())])

    lp = LagrangeInterpolator()
    sv = Function(V)
    lp.interpolate(sv, su)

    bc0 = DirichletBC(V, 0, walls)
    bc1 = DirichletBC(V, 0, inners)
    bcp1 = DirichletBC(Q, 0, outlet)
    bc2 = DirichletBC(V, 0, inlet)
    bc3 = DirichletBC(V, sv, inlet)
    return dict(u0=[bc0, bc1, bc3],
                u1=[bc0, bc1, bc2],
                u2=[bc0, bc1, bc2],
                p=[bcp1])


def pre_solve_hook(mesh, velocity_degree, u_,
                   AssignedVectorFunction, **NS_namespace):
    return dict(uv=AssignedVectorFunction(u_))


def temporal_hook(tstep, uv, p_, plot_interval, **NS_namespace):
    if tstep % plot_interval == 0:
        uv()
        plot(uv, title='Velocity')
        plot(p_, title='Pressure')


def theend_hook(p_, uv, **NS_namespace):
    uv()
    plot(uv, title='Velocity')
    plot(p_, title='Pressure')
