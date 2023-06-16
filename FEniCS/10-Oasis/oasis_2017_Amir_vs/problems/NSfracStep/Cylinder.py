from __future__ import print_function
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-03-21"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
from ..Cylinder import *
from os import getcwd
import pickle

def problem_parameters(commandline_kwargs, NS_parameters, scalar_components,
                       Schmidt, **NS_namespace):
    # Example: python NSfracstep.py [...] restart_folder="results/data/8/Checkpoint"
    if "restart_folder" in commandline_kwargs.keys():
        restart_folder = commandline_kwargs["restart_folder"]
        restart_folder = path.join(getcwd(), restart_folder)
        f = open(path.join(restart_folder, 'params.dat'), 'r')
        NS_parameters.update(pickle.load(f))
        NS_parameters['restart_folder'] = restart_folder
        globals().update(NS_parameters)

    else:
        # Override some problem specific parameters
        NS_parameters.update(
            T=5.0,
            dt=0.05,
            checkpoint=1000,
            save_step=5000,
            plot_interval=10,
            velocity_degree=2,
            print_intermediate_info=100,
            use_krylov_solvers=True,
            krylov_solvers=dict(monitor_convergence=True))

    scalar_components.append("alfa")
    Schmidt["alfa"] = 0.1

def create_bcs(V, Q, Um, H, **NS_namespace):
    inlet = Expression(
        "4.*{0}*x[1]*({1}-x[1])/pow({1}, 2)".format(Um, H), degree=2)
    ux = Expression("0.00*x[1]", degree=1)
    uy = Expression("-0.00*(x[0]-{})".format(center), degree=1)
    bc00 = DirichletBC(V, inlet, Inlet)
    bc01 = DirichletBC(V, 0, Inlet)
    bc10 = DirichletBC(V, ux, Cyl)
    bc11 = DirichletBC(V, uy, Cyl)
    bc2 = DirichletBC(V, 0, Wall)
    bcp = DirichletBC(Q, 0, Outlet)
    bca = DirichletBC(V, 1, Cyl)
    return dict(u0=[bc00, bc10, bc2],
                u1=[bc01, bc11, bc2],
                p=[bcp],
                alfa=[bca])


def initialize(x_1, x_2, bcs, **NS_namespace):
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(mesh, V, newfolder, tstepfiles, tstep, ds, u_,
                   AssignedVectorFunction, **NS_namespace):
    uv = AssignedVectorFunction(u_, name='Velocity')
    omega = Function(V, name='omega')
    # Store omega each save_step
    add_function_to_tstepfiles(omega, newfolder, tstepfiles, tstep)
    ff = FacetFunction("size_t", mesh, 0)
    Cyl.mark(ff, 1)
    n = FacetNormal(mesh)
    ds = ds[ff]

    return dict(uv=uv, omega=omega, ds=ds, ff=ff, n=n)

def temporal_hook(q_, u_, tstep, V, uv, p_, plot_interval, omega, ds,
                  save_step, mesh, nu, Umean, D, n, **NS_namespace):
    if tstep % plot_interval == 0:
        uv()
        plot(uv, title='Velocity')
        plot(p_, title='Pressure')
        plot(q_['alfa'], title='alfa')

    R = VectorFunctionSpace(mesh, 'R', 0)
    c = TestFunction(R)
    tau = -p_ * Identity(2) + nu * (grad(u_) + grad(u_).T)
    forces = assemble(dot(dot(tau, n), c) * ds(1)).array() * 2 / Umean**2 / D

    print("Cd = {}, CL = {}".format(*forces))

    if tstep % save_step == 0:
        try:
            from fenicstools import StreamFunction
            omega.assign(StreamFunction(u_, []))
        except:
            omega.assign(project(curl(u_), V,
                                 bcs=[DirichletBC(V, 0, DomainBoundary())]))

def theend_hook(q_, u_, p_, uv, mesh, ds, V, nu, Umean, D, **NS_namespace):
    uv()
    plot(uv, title='Velocity')
    plot(p_, title='Pressure')
    plot(q_['alfa'], title='alfa')
    R = VectorFunctionSpace(mesh, 'R', 0)
    c = TestFunction(R)
    tau = -p_ * Identity(2) + nu * (grad(u_) + grad(u_).T)
    ff = FacetFunction("size_t", mesh, 0)
    Cyl.mark(ff, 1)
    n = FacetNormal(mesh)
    ds = ds[ff]
    forces = assemble(dot(dot(tau, n), c) * ds(1)).array() * 2 / Umean**2 / D

    print("Cd = {}, CL = {}".format(*forces))

    from fenicstools import Probes
    from numpy import linspace, repeat, where, resize
    xx = linspace(0, L, 10000)
    x = resize(repeat(xx, 2), (10000, 2))
    x[:, 1] = 0.2
    probes = Probes(x.flatten(), V)
    probes(u_[0])
    nmax = where(probes.array() < 0)[0][-1]
    print("L = ", x[nmax, 0] - 0.25)
    print("dP = ", p_(Point(0.15, 0.2)) - p_(Point(0.25, 0.2)))
