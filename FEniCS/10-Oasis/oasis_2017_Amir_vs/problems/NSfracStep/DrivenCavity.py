__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
from ..DrivenCavity import *

#set_log_active(False)

# Override some problem specific parameters
def problem_parameters(NS_parameters, scalar_components, Schmidt, **NS_namespace):
    NS_parameters.update(
        nu=0.001,
        T=1.0,
        dt=0.001,
        folder="drivencavity_results",
        plot_interval=20,
        save_step=10000,
        checkpoint=10000,
        print_intermediate_info=100,
        use_krylov_solvers=True)

    scalar_components += ["alfa", "beta"]
    Schmidt["alfa"] = 1.
    Schmidt["beta"] = 10.

    #NS_parameters['krylov_solvers'] = {'monitor_convergence': False,
    #                                   'report': False,
    #                                   'relative_tolerance': 1e-10,
    #                                   'absolute_tolerance': 1e-10}


# Specify boundary conditions
def create_bcs(V, **NS_namespace):
    bc0 = DirichletBC(V, 0, noslip)
    bc00 = DirichletBC(V, 1, top)
    bc01 = DirichletBC(V, 0, top)
    return dict(u0=[bc00, bc0],
                u1=[bc01, bc0],
                p=[],
                alfa=[bc00],
                beta=[DirichletBC(V, 1, bottom)])


def initialize(x_1, x_2, bcs, **NS_namespace):
    for ui in x_1:
        [bc.apply(x_1[ui]) for bc in bcs[ui]]
    for ui in x_2:
        [bc.apply(x_2[ui]) for bc in bcs[ui]]


def pre_solve_hook(mesh, velocity_degree, **NS_namespace):
    Vv = VectorFunctionSpace(mesh, 'CG', velocity_degree)
    return dict(uv=Function(Vv))


def temporal_hook(q_, tstep, u_, uv, p_, plot_interval, testing, **NS_namespace):
    #if tstep % plot_interval == 0 and not testing:
    if(0):#changed by Amir
        assign(uv.sub(0), u_[0])
        assign(uv.sub(1), u_[1])
        plot(uv, title='Velocity')
        plot(p_, title='Pressure')
        plot(q_['alfa'], title='alfa')
        plot(q_['beta'], title='beta')


def theend_hook(u_, p_, uv, mesh, testing, **NS_namespace):
    if not testing:
        assign(uv.sub(0), u_[0])
        assign(uv.sub(1), u_[1])
        plot(uv, title='Velocity')
        plot(p_, title='Pressure')

    u_norm = norm(u_[0].vector())
    if MPI.rank(mpi_comm_world()) == 0 and testing:
        print("Velocity norm = {0:2.6e}".format(u_norm))

    if not testing:
        try:
            from fenicstools import StreamFunction
            psi = StreamFunction(uv, [], mesh, use_strong_bc=True)
            plot(psi, title='Streamfunction', interactive=True)
        except:
            pass
