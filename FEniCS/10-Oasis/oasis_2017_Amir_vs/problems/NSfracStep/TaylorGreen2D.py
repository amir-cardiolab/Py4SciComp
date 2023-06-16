from __future__ import print_function
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *


# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, **NS_namespace):
    NS_parameters.update(
        nu=0.01,
        T=1.,
        dt=0.001,
        Nx=20, Ny=20,
        folder="taylorgreen2D_results",
        plot_interval=1000,
        save_step=10000,
        checkpoint=10000,
        print_intermediate_info=1000,
        compute_error=1,
        use_krylov_solvers=True,
        velocity_degree=1,
        pressure_degree=1,
        krylov_report=False)

    NS_parameters['krylov_solvers'] = {'monitor_convergence': False,
                                       'report': False,
                                       'relative_tolerance': 1e-12,
                                       'absolute_tolerance': 1e-12}
    NS_expressions.update(dict(
        constrained_domain=PeriodicDomain(),
        initial_fields=dict(
            u0='-sin(pi*x[1])*cos(pi*x[0])*exp(-2.*pi*pi*nu*t)',
            u1='sin(pi*x[0])*cos(pi*x[1])*exp(-2.*pi*pi*nu*t)',
            p='-(cos(2*pi*x[0])+cos(2*pi*x[1]))*exp(-4.*pi*pi*nu*t)/4.'),
        dpdx=('sin(2*pi*x[0])*2*pi*exp(-4.*pi*pi*nu*t)/4.',
              'sin(2*pi*x[1])*2*pi*exp(-4.*pi*pi*nu*t)/4.'),
        total_error=zeros(3)))


def mesh(Nx, Ny, **params):
    return RectangleMesh(Point(0, 0), Point(2, 2), Nx, Ny)


class PeriodicDomain(SubDomain):

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 2) and (2, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], 2)) or
                          (near(x[0], 2) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], 2) and near(x[1], 2):
            y[0] = x[0] - 2.0
            y[1] = x[1] - 2.0
        elif near(x[0], 2):
            y[0] = x[0] - 2.0
            y[1] = x[1]
        else:
            y[0] = x[0]
            y[1] = x[1] - 2.0


def initialize(q_, q_1, q_2, VV, t, nu, dt, initial_fields, **NS_namespace):
    """Initialize solution.

    Use t=dt/2 for pressure since pressure is computed in between timesteps.

    """
    for ui in q_:
        if 'IPCS' in NS_parameters['solver']:
            deltat = dt / 2. if ui is 'p' else 0.
        else:
            deltat = 0.
        vv = interpolate(Expression((initial_fields[ui]),
                                     element=VV[ui].ufl_element(),
                                     t=t + deltat, nu=nu), VV[ui])
        q_[ui].vector()[:] = vv.vector()[:]
        if not ui == 'p':
            q_1[ui].vector()[:] = vv.vector()[:]
            deltat = -dt
            vv = interpolate(Expression((initial_fields[ui]),
                                        element=VV[ui].ufl_element(),
                                        t=t + deltat, nu=nu), VV[ui])
            q_2[ui].vector()[:] = vv.vector()[:]
    q_1['p'].vector()[:] = q_['p'].vector()[:]


def temporal_hook(q_, t, nu, VV, dt, plot_interval, initial_fields, tstep, sys_comp,
                  compute_error, total_error, **NS_namespace):
    """Function called at end of timestep.

    Plot solution and compute error by comparing to analytical solution.
    Remember pressure is computed in between timesteps.

    """
    if tstep % plot_interval == 0:
        plot(q_['u0'], title='u')
        plot(q_['u1'], title='v')
        plot(q_['p'], title='p')
        interactive()

    if tstep % compute_error == 0:
        err = {}
        for i, ui in enumerate(sys_comp):
            if 'IPCS' in NS_parameters['solver']:
                deltat_ = dt / 2. if ui is 'p' else 0.
            else:
                deltat_ = 0.
            ue = Expression((initial_fields[ui]),
                            element=VV[ui].ufl_element(),
                            t=t - deltat_, nu=nu)
            ue = interpolate(ue, VV[ui])
            uen = norm(ue.vector())
            ue.vector().axpy(-1, q_[ui].vector())
            error = norm(ue.vector()) / uen
            err[ui] = "{0:2.6e}".format(norm(ue.vector()) / uen)
            total_error[i] += error * dt
        if MPI.rank(mpi_comm_world()) == 0:
            print("Error is ", err, " at time = ", t)


def theend_hook(mesh, q_, t, dt, nu, VV, sys_comp, total_error, initial_fields, **NS_namespace):
    final_error = zeros(len(sys_comp))
    for i, ui in enumerate(sys_comp):
        if 'IPCS' in NS_parameters['solver']:
            deltat = dt / 2. if ui is 'p' else 0.
        else:
            deltat = 0.
        ue = Expression((initial_fields[ui]),
                        element=VV[ui].ufl_element(),
                        t=t - deltat, nu=nu)
        ue = interpolate(ue, VV[ui])
        final_error[i] = errornorm(q_[ui], ue)

    hmin = mesh.hmin()
    if MPI.rank(mpi_comm_world()) == 0:
        print("hmin = {}".format(hmin))
    s0 = "Total Error:"
    s1 = "Final Error:"
    for i, ui in enumerate(sys_comp):
        s0 += " {0:}={1:2.6e}".format(ui, total_error[i])
        s1 += " {0:}={1:2.6e}".format(ui, final_error[i])

    if MPI.rank(mpi_comm_world()) == 0:
        print(s0)
        print(s1)
