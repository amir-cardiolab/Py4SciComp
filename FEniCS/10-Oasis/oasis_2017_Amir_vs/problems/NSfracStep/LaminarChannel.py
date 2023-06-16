from __future__ import print_function
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
from numpy import pi, arctan, array
set_log_active(False)


# Override some problem specific parameters
def problem_parameters(NS_parameters, NS_expressions, **NS_namespace):
    nu = 0.01
    Re = 1. / nu
    L = 10.
    NS_parameters.update(dict(
        nu=nu,
        L=L,
        H=1.,
        T=10,
        dt=0.01,
        Re=Re,
        Nx=40,
        Ny=40,
        folder="laminarchannel_results",
        max_iter=1,
        velocity_degree=1,
        use_krylov_solvers=False))

    NS_expressions.update(dict(constrained_domain=PeriodicDomain(L)))


# Create a mesh here
def mesh(Nx, Ny, L, H, **params):
    m = RectangleMesh(Point(0., -H), Point(L, H), Nx, Ny)

    # Squeeze towards walls
    x = m.coordinates()
    x[:, 1] = arctan(1. * pi * (x[:, 1])) / arctan(1. * pi)
    return m


class PeriodicDomain(SubDomain):
    def __init__(self, L):
        self.L = L
        SubDomain.__init__(self)

    def inside(self, x, on_boundary):
        return bool(near(x[0], 0) and on_boundary)

    def map(self, x, y):
        y[0] = x[0] - self.L
        y[1] = x[1]


def create_bcs(V, H, sys_comp, **NS_namespace):
    def walls(x, on_boundary):
        return (on_boundary and (near(x[1], -H) or near(x[1], H)))

    bcs = dict((ui, []) for ui in sys_comp)
    bc0 = DirichletBC(V, 0., walls)
    bcs['u0'] = [bc0]
    bcs['u1'] = [bc0]
    return bcs


def body_force(Re, **NS_namespace):
    return Constant((2. / Re, 0.))


def reference(Re, t, num_terms=100):
    u = 1.0
    c = 1.0
    for n in range(1, 2 * num_terms, 2):
        a = 32. / (pi**3 * n**3)
        b = (0.25 / Re) * pi**2 * n**2
        c = -c
        u += a * exp(-b * t) * c
    return u


def temporal_hook(tstep, q_, t, Re, L, **NS_namespace):
    if tstep % 20 == 0:
        plot(q_['u0'])
    try:
        # point is found on one processor, the others pass
        u_computed = q_['u0'](array([L, 0.]))
        u_exact = reference(Re, t)
        print("Error = ", (u_exact - u_computed) / u_exact)
    except:
        pass
