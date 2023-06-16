from __future__ import print_function
__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
from ..SkewedFlow import *
from numpy import cos, pi, cosh

warning("""
This problem does not work well with IPCS since the outflow
boundary condition

    grad(u)*n=0, p=0

here is a poor representation of actual physics.

Need to use coupled solver with pseudo-traction

    (grad(u)-p)*n = 0

or extrude outlet such that the outflow boundary condition
becomes more realistic.
""")

# Override some problem specific parameters
def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        nu=0.001,
        T=0.05,
        dt=0.01,
        use_krylov_solvers=True,
        print_velocity_pressure_convergence=True)


def create_bcs(V, Q, mesh, **NS_namespace):
    # Create inlet profile by solving Poisson equation on boundary
    bmesh = BoundaryMesh(mesh, 'exterior')
    cc = CellFunction('size_t', bmesh, 0)
    ii = AutoSubDomain(inlet)
    ii.mark(cc, 1)
    smesh = SubMesh(bmesh, cc, 1)
    Vu = FunctionSpace(smesh, 'CG', 1)
    su = Function(Vu)
    us = TrialFunction(Vu)
    vs = TestFunction(Vu)
    solve(inner(grad(us), grad(vs)) * dx == Constant(10.0) * vs * dx, su,
          bcs=[DirichletBC(Vu, Constant(0), DomainBoundary())])

    # Wrap the boundary function in an Expression to avoid the need to interpolate it back to V
    class MyExp(Expression):
        def eval(self, values, x):
            try:
                values[0] = su(x)
            except:
                values[0] = 0

    bc0 = DirichletBC(V, 0, walls)
    bc1 = DirichletBC(V, MyExp(element=V.ufl_element()), inlet)
    bc2 = DirichletBC(V, 0, inlet)
    return dict(u0=[bc0, bc1],
                u1=[bc0, bc2],
                u2=[bc0, bc2],
                p=[DirichletBC(Q, 0, outlet)])

def temporal_hook(u_, mesh, tstep, print_intermediate_info,
                  plot_interval, **NS_namespace):

    if tstep % print_intermediate_info == 0:
        print("Continuity ", assemble(dot(u_, FacetNormal(mesh)) * ds()))

    if tstep % plot_interval == 0:
        plot(u_, title='Velocity')
        plot(p_, title='Pressure')


def theend_hook(u_, p_, **NS_namespace):
    plot(u_, title='Velocity')
    plot(p_, title='Pressure')
