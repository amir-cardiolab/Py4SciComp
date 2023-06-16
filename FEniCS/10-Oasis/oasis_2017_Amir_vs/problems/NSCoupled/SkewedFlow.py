__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSCoupled import *
from ..SkewedFlow import *
# set_log_active(False)

# Override some problem specific parameters
def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        nu=0.1,
        omega=1.0,
        plot_interval=10,
        max_iter=100,
        max_error=1e-12)


def create_bcs(V, VQ, mesh, **NS_namespace):
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
                values[1] = 0
                values[2] = 0
            except:
                values[:] = 0

        def value_shape(self):
            return (3,)

    bc0 = DirichletBC(VQ.sub(0), (0, 0, 0), walls)
    bc1 = DirichletBC(VQ.sub(0), MyExp(element=VQ.sub(0).ufl_element()), inlet)
    return dict(up=[bc0, bc1])


def theend_hook(u_, p_, **NS_namespace):
    plot(u_, title='Velocity')
    plot(p_, title='Pressure')
