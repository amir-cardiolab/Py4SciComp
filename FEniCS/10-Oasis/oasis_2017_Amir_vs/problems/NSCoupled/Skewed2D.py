__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSCoupled import *
from ..Skewed2D import *

# Override some problem specific parameters
def problem_parameters(NS_parameters, **NS_namespace):
    NS_parameters.update(
        nu=0.1,
        omega=1.0,
        plot_interval=10,
        max_iter=100,
        max_error=1e-12)


def create_bcs(VQ, mesh, **NS_namespace):
    u_inlet = Expression(("10*x[1]*(0.2-x[1])", "0"), element=VQ.sub(0).ufl_element())
    bc0 = DirichletBC(VQ.sub(0), (0, 0), walls)
    bc1 = DirichletBC(VQ.sub(0), u_inlet, inlet)
    return dict(up=[bc1, bc0])


def theend_hook(u_, p_, **NS_namespace):
    plot(u_, title='Velocity')
    plot(p_, title='Pressure')
