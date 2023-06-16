__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
#from oasis.solvers import *
from solvers import *

"""Define all functions required by fractional step solver."""
__all__ = ["assemble_first_inner_iter", "velocity_tentative_assemble",
           "velocity_tentative_solve", "pressure_assemble",
           "pressure_solve", "velocity_update", "scalar_assemble",
           "scalar_solve", "get_solvers", "setup",
           "print_velocity_pressure_info"]


def get_solvers(**NS_namespace):
    """Return 4 linear solvers.

    We are solving for
       - tentative velocity
       - pressure correction
       - velocity update (unless lumping is switched on)

       and possibly:
       - scalars

    """
    return (None, ) * 3

def assemble_first_inner_iter(**NS_namespace):
    """Called first thing on a new velocity/pressure iteration."""
    pass


def velocity_tentative_solve(**NS_namespace):
    """Linear algebra solve of tentative velocity component."""
    pass


def velocity_tentative_assemble(**NS_namespace):
    """Assemble remaining system for tentative velocity component."""
    pass

def pressure_assemble(**NS_namespace):
    """Assemble rhs of pressure equation."""
    pass


def pressure_solve(**NS_namespace):
    """Solve pressure equation."""
    pass


def velocity_update(**NS_namespace):
    """Update the velocity after finishing pressure velocity iterations."""
    pass

def print_velocity_pressure_info(num_iter, print_velocity_pressure_convergence, norm,
                                 info_blue, inner_iter, udiff, dp_, **NS_namespace):
    if num_iter > 1 and print_velocity_pressure_convergence:
        if inner_iter == 1:
            info_blue('  Inner iterations velocity pressure:')
            info_blue('                 error u  error p')
        info_blue('    Iter = {0:4d}, {1:2.2e} {2:2.2e}'.format(
            inner_iter, udiff[0], norm(dp_.vector())))
