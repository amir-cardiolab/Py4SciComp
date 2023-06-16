__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-04"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
from oasis.solvers import *

"""Define all functions required by coupled solver."""
__all__ = ["NS_assemble", "NS_solve", "scalar_assemble",
           "scalar_solve", "get_solvers", "setup",
           "print_velocity_pressure_info",
           "elements"]

elements = {
    "TaylorHood":
        dict(family={"u": "CG", "p": "CG"},
             degree={"u": 2, "p": 1},
             bubble=False),
    "MINI":
        dict(family={"u": "CG", "p": "CG"},
             degree={"u": 1, "p": 1},
             bubble=True),
    "CR":
        dict(family={"u": "CR", "p": "DG"},
             degree={"u": 1, "p": 0},
             bubble=False)
}

def NS_assemble(**NS_namespace):
    pass

def NS_solve(**NS_namespace):
    pass

def get_solvers(**NS_namespace):
    """Return 2 linear solvers.

    We are solving for
       velocity/pressure

       and possibly:
       - scalars

    """
    up_sol, c_sol = LUSolver(), LUSolver()
    up_sol.parameters["same_nonzero_pattern"] = True
    c_sol .parameters["same_nonzero_pattern"] = True
    return up_sol, c_sol

def print_velocity_pressure_info(iter, error, **NS_namespace):
    if MPI.rank(mpi_comm_world()) == 0:
        print("Iter {}, Error = {}".format(iter + 1, error))
