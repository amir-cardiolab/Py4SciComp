__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2014-04-09"
__copyright__ = "Copyright (C) 2014 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from oasis.problems import *

# Default parameters NSCoupled solver
NS_parameters.update(
    # Solver parameters
    omega=1.0,           # Underrelaxation factor

    # Some discretization options
    solver="default",    # "default", "naive"

    # Parameters used to tweek solver
    max_iter=10,         # Maximum number of iterations
    max_error=1e-8,      # Tolerance for absolute error
    print_velocity_pressure_convergence=False,

    # Parameter set when enabling test mode
    testing=False,

    # Parameters used to tweek output
    plot_interval=10,
    output_timeseries_as_vector=True,  # Store velocity as vector in Timeseries
)


def NS_hook(**NS_namespace):
    pass


def start_iter_hook(**NS_namespace):
    pass

def end_iter_hook(**NS_namespace):
    pass
