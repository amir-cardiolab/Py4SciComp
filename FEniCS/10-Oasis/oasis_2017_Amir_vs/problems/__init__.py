__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from dolfin import *
import subprocess
from os import getpid, path
from collections import defaultdict
from numpy import array, maximum, zeros
import six

# UnitSquareMesh(20, 20) # Just due to MPI bug on Scinet

# try:
#from fenicstools import getMemoryUsage

# except:


def getMemoryUsage(rss=True):
    mypid = str(getpid())
    rss = "rss" if rss else "vsz"
    process = subprocess.Popen(['ps', '-o', rss, mypid],
                                stdout=subprocess.PIPE)
    out, _ = process.communicate()
    mymemory = out.split()[1]
    return eval(mymemory) / 1024


parameters["linear_algebra_backend"] = "PETSc"
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs" #"quadrature" # Changed to uflacs (Amir). Nonnewtonian compilation is slow. 
parameters["form_compiler"]["quadrature_degree"] = 4
#parameters["form_compiler"]["cache_dir"] = "instant"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3"
#parameters["mesh_partitioner"] = "ParMETIS"
#parameters["form_compiler"].add("no_ferari", True)
set_log_active(False)

# Default parameters for all solvers
NS_parameters = dict(
    nu=0.01,             # Kinematic viscosity
    folder='results',    # Relative folder for storing results
    velocity_degree=2,   # default velocity degree
    pressure_degree=1    # default pressure degree
)

NS_expressions = {}

constrained_domain = None

# To solve for scalars provide a list like ['scalar1', 'scalar2']
scalar_components = []

# With diffusivities given as a Schmidt number defined by:
#   Schmidt = nu / D (= momentum diffusivity / mass diffusivity)
Schmidt = defaultdict(lambda: 1.)
Schmidt_T = defaultdict(lambda: 0.7)  # Turbulent Schmidt number (LES)

Scalar = defaultdict(lambda: dict(Schmidt=1.0,
                                  family="CG",
                                  degree=1))

# The following helper functions are available in dolfin
# They are redefined here for printing only on process 0.
RED = "\033[1;37;31m%s\033[0m"
BLUE = "\033[1;37;34m%s\033[0m"
GREEN = "\033[1;37;32m%s\033[0m"


def info_blue(s, check=True):
    if MPI.rank(mpi_comm_world()) == 0 and check:
        print(BLUE % s)


def info_green(s, check=True):
    if MPI.rank(mpi_comm_world()) == 0 and check:
        print(GREEN % s)


def info_red(s, check=True):
    if MPI.rank(mpi_comm_world()) == 0 and check:
        print(RED % s)


class OasisTimer(Timer):
    def __init__(self, task, verbose=False):
        Timer.__init__(self, task)
        info_blue(task, verbose)


class OasisMemoryUsage:
    def __init__(self, s):
        self.memory = 0
        self.memory_vm = 0
        self(s)

    def __call__(self, s, verbose=False):
        self.prev = self.memory
        self.prev_vm = self.memory_vm
        self.memory = MPI.sum(mpi_comm_world(), getMemoryUsage())
        self.memory_vm = MPI.sum(mpi_comm_world(), getMemoryUsage(False))
        if MPI.rank(mpi_comm_world()) == 0 and verbose:
            info_blue('{0:26s}  {1:10d} MB {2:10d} MB {3:10d} MB {4:10d} MB'.format(s,
                        int(self.memory - self.prev), int(self.memory),
                        int(self.memory_vm - self.prev_vm), int(self.memory_vm)))


# Print memory use up til now
initial_memory_use = getMemoryUsage()
oasis_memory = OasisMemoryUsage('Start')


# Convenience functions
def strain(u):
    return 0.5 * (grad(u) + grad(u).T)


def omega(u):
    return 0.5 * (grad(u) - grad(u).T)


def Omega(u):
    return inner(omega(u), omega(u))


def Strain(u):
    return inner(strain(u), strain(u))


def QC(u):
    return Omega(u) - Strain(u)


def recursive_update(dst, src):
    """Update dict dst with items from src deeply ("deep update")."""
    for key, val in src.items():
        if key in dst and isinstance(val, dict) and isinstance(dst[key], dict):
            dst[key] = recursive_update(dst[key], val)
        else:
            dst[key] = val
    return dst


def add_function_to_tstepfiles(function, newfolder, tstepfiles, tstep):
    name = function.name()
    tstepfolder = path.join(newfolder, "Timeseries")
    tstepfiles[name] = XDMFFile(mpi_comm_world(),
                                path.join(tstepfolder,
                                          '{}_from_tstep_{}.xdmf'.format(name, tstep)))
    tstepfiles[name].function = function
    tstepfiles[name].parameters["rewrite_function_mesh"] = False


def body_force(mesh, **NS_namespace):
    """Specify body force"""
    return Constant((0,) * mesh.geometry().dim())


def initialize(**NS_namespace):
    """Initialize solution."""
    pass


def create_bcs(sys_comp, **NS_namespace):
    """Return dictionary of Dirichlet boundary conditions."""
    return dict((ui, []) for ui in sys_comp)


def scalar_hook(**NS_namespace):
    """Called prior to scalar solve."""
    pass


def scalar_source(scalar_components, **NS_namespace):
    """Return a dictionary of scalar sources."""
    return dict((ci, Constant(0)) for ci in scalar_components)


def pre_solve_hook(**NS_namespace):
    """Called just prior to entering time-loop. Must return a dictionary."""
    return {}


def theend_hook(**NS_namespace):
    """Called at the very end."""
    pass


def problem_parameters(**NS_namespace):
    """Updates problem specific parameters, and handles restart"""
    pass


def post_import_problem(NS_parameters, mesh, commandline_kwargs,
                        NS_expressions, **NS_namespace):
    """Called after importing from problem."""

    # Update NS_parameters with all parameters modified through command line
    for key, val in six.iteritems(commandline_kwargs):
        if isinstance(val, dict):
            NS_parameters[key].update(val)
        else:
            NS_parameters[key] = val

    # If the mesh is a callable function, then create the mesh here.
    if callable(mesh):
        mesh = mesh(**NS_parameters)

    assert(isinstance(mesh, Mesh))

    # Returned dictionary to be updated in the NS namespace
    d = dict(mesh=mesh)
    d.update(NS_parameters)
    d.update(NS_expressions)
    return d
