__author__ = 'Mikael Mortensen <mikaem@math.uio.no>'
__date__ = '2014-04-04'
__copyright__ = 'Copyright (C) 2014 ' + __author__
__license__ = 'GNU Lesser GPL version 3 or any later version'

import importlib
from oasis.common import *

"""
This module implements a generic steady state coupled solver for the
incompressible Navier-Stokes equations. Several mixed function spaces
are supported. The spaces are chosen at run-time through the parameter
elements, that may be

    "TaylorHood" Pq continuous Lagrange elements for velocity and Pq-1 for pressure
    "CR"         Crouzeix-Raviart for velocity - discontinuous Galerkin (DG0) for pressure
    "MINI"       P1 velocity with bubble - P1 for pressure

Each new problem needs to implement a new problem module to be placed in
the problems/NSCoupled folder. From the problems module one needs to import
a mesh and a control dictionary called NS_parameters. See
problems/NSCoupled/__init__.py for all possible parameters.

"""

commandline_kwargs = parse_command_line()

default_problem = 'DrivenCavity'
#exec('from problems.NSCoupled.{} import *'.format(commandline_kwargs.get('problem', default_problem)))
problemname = commandline_kwargs.get('problem', default_problem)
try:
    problemmod = importlib.import_module('.'.join(('oasis.problems.NSCoupled', problemname)))
except ImportError:
    problemmod = importlib.import_module(problemname)
except:
    raise RuntimeError(problemname+' not found')

vars().update(**vars(problemmod))

# Update problem spesific parameters
problem_parameters(**vars())

# Update current namespace with NS_parameters and commandline_kwargs ++
vars().update(post_import_problem(**vars()))

# Import chosen functionality from solvers
#exec('from solvers.NSCoupled.{} import *'.format(solver))
solver = importlib.import_module('.'.join(('oasis.solvers.NSCoupled', solver)))
vars().update({name:solver.__dict__[name] for name in solver.__all__})

# Create lists of components solved for
u_components = ['u']
sys_comp = ['up'] + scalar_components

# Get the chosen mixed elment
element = commandline_kwargs.get('element', 'TaylorHood')
vars().update(elements[element])

# TaylorHood may overload degree of elements
if element == 'TaylorHood':
    degree['u'] = commandline_kwargs.get('velocity_degree', degree['u'])
    degree['p'] = commandline_kwargs.get('pressure_degree', degree['p'])
    # Should assert that degree['p'] = degree['u']-1 ??

# Declare Elements
V = VectorElement(family['u'], mesh.ufl_cell(), degree['u'])
Q = FiniteElement(family['p'], mesh.ufl_cell(), degree['p'])

# Create Mixed space
# MINI element has bubble, add to V
if bubble:
    B = VectorElement("Bubble", mesh.ufl_cell(), mesh.geometry().dim() + 1)
    VQ = FunctionSpace(mesh, (V + B) * Q,
                       constrained_domain=constrained_domain)

else:
    VQ = FunctionSpace(mesh, V * Q, constrained_domain=constrained_domain)

# Create trial and test functions
up = TrialFunction(VQ)
u, p = split(up)
v, q = TestFunctions(VQ)

# For scalars use CG space
CG = FunctionSpace(mesh, 'CG', 1, constrained_domain=constrained_domain)
c = TrialFunction(CG)
ct = TestFunction(CG)

VV = dict(up=VQ)
VV.update(dict((ui, CG) for ui in scalar_components))

# Create dictionaries for the solutions at two timesteps
q_ = dict((ui, Function(VV[ui], name=ui)) for ui in sys_comp)
q_1 = dict((ui, Function(VV[ui], name=ui + '_1')) for ui in sys_comp)

# Short forms
up_ = q_['up']    # Solution at next iteration
up_1 = q_1['up']  # Solution at previous iteration
u_, p_ = split(up_)
u_1, p_1 = split(up_1)

# Create short forms for accessing the solution vectors
x_  = dict((ui, q_ [ui].vector()) for ui in sys_comp)     # Solution vectors
x_1 = dict((ui, q_1[ui].vector()) for ui in sys_comp)     # Solution vectors previous iteration

# Create vectors to hold rhs of equations
b = dict((ui, Vector(x_[ui])) for ui in sys_comp)

# Short form scalars
for ci in scalar_components:
    exec("{}_   = q_ ['{}']".format(ci, ci))
    exec("{}_1  = q_1['{}']".format(ci, ci))

# Boundary conditions
bcs = create_bcs(**vars())

# Initialize solution
initialize(**vars())

#  Fetch linear algebra solvers
up_sol, c_sol = get_solvers(**vars())

# Get constant body forces
f = body_force(**vars())

# Get scalar sources
fs = scalar_source(**vars())

# Preassemble and allocate
vars().update(setup(**vars()))

# Anything problem specific
vars().update(pre_solve_hook(**vars()))


def iterate(iters=max_iter):
    # Newton iterations for steady flow
    iter = 0
    error = 1

    while iter < iters and error > max_error:
        start_iter_hook(**globals())
        NS_assemble(**globals())
        NS_hook(**globals())
        NS_solve(**globals())
        end_iter_hook(**globals())

        # Update to next iteration
        for ui in sys_comp:
            x_1[ui].zero()
            x_1[ui].axpy(1.0, x_[ui])

        error = b['up'].norm('l2')
        print_velocity_pressure_info(**locals())

        iter += 1

def iterate_scalar(iters=max_iter, errors=max_error):
    # Newton iterations for scalars
    if len(scalar_components) > 0:
        err = {ci: 1 for ci in scalar_components}
        for ci in scalar_components:
            globals().update(ci=ci)
            citer = 0
            while citer < iters and err[ci] > errors:
                scalar_assemble(**globals())
                scalar_hook(**globals())
                scalar_solve(**globals())
                err[ci] = b[ci].norm('l2')
                if MPI.rank(mpi_comm_world()) == 0:
                    print('Iter {}, Error {} = {}'.format(citer, ci, err[ci]))
                citer += 1



timer = OasisTimer('Start Newton iterations flow', True)
# Assemble rhs once, before entering iterations (velocity components)
b['up'] = assemble(Fs['up'], tensor=b['up'])
for bc in bcs['up']:
    bc.apply(b['up'], x_['up'])

iterate(max_iter)
timer.stop()

# Assuming there is no feedback to the flow solver from the scalar field,
# we solve the scalar only after converging the flow
if len(scalar_components) > 0:
    scalar_timer = OasisTimer('Start Newton iterations scalars', True)
    # Assemble rhs once, before entering iterations (velocity components)
    for scalar in scalar_components:
        b[scalar] = assemble(Fs[scalar], tensor=b[scalar])
        for bc in bcs[scalar]:
            bc.apply(b[scalar], x_[scalar])

    iterate_scalar()
    scalar_timer.stop()

list_timings(TimingClear_clear, [TimingType_wall])
info_red('Total computing time = {0:f}'.format(timer.elapsed()[0]))
oasis_memory('Final memory use ')
total_initial_dolfin_memory = MPI.sum(mpi_comm_world(), initial_memory_use)
info_red('Memory use for importing dolfin = {} MB (RSS)'.format(
    total_initial_dolfin_memory))
info_red('Total memory use of solver = ' +
            str(oasis_memory.memory - total_initial_dolfin_memory) + ' MB (RSS)')

# Final hook
theend_hook(**vars())
