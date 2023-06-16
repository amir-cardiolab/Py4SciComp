from fenics import *
from mshr import *
import numpy 
from inlet_data import *


#IPCS method:
#https://github.com/hplgit/fenics-tutorial/blob/master/pub/python/vol1/ft07_navier_stokes_channel.py




output_dir = 'results/'
T = 2.85  #Three cardiac cycles
num_steps = 30000
dt = T / num_steps  # time step size
mu = 0.035          # dynamic viscosity
rho = 1.06            # density

# Create mesh
tube = Rectangle(Point(0, 0), Point(3.8, 0.4))
aneurysm = Circle(Point(2.4, 0.7), 0.5)
domain = tube + aneurysm
mesh = generate_mesh(domain, 100)



basis_order = 1 
V = VectorFunctionSpace(mesh, 'P', basis_order)
Q = FunctionSpace(mesh, 'P', 1)

# Define boundaries
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 3.8)'
walls    = 'on_boundary && (x[1]>=0.4 || near(x[1], 0) )'

# class Inflow(SubDomain):
#   def inside(self, x, on_boundary):
#     return near(x[0], 0)
# class Outflow(SubDomain):
#   def inside(self, x, on_boundary):
#     return near(x[0], 3.8)
# class Walls(SubDomain):
#   def inside(self, x, on_boundary):
#     return on_boundary and x[1] >= 0.4 or near(x[1], 0)
# 
# inflow = Inflow()
# outflow = Outflow()
# walls = Walls()
#         
# boundaries = MeshFunction('size_t', mesh, 1)
# boundaries.set_all(0)
# inflow.mark(boundaries, 1)
# outflow.mark(boundaries, 2)
# walls.mark(boundaries, 3)

# File(output_dir + 'boundaries.pvd') << boundaries         

# Define inflow profile
# inflow_profile = ('4.0*1.5*x[1]*(0.81 - x[1]) / pow(0.81, 2)', '0')

#inflow_profile = ('32', '0')

# Define boundary conditions
#bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree = 2), inflow)
bcu_walls = DirichletBC(V, Constant((0., 0.)), walls)
#bcu = [bcu_inflow, bcu_walls]

bcp_outflow = DirichletBC(Q, Constant(0.), outflow)

bcp = [bcp_outflow]

# Define trial and test functions
u = TrialFunction(V)
v = TestFunction(V)
p = TrialFunction(Q)
q = TestFunction(Q)

# Define functions for solutions at previous and current time steps
u_n = Function(V) #previous time-step
u_  = Function(V)
p_n = Function(Q) #previous time-step
p_  = Function(Q)

# Define expressions used in variational forms
U  = 0.5*(u_n + u)
n  = FacetNormal(mesh)
f  = Constant((0, 0))
k  = Constant(dt)
mu = Constant(mu)
rho = Constant(rho)

# Define symmetric gradient
def epsilon(u):
   return sym(nabla_grad(u))

# Define stress tensor
def sigma(u, p):
   return 2*mu*epsilon(u) - p*Identity(len(u))

# Define variational problem for step 1
F1 = rho*dot((u - u_n) / k, v)*dx \
  + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx \
  + inner(sigma(U, p_n), epsilon(v))*dx \
  + dot(p_n*n, v)*ds - dot(mu*nabla_grad(U)*n, v)*ds \
  - dot(f, v)*dx
a1 = lhs(F1)
L1 = rhs(F1)

# Define variational problem for step 2
a2 = dot(nabla_grad(p), nabla_grad(q))*dx
L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

# Define variational problem for step 3
a3 = dot(u, v)*dx
L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)

# Apply boundary conditions to matrices
#[bc.apply(A1) for bc in bcu]
[bc.apply(A2) for bc in bcp]

# Create XDMF files for visualization output
xdmffile_u = XDMFFile(output_dir + 'velocity.xdmf')
xdmffile_p = XDMFFile(output_dir  + 'pressure.xdmf')

# Create time series (for use in reaction_system.py)
timeseries_u = TimeSeries(output_dir + 'velocity_series')
timeseries_p = TimeSeries(output_dir  + 'pressure_series')

# Save mesh to file (for use in reaction_system.py)
#File(output_dir + 'aneurysm.xml.gz') << mesh


# Create progress bar
#progress = Progress('Looping', num_steps)
#set_log_level(LogLevel.PROGRESS)

# Time-stepping
t = 0

for n in range(num_steps):

  v_inlet_BC = numpy.interp(t, time_input, V_vel)
  bcu_inflow = DirichletBC(V, Constant((v_inlet_BC, 0.)), inflow)
  bcu = [bcu_inflow, bcu_walls]

  # Apply boundary conditions to matrices
  [bc.apply(A1) for bc in bcu]

  # Step 1: Tentative velocity step
  b1 = assemble(L1)
  [bc.apply(b1) for bc in bcu]
  solve(A1, u_.vector(), b1, 'gmres', 'default')

  # Step 2: Pressure correction step
  b2 = assemble(L2)
  [bc.apply(b2) for bc in bcp]
  solve(A2, p_.vector(), b2, 'gmres', 'default')

  # Step 3: Velocity correction step
  b3 = assemble(L3)
  solve(A3, u_.vector(), b3, 'cg', 'sor')

  # Save solution to file (XDMF/HDF5)
  if n % 500 == 0 and t > 1.9 :
    xdmffile_u.write(u_, t)
    #   xdmffile_p.write(p_, t)
      # Save nodal values to file
    #timeseries_u.store(u_.vector(), t)
      #timeseries_p.store(p_.vector(), t)

  # Update previous solution
  u_n.assign(u_)
  p_n.assign(p_)


  # Update current time
  t += dt

  #print 'u max:', u_.vector().get_local().max()
  if n % 50 == 0:
    if MPI.rank(MPI.comm_world) == 0:  
      print('u max:', u_.vector().get_local().max(),flush=True)
      print('t', t,flush=True)




