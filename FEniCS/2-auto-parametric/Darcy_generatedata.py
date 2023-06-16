"""
This program solves steady-state Darcy's flow on a
square plate with spatially varying permeability.
        (K/mu)*u + grad(p) = 0
                  div(u) = 0
Which, in weak form reads:
 (v, (K/mu)*u) - (div(v), p) = 0           for all v
               (q, div(u)) = 0             for all q
"""


from dolfin import *
from fenics import *
#from mshr import *
from mshr import *
import numpy as np
import sys
import math
import numpy as np
from ufl import nabla_div


output_dir = './'




set_log_level(30)

domain = Rectangle(Point(0, 0), Point(1.0, 1.0))    
mesh = generate_mesh(domain, 150) #generate_mesh(domain, 100)
V = FunctionSpace(mesh, 'P', 1)  #1st order FEM


xdmffile_k = XDMFFile('results/porous_Ks.xdmf')
xdmffile_k.parameters['rewrite_function_mesh'] = False
xdmffile_k.parameters['flush_output'] = True


pi = math.pi


N_param = 15 


A_vals= np.linspace(0.,1.,N_param)
B_vals = np.linspace(0.,4.,N_param)

if(1): #Save all Ks 
 n = 0
 kprojected = Function(V)
 for i in range(N_param):
  for j in range(N_param):
      A = A_vals[i]
      B = B_vals[j]
      k = Expression('exp(-4*A*x[0]) * abs( sin(2.*PI*x[0]) ) * abs( cos(2.*PI*B*x[1]) )  + 1.0  ', PI = pi, A=A,B=B, degree = 2)  #Note int(sin(2*pi/T)  from 0-T/2 = T/pi )
      n = n + 1
      if n % 20 == 0:
        if MPI.rank(MPI.comm_world) == 0: 
            print('n= ', n)
      if(1): #save k
        kprojected = project(k, V, solver_type='bicgstab', preconditioner_type='default' )
        kprojected.rename("kprojected", "Kperm")
        xdmffile_k.write(kprojected, n)
#exit()



mu = 10

inflow  = 'near(x[0], 0)'
outflow = 'near(x[0], 1)'
walls   = 'near(x[1], 0) || near(x[1], 1)'




basis_order = 2
Element1 =  VectorElement("CG", mesh.ufl_cell(), basis_order)
Element2 =  FiniteElement("CG", mesh.ufl_cell(), 1)
W_elem = MixedElement([Element1, Element2])
W = FunctionSpace(mesh, W_elem)


(v,q) = TestFunctions(W)

w = TrialFunction(W)
(u, p) = split(w)





# Define variational problem
# (u, p) = TrialFunctions(V)
# (v, q) = TestFunctions(V)
bcu_walls = DirichletBC(W.sub(0).sub(1),Constant(0.0), walls) #free slip(partial(u)/partial(y) =0 $ v =0)
bcp_outflow = DirichletBC(W.sub(1), Constant(0.0), outflow) # outlet pressure
bcp_inflow = DirichletBC(W.sub(1), Constant(1.0), inflow) # inlet pressure
# bcu_inflow = DirichletBC(W.sub(0), Constant((1.0,0.0)), inflow) # inlet pressure
bc = [bcu_walls,bcp_inflow, bcp_outflow]
# bc_p = [bcp_inflow, bcp_outflow]

f = Constant(0.0)


k = Expression('exp(-4*A*x[0]) * abs( sin(2.*PI*x[0]) ) * abs( cos(2.*PI*B*x[1]) )  + 1.0  ', PI = pi, A=0,B=0, degree = 2)  #Note int(sin(2*pi/T)  from 0-T/2 = T/pi )
#k = Expression('sin(x[1]) + 0.5 + B ', B=0, degree = 2 ) #testing
F1 = dot((mu/k)*v, u)*dx + inner(grad(p), v)*dx + q*div(u)*dx #divide by k causes issue!!!!
F2 = q*f*dx #+ dot(g, v)*dx



w_sol = Function(W)

xdmffile_u = XDMFFile('results/velocity_Kall.xdmf')
xdmffile_u.parameters['rewrite_function_mesh'] = False
xdmffile_u.parameters['flush_output'] = True
xdmffile_P = XDMFFile('results/pressure_Kall.xdmf')
xdmffile_P.parameters['rewrite_function_mesh'] = False
xdmffile_P.parameters['flush_output'] = True

problem = LinearVariationalProblem(F1, F2, w_sol,bc)
solver = LinearVariationalSolver(problem)
solver.parameters["linear_solver"] = "lu" #"gmres" #"bicgstab"
solver.parameters["preconditioner"] ="default"
solver.parameters['krylov_solver']['nonzero_initial_guess'] = False #True  


n=0
for i in range(N_param):
  for j in range(N_param):
      Av = A_vals[i]
      Bv = B_vals[j]
      k.A = Av
      k.B = Bv
      #w = Function(W)
      n = n + 1
      if n % 1 == 0:
        if MPI.rank(MPI.comm_world) == 0: 
            print('Solving weak form n= ', n,flush=True)
      #problem_primal = solve ( F1 == F2, w, bc)
      solver.solve()
      u_sol , p_sol = w_sol.split()
      u_sol.rename("u_sol", "Vel")
      p_sol.rename("p_sol", "P")
      xdmffile_u.write(u_sol, n)
      xdmffile_P.write(p_sol, n)
      if MPI.rank(MPI.comm_world) == 0: 
            print('Written ',flush=True)


