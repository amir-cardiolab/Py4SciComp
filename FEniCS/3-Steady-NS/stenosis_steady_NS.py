from fenics import *
#from mshr import *
import numpy as np
import sys

#Solves steady N.S equations





mu =  0.001  		# dynamic viscosity
rho = 1.0          	# density
nu = mu / rho
Mesh_filename = '/home/sci/amir.arzani/Python_tutorials/Fenics/NS_steady/sten_mesh_physical' #'bwd_step_mesh'



#For high Re number flow (high inlet velocity) it is difficult to converge for Newton solver. Slowly ramp up the inlet velocity to help with convergence.
Inlet_U_target = 1.5  #Target inlet velocity BC
num_simulation = 5 
inlet_U =np.linspace(0.1,Inlet_U_target,num_simulation) 



xdmffile_v = XDMFFile('results/vel_stenosis.xdmf')
xdmffile_v.parameters['rewrite_function_mesh'] = False
xdmffile_v.parameters['flush_output'] = True




if MPI.rank(MPI.comm_world) == 0: 
	print('Reading mesh..  ', Mesh_filename + '.xml')
mesh = Mesh( Mesh_filename + '.xml')




# File(output_dir + 'mesh.pvd') << mesh

# Define function spaces
basis_order = 2 
#V = VectorFunctionSpace(mesh, 'P', basis_order)
#Q = FunctionSpace(mesh, 'P', 1)

#V = VectorFunctionSpace(mesh, "CG", 2)
#Q = FunctionSpace(mesh, "CG", 1)
#W = V * Q

Element1 =  VectorElement("CG", mesh.ufl_cell(), basis_order)
Element2 =  FiniteElement("CG", mesh.ufl_cell(), 1)
W_elem = MixedElement([Element1, Element2])
W = FunctionSpace(mesh, W_elem)




inflow   = 'on_boundary && ( near(x[0], 0)  )'
outflow  = 'on_boundary && ( near(x[0], 2.0)   )'
walls    = 'on_boundary &&  not ( near(x[0], 0) || near(x[0], 2.0)  )  '


# class Inflow(SubDomain):
# 	def inside(self, x, on_boundary):
# 		return near(x[0], 0)
# class Outflow(SubDomain):
# 	def inside(self, x, on_boundary):
# 		return near(x[0], 3.8)
# class Walls(SubDomain):
# 	def inside(self, x, on_boundary):
# 		return on_boundary and x[1] >= 0.4 or near(x[1], 0)
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


#if(0):
#	inflow_profile = ('0.5 ', '0') #('4.0*1.5*x[1]*(0.81 - x[1]) / pow(0.81, 2)', '0')
#else:
#	inflow_profile =  (' x[1]* (0.3 - x[1]) / 0.0225 * 0.5 ','0')    #u=0.5 at the centerline


inlet_flow  = Expression( (' x[1]* (0.3 - x[1]) / 0.0225 * U ','0')  , U =0.5, degree = 2)


# Define test functions
(v,q) = TestFunctions(W)

# Define trial functions
w     = Function(W)
#(u,p) = (as_vector((w[0], w[1])), w[2])
#(u, p) = TrialFunctions(W)
(u, p) = split(w)


# Define boundary conditions
bcu_inflow = DirichletBC(W.sub(0), inlet_flow , inflow)
bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls)
bcp_outflow = DirichletBC(W.sub(1), Constant(0), outflow)

bc = [bcu_inflow, bcu_walls, bcp_outflow ]


# Define expressions used in variational forms
nu = Constant(nu)

#Week form
F =   dot(dot(u, nabla_grad(u)), v)*dx\
    	+ nu*inner(grad(u), grad(v))*dx \
    	- div(v)*p*dx \
    	- q*div(u)*dx


dw = TrialFunction(W)
dF = derivative(F, w)

nsproblem = NonlinearVariationalProblem(F, w, bc, dF)
solver = NonlinearVariationalSolver(nsproblem)
#solver.parameters["newton_solver"]["linear_solver"] = "gmres"
#solver.parameters["newton_solver"]["preconditioner"] = "default"




#set_log_level(30)

n=0
for i in range(num_simulation):

	inlet_flow.U  = inlet_U[i]
	n = n + 1
	if n % 1 == 0:
	 if MPI.rank(MPI.comm_world) == 0: 
	 	print('Solving weak form n= ', n,flush=True)
	solver.solve()
	if i== num_simulation-1 :  #Save the last one (steady solution of interest)
		u_sol , p_sol = w.split()
		u_sol.rename("u_sol", "Vel")
		xdmffile_v.write(u_sol, n)





if MPI.rank(MPI.comm_world) == 0:
	print('Done!',flush=True)



