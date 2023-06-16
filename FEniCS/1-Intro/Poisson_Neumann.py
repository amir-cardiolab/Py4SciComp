"""
    FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
    Test problem is chosen to give an exact solution at all nodes of the mesh.
    -Diff*Laplace(u) = 0    in the unit square
    Diff = 0.0001 (Diffusion coefficient)
    u = u_D = 5.  on part of the boundary (y=2);  Dirichlet
	Diff* partial u /partial n =  0.05 on part of the boundary (x=0); Neumann; positive derivative --> sink flux 
    f = 0 #no source term
    """

from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
#import matplotlib.pyplot as plt


# Create mesh and define function space
channel = Rectangle(Point(0, 0), Point(1.0, 2.0))  #0<x<1 and 0<y<2
mesh = generate_mesh(channel, 20) #the higher the set value here, the more eresolved the mesh
#mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)  #1st order FEM





boundary_markers = MeshFunction('size_t', mesh,mesh.topology().dim()-1)  
boundary_markers.set_all(3) #Initialize all boundaries with a tag  = 3



#Neumann BC on x=0
class Boundary_wall_flux(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[0], 0., tol)  #x=0
bx1 = Boundary_wall_flux()
bx1.mark(boundary_markers, 1)  #lets tag 1 to the boundary that we want to specify flux (Neumann) BC


#Dirichlet BC on y=2.0
class Boundary_wall_constant(SubDomain):
    def inside(self, x, on_boundary):
        tol = 1E-14
        return on_boundary and near(x[1], 2., tol)  #y=2
bx2 = Boundary_wall_constant()
bx2.mark(boundary_markers, 2)  #lets tag 2 to the boundary that we want to specify constant concentration (Dirichlet) BC


#All the other boundaries will be the default boundary in FEM (no flux; zero Neumann)


bc_dirichlet = DirichletBC(V, Constant(5.),boundary_markers, 2) # I am assigning a constant Dirichlet BC u=5 on the boundary that I tagged as 2


ds = Measure("ds")[boundary_markers] #I need to define this to be able to do Neumann boundary; ds: boundary integral

# Define variational problem
Diff_coef = 0.0001 #Diffusion coefficient
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.)
flux = Constant(0.05)
a = Diff_coef * dot(grad(u), grad(v))*dx 
#L: right hand side. Note that all terms that are known (do not have u in them; should go to L)
L = f*v*dx + flux *v*ds(1) # Adding the Neumann BC term to the weak form (RHS). Note: It is on boundary that is tagged as 1: ds(1) 



# Compute solution
u = Function(V)
solve(a == L, u, bc_dirichlet)

# Plot solution and mesh
#plot(u)
#plot(mesh)

# Save solution to file in VTK format (use .pvd extension)
vtkfile = File('Results/solution_mixBC.pvd')
vtkfile << u

# Compute error in L2 norm
#error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
#vertex_values_u_D = u_D.compute_vertex_values(mesh)
#vertex_values_u = u.compute_vertex_values(mesh)
#error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

# Print errors
#print('error_L2  =', error_L2)
#print('error_max =', error_max)

print('Done!')

# Hold plot
#plt.show()