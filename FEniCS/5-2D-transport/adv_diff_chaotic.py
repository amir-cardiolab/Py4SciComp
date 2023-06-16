"""
    ME 599 - Advanced CFD and FEM
    Midterm
    Transient Advection-diffusion equation in 2D
    a*grad(u) = D laplace(u) + f
    Domain - rectange [0,2]x[0,1]
    BC: zero Neumann everywhere   
    """

#from __future__ import print_function
from fenics import *
from mshr import *
import numpy as np
from math import pi
#import matplotlib.pyplot as plt


# Create mesh and define function space
domain = Rectangle(Point(0, 0), Point(2.0, 1.0))  #0<x<2 and 0<y<1
mesh = generate_mesh(domain, 120) #the higher the set value here, the morr resolved the mesh

V = FunctionSpace(mesh, 'P', 1)  #1st order FEM
VV = VectorFunctionSpace(mesh, 'P', 1)
# Parameters
A = 0.1
omega = 2.0*pi/10.0
epsilon = 0.25 

T = 20.0             # final time
num_steps = 2000     # number of time steps
dt = T / num_steps   # time step size

# Define initial condition
C0 = Expression('x[0] <= 1 ? 1 : 0', degree = 2)
# interpolate to functionspace and save IC
# u_n will be the solution at the previous timesteps
u_n = interpolate(C0, V)
vtkfile = File('advection/adv_IC.pvd')
vtkfile << u_n

# Define time dependent expressions for the velocity vector
a_t = Expression('epsilon*sin(omega*t)', degree = 2, t = 0.0, epsilon = epsilon, omega = omega)
b_t = Expression('1-2.0*epsilon*sin(omega*t)',degree = 2, t= 0.0, epsilon = epsilon, omega = omega)
f = Expression('a*pow(x[0],2)+b*x[0]', degree = 2, a = a_t, b= b_t)

u_vel = Expression('-pi*A*sin(pi*f)*cos(pi*x[1])', degree = 2, A = A, f = f)
v_vel = Expression('pi*A*(2*a*x[0]+b)*cos(pi*f)*sin(pi*x[1])', degree = 2,A=A, a=a_t, b=b_t, f=f)

Vvec=as_vector([u_vel, v_vel])

# Define variational problem
Diff_coef = 0.001   #Diffusion coefficient
u = TrialFunction(V)
v = TestFunction(V)

a = v*u*dx + dt*v * dot(Vvec,grad(u))*dx + dt*Diff_coef * dot(grad(u), grad(v))*dx 
# #L: right hand side. Note that all terms that are known (do not have u in them; should go to L)
L =  v*u_n*dx # no Neumann term, no source term, only the term containing the previous timesteps

u = Function(V)
t = 0

# Create files for saving timeseries
timeseries_u = TimeSeries('results/scalar_series')
vtkfile_u = File('results/scalar.pvd')
timeseries_Vvec = TimeSeries('results/velocity_series')
vtkfile_Vvec = File('results/velocity.pvd')

Vvec_n = Function(VV)
# Compute solution
for n in range(num_steps):
    
    # update time
    t += dt
    # update time dependent parameters and velocity vector
    a_t.t = t
    b_t.t = t
    f.a = a_t
    f.b = b_t
    u_vel.f = f
    v_vel.f = f
    v_vel.a = a_t
    v_vel.b = b_t
    
    solve(a == L, u)
    # save solution every 10 timesteps
    if n%10==0:
        vtkfile_u  << (u, t)
        Vvec_n = project(Vvec,VV)
        Vvec_n.rename('velocity','Label')
        vtkfile_Vvec << Vvec_n
        timeseries_u.store(u.vector(), t)
        timeseries_Vvec.store(Vvec_n.vector(), t)
    
    #Update previous solution
    u_n.assign(u)



print('Done!')