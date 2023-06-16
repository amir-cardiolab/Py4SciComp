#!/usr/bin/env python

__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-06"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

"""
This module implements a generic form of the fractional step method for
solving the incompressible Navier-Stokes equations. There are several
possible implementations of the pressure correction and the more low-level
details are chosen at run-time and imported from any one of:

  solvers/NSfracStep/IPCS_ABCN.py    # Implicit convection
  solvers/NSfracStep/IPCS_ABE.py     # Explicit convectionesh
  solvers/NSfracStep/IPCS.py         # Naive implict convection
  solvers/NSfracStep/BDFPC.py        # Naive Backwards Differencing IPCS in rotational form
  solvers/NSfracStep/BDFPC_Fast.py   # Fast Backwards Differencing IPCS in rotational form
  solvers/NSfracStep/Chorin.py       # Naive

The naive solvers are very simple and not optimized. They are intended
for validation of the other optimized versions. The fractional step method
can be used both non-iteratively or with iterations over the pressure-
velocity system.

The velocity vector is segregated, and we use three (in 3D) scalar
velocity components.

Each new problem needs to implement a new problem module to be placed in
the problems/NSfracStep folder. From the problems module one needs to import
a mesh and a control dictionary called NS_parameters. See
problems/NSfracStep/__init__.py for all possible parameters.

"""



#!!!!!!!!!Amir: changed imports in some of the files similar to the old 2016.2 oasis version to get rid of the complex path imports

#!!!! set BC file here

#what worked: WSS_star_Exp = 0.2e-1 , growth_Rate = 0.004, Tol_smoothing = 0.03 --> diverges at 90 (good smooth growth but the problem is at the end of stenosis, sudden restoration to healthy)
#Tol_smoothing = 0.02 --> diverges 97. Tol_smoothing = 0.01 --> has a bump

#unsteady:
#WSS_star_Exp = 0.2e-1 or 0.4e-1 still does not work (same issue as without unsteady).
#WSS_star_Exp = 0.2e-1: diverges at 66, but has grown a good amount and not that much problem with sudden jump to healthy.
#WSS_star_Exp = 0.1e-1 and growth_Rate = 0.002: diverges at 102. does not have issue with sudden jump but overlapping wierd elements emerge at the middle of stenosis. Same issue without unsteady (diverges at 81)
#WSS_star_Exp = 0.1e-1 and growth_Rate = 0.001: same issue. diverge at 168
#WSS_star_Exp = 0.05e-1. growth_rate = 0.0005; same propblem.

#unsteady (mesh_iso):
#WSS_star_Exp = 0.1e-1 ; growth_Rate=0.001  same issue, diverges at 103
#WSS_star_Exp = 0.15e-1 ; growth_Rate=0.004  same issue, diverges at 72 (also sudden jump). But has grown a good amount
#FINAL!!--> #WSS_star_Exp = 0.1e-1; growth_rate = 0.002, Tol_smoothing = 0.02 (or even 1e-12)==> same issue but good smooth jump to healthy
#injury2: same issue diverges 83

import sys, os #added by Amir

#sys.path.append(os.getcwd()) #added by Amir
import importlib
#from oasis import * #added by amir
#from oasis.common import *
from common import * #change by Amir
import gc #garbage collector (us gc.collect() after del to prevent memory leak)
#import resource # for tracking memory
import math
import vtk
from vtk.util import numpy_support as VN

#from problems import * #added by Amir
#from problems.NSfracStep import * #added by Amir

#added parabolic inlet (Mostafa's code) for image-based models

#input files
inlet_facet_ID = 1
if(0):
 output_root =  '/Users/aa3878/data/stenosis_ALE/coronary/mesh_results/'
 output_mesh_name = 'stenosis_mesh'
 output_filename =   output_root +  output_mesh_name + '.pvd'
 output_filename_vel =  '/Users/aa3878/data/stenosis_ALE/coronary/mesh_results/stenosis_vel.pvd'
 output_filename_wss =  '/Users/aa3878/data/stenosis_ALE/coronary/mesh_results/stenosis_wss.pvd'
 distance_file = '/Users/aa3878/data/stenosis_ALE/tube_mesh/distance_tube.xml'
if(0): #final tube model (BL meshing)
 output_root = '/Users/aa3878/data/stenosis_ALE/coronary/temp_results/' #'/Users/aa3878/data/stenosis_ALE/coronary/mesh_results_BL_O2/proximal_stenosis/'
 #output_mesh_name = 'stenosis_mesh' + '5e1s'
 #output_mesh_name = 'stenosis_mesh' + '_restrict' + '_asym'
 output_mesh_name = 'stenosis_mesh' + '_normals'
 output_filename =  output_root +  output_mesh_name + '.pvd'
 output_filename_vel =  '/Users/aa3878/data/stenosis_ALE/coronary/temp_results/stenosis_vel_normal.pvd'
 output_filename_wss =  '/Users/aa3878/data/stenosis_ALE/coronary/temp_results/stenosis_wss_normal.pvd'
 distance_file = '/Users/aa3878/data/stenosis_ALE/tube_final/distance_tube.xml'#'/Users/aa3878/data/stenosis_ALE/tube_mesh/distance_tube_meshBL.xml'
if(0): #image-based coronary
 output_root =  '/Users/aa3878/data/stenosis_ALE/Symon-models/growth_results/'
 output_mesh_name = 'IBstenosis_mesh' + '_normals' #'mesh_nosmooth'
 output_filename =  output_root +  output_mesh_name + '.pvd'
 output_filename_vel =  '/Users/aa3878/data/stenosis_ALE/Symon-models/growth_results/IBstenosis_vel_normal.pvd'
 output_filename_wss =  '/Users/aa3878/data/stenosis_ALE/Symon-models/growth_results/IBstenosis_wss_normal.pvd'
 if(1):
  distance_file = '/Users/aa3878/data/stenosis_ALE/Symon-models/distance_coronary_iso.xml' #'/Users/aa3878/data/stenosis_ALE/Symon-models/distance_coronary.xml'
 else:
  distance_file = '/Users/aa3878/data/stenosis_ALE/Symon-models/distance2_coronary_iso.xml'
 injury_file =  '/Users/aa3878/data/stenosis_ALE/Symon-models/C0_pz-mesh-complete/injury.xml'
if(1): #tube final
  output_root =  '/Users/aa3878/data/stenosis_ALE/tube_final/results/' #+'steady/'
  output_mesh_name = 'tubeFinal_mesh' + '_normals'
  output_filename =  output_root +  output_mesh_name + '.pvd'
  output_filename_vel = '/Users/aa3878/data/stenosis_ALE/tube_final/results/tubeFinal_vel_normal.pvd'
  output_filename_wss = '/Users/aa3878/data/stenosis_ALE/tube_final/results/ItubeFinal_wss_normal.pvd'
  distance_file = '/Users/aa3878/data/stenosis_ALE/tube_final/distance_tube.xml'
  injury_file =  '/Users/aa3878/data/stenosis_ALE/Symon-models/C0_pz-mesh-complete/injury.xml'#not needed here
#parameters:
if(0):#tube
 N_growth = 120 #150 #200
 growth_rate = 0.04  #0.06 + 0.02
 WSS_star = 10000 #410000. #mg/mm.s^2 (1000 = 1 Pa)    #zohdi paper
 WSS_star_Exp = 1e-4 #parameter in exponential WSS dependent growth function
 flag_smooth_boundary = True #Use VTK to smooth surface mesh
 Flag_smooth_method = 1 #if 1 then uses taubin (VMTK) to smooth; if =0 use sLaplace smoothing
 N_freq_Bsmooth = 1
 Num_iterations_Bsmooth = 25 #20  #for Flag_smooth_method = 0
 Num_iterations_Windowed = 10 #20  #for Flag_smooth_method = 1
 Tol_smoothing = 1e-3 #1e-2 #1e-3 (also 1e-2, 1e-1) gives acceptable results (the tolerance, based on initial Gaussian function to apply smoothing). 1e-2 used for Flag_smooth_method = 0
 Flag_writeVelWSS = True
 Flag_restrict_smoothing = False #True #restricts smoothing to only nodes that have high displacement
 Thresh_restrict_smooth = 0.01 * growth_rate #what percentage of growth_rate is our threshold s.t. only nodes with higher displacement are smoothed. 0.1 does not give smooth results
 Flag_assym = False #Assymetric injury (stenosis)
 Flag_read_injury = False #if True reads the injury model (for image-based models)
 Flag_unsteady = True #False # unsteady sigma (According to a diffusion model set below)
 Flag_normal_update = True #if true, it will not update the normal vectors as the stenosis grows
if(1):#tubefinal
 N_growth = 120
 growth_rate = 0.002 #0.004
 WSS_star = 10000 #410000. #mg/mm.s^2 (1000 = 1 Pa)    #zohdi paper
 WSS_star_Exp = 0.1e-1 #0.2e-1   #1e-1 #parameter in exponential WSS dependent growth function. (0.1e-1 gets distorted fast). (0.2e-1 with tol_smoth 1e-1 gets distorted at t.s. 74)
 flag_smooth_boundary = True #Use VTK to smooth surface mesh
 Flag_smooth_method = 1 #if 1 then uses taubin (VMTK) to smooth; if =0 use sLaplace smoothing
 N_freq_Bsmooth = 1
 Num_iterations_Bsmooth = 25 #20  #for Flag_smooth_method = 0
 Num_iterations_Windowed = 10 #10 #20  #for Flag_smooth_method = 1 (default value we used 10)
 Tol_smoothing = 0.02 #0.03  #1e-1 #1e-3 #1e-2 better than 1e-3 (there is a bump). 1e-1 has no bump (a bit less smooth). 0.07 with grouwth 0.008 not good.
 Flag_writeVelWSS = True
 Flag_restrict_smoothing = False #True #restricts smoothing to only nodes that have high displacement
 Thresh_restrict_smooth = 0.01 * growth_rate #what percentage of growth_rate is our threshold s.t. only nodes with higher displacement are smoothed. 0.1 does not give smooth results
 Flag_assym = False #Assymetric injury (stenosis)
 Flag_read_injury = False  #if True reads the injury model (for image-based models)
 Flag_unsteady = True # unsteady sigma (According to a diffusion model set below)
 if(1): # not needed here
  injury_file_dist = '/Users/aa3878/data/stenosis_ALE/Symon-models/growth_results/restart_50/injury_dist.xml' #'/Users/aa3878/data/stenosis_ALE/Symon-models/C0_pz-mesh-complete/injury_dist_iso.xml'  #'/Users/aa3878/data/stenosis_ALE/Symon-models/C0_pz-mesh-complete/injury_dist.xml' # just saves -dist^2 (for unsteady sigma model)
 else:
  injury_file_dist = '/Users/aa3878/data/stenosis_ALE/Symon-models/C0_pz-mesh-complete/injury2_dist_iso.xml'
 Flag_normal_update = True #if false, it will not update the normal vectors as the stenosis grows (a small part gets overlapping surface elements)

#u0_zero = Constant((0.0,0.0,0.) )

def get_normal(mymesh): #how to get normals? it gives normals on the interior (next to boundary) too. The solution vector is also not normalized!
    facets = facet_domains
    ds = Measure("ds")[facets]
    n_normal_m = FacetNormal(mymesh)
    h = FacetArea(mymesh)
    h_v = CellVolume(mymesh)
    #Weak form to compute boundary normals as a global function
    VVn = VectorFunctionSpace(mymesh, 'CG', 1)
    normal_fn = Function(VVn)
    un = TrialFunction(VVn)
    vn = TestFunction(VVn)
    L= assemble( inner(un,vn)/h_v*dx, keep_diagonal=True )
    #a = assemble( inner(n_normal_m,v)/h*ds(1) )
    a = assemble( inner(n_normal_m,vn)/h*ds() )
    solve(L, normal_fn.vector(), a, 'gmres', 'default')
    #File( '/Users/aa3878/data/stenosis_ALE/test/teeeeeeeesting_n.pvd') <<  normal_fn
    return normal_fn

def solve_Laplace(mymesh,disp_BC): #solve Laplace eqn for smoothing. #!!!!!!!!!!!!!!!!! set the boundary tag faces below
    VVp = VectorFunctionSpace(mymesh, 'CG', 1)
    bc = [DirichletBC(VVp, (0.0,0.0,0.), facet_domains, 1),DirichletBC(VVp, (0.0,0.0,0.), facet_domains, 3),DirichletBC(VVp, (0.0,0.0,0.), facet_domains, 4), DirichletBC(VVp, disp_BC, facet_domains, 2) ] #All other faces zero
    u = TrialFunction(VVp)
    v = TestFunction(VVp)
    if(1): # Use a diffusion coeff. (See Fluent theory guide); This  makes it work !!!!!
        a = myfunc_smooth2 * inner(grad(u), grad(v))*dx
    else:
        a = inner(grad(u), grad(v))*dx
    u0_zero = Constant((0.0,0.0,0.) )
    L = inner(u0_zero,v)*dx
    uu = Function(VVp)
    solve(a == L, uu, bc, solver_parameters={'linear_solver': 'gmres'})
    #File( '/Users/aa3878/data/stenosis_ALE/test/teeeeeeeesting_Laplace.pvd') <<  uu
    return uu

def update_mesh(mesh_in, displacement): # https://fenicsproject.org/qa/13470/ale-move-class-and-meshes/
    new_mesh = mesh_in
    ALE.move(new_mesh, displacement)
    return new_mesh

def copy_mesh(mesh_in):
    new_mesh = mesh_in
    return new_mesh


commandline_kwargs = parse_command_line()

default_problem = 'DrivenCavity'

if(0):
 problemname = commandline_kwargs.get('problem', default_problem)
 try:
    #problemmod = importlib.import_module('.'.join(('oasis.problems.NSfracStep', problemname)))
    problemmod = importlib.import_module('.' + problemname,package='problems.NSfracStep') #changed by Amir
 except ImportError:
    problemmod = importlib.import_module(problemname)
 #except: #commented by Amir
 #    raise RuntimeError(problemname+' not found')

 vars().update(**vars(problemmod))
 # Update problem spesific parameters
 problem_parameters(**vars())
 # Update current namespace with NS_parameters and commandline_kwargs ++
 vars().update(post_import_problem(**vars()))
 # Import chosen functionality from solvers
 solver = importlib.import_module('.'.join(('oasis.solvers.NSfracStep', solver)))
 vars().update({name:solver.__dict__[name] for name in solver.__all__})

if(1):
 exec("from problems.NSfracStep.{} import *".format(commandline_kwargs.get('problem', default_problem)))
 # Update current namespace with NS_parameters and commandline_kwargs ++
 vars().update(post_import_problem(**vars()))
 # Import chosen functionality from solvers
 exec("from solvers.NSfracStep.{} import *".format(solver))

# Create lists of components solved for
dim = mesh.geometry().dim()
u_components = ['u' + str(x) for x in range(dim)]
sys_comp = u_components + ['p'] + scalar_components
uc_comp = u_components + scalar_components

# Set up initial folders for storing results
newfolder, tstepfiles,tstepfiles_wss = create_initial_folders(**vars())

############################---------------
num_nodes = mesh.num_vertices()

if (Flag_read_injury):
  if (Flag_unsteady):
      myfunc_disp_dist = Function(FunctionSpace(mesh, 'CG', 1),  injury_file_dist ) #read -dist^2
      myfunc_disp = Function(FunctionSpace(mesh, 'CG', 1) )
  else:
      myfunc_disp = Function(FunctionSpace(mesh, 'CG', 1),  injury_file )  #read full Gausian injury
else:
  if (Flag_unsteady):
    myexp_disp = Expression( "-pow( (x[2]-0.86) , 2)" ,degree=2 ) #-dist^2
    myfunc_disp_dist = interpolate(myexp_disp,FunctionSpace(mesh, 'CG', 1))
    myfunc_disp = Function(FunctionSpace(mesh, 'CG', 1) )
  else:
    #myexp_disp = Expression( "exp( -pow((x[2]-14.18)/0.54 , 2)  )" ,degree=2 ) #Gaussian injury (at the center of the length)
    myexp_disp = Expression( "exp( -pow((x[2]-0.86)/0.1 , 2)  )" ,degree=2 ) #Gaussian injury (closer to the inlet) #!!!!!!!!!!!!! need to update distance_file  for this!!!!
    myfunc_disp = interpolate(myexp_disp,FunctionSpace(mesh, 'CG', 1))


if(Flag_assym): #Assymetric injury
  myexp_disp_a = Expression( "x[1] " ,degree=2 ) #Gaussian injury
  myfunc_disp_a = interpolate(myexp_disp_a,FunctionSpace(mesh, 'CG', 1))
  for i in xrange(num_nodes):
    if ( myfunc_disp_a.vector()[i] < 1.9 ): #only injury for y > 1.9 (assymetric)
        myfunc_disp.vector()[i] = 0.0
  myfunc_disp.vector().apply('')


myfunc_smooth2 = Function(FunctionSpace(mesh, 'CG', 1),  distance_file )
#myfunc_smooth2 = interpolate(myexp_smooth2,FunctionSpace(mesh, 'CG', 1))

#bmesh = mesh
V_vect = VectorFunctionSpace(mesh, 'CG', 1)
Q_s = FunctionSpace(mesh, 'CG', 1)
disp_vect = Function(V_vect)
disp_s1 = Function(Q_s)
disp_s2 = Function(Q_s)
disp_s3 = Function(Q_s)
disp_nparray = numpy.zeros(num_nodes*3)
disp_nparray2 = numpy.zeros(num_nodes*3)

file_output2 = File(output_filename)
file_output_vel = File(output_filename_vel)
file_output_wss = File(output_filename_wss)
file_output_before_smooth = File (output_root + 'mesh_nosmooth.pvd')

tic()
total_timer = OasisTimer("Start simulations", True)

for jj in xrange(N_growth):
 print 'jj=', jj
 # Declare FunctionSpaces and arguments
 if (jj>0):
  #del K,A,M,u_ab,u_ab_prev,a_conv,a_scalar,LT,KT,visc_nu,Ap2,Ap,divu,gradp
  #del n_normal,uv,p_,p_1,dp_, b,b_tmp,x_,x_1,x_2,U_AB,u_,u_1,u_2,q_,q_1,q_2,u,v,VV,p,q,V,Q
  del K,A,M,u_ab,u_ab_prev,a_conv,a_scalar,LT,KT,visc_nu,Ap2,Ap,divu,dp_,gradp,x_
  gc.collect()
  del n_normal,uv,p_,p_1, b,b_tmp,x_1,x_2,U_AB,u_,u_1,u_2,q_,q_1,q_2,u,v,VV,p,q,V,Q
  gc.collect()
 V = FunctionSpace(mesh, 'CG', velocity_degree,constrained_domain=constrained_domain)
 if velocity_degree == pressure_degree:
  Q = FunctionSpace(mesh, 'CG', velocity_degree,constrained_domain=constrained_domain)
 if velocity_degree != pressure_degree:
    Q = FunctionSpace(mesh, 'CG', pressure_degree,
                      constrained_domain=constrained_domain)

 
 u = TrialFunction(V)
 v = TestFunction(V)
 p = TrialFunction(Q)
 q = TestFunction(Q)
 if (jj==0):
  bc_file = '/Users/aa3878/data/stenosis_ALE/tube_final/meshtube_final-mesh-complete/mesh-surfaces/BCnodeFacets.xml' #'/Users/aa3878/data/stenosis_ALE/Symon-models/iso-mesh-complete/mesh-surfaces/BCnodeFacets.xml' #'/Users/aa3878/data/stenosis_ALE/Symon-models/C0_pz-mesh-complete/mesh-surfaces/BCnodeFacets.xml'
  facet_domains = MeshFunction('size_t', mesh,bc_file )
  if (not Flag_normal_update):
    my_normal = get_normal(mesh)
 
 if(jj==0): #Rotating the mesh so that the average normal vector aligned with z-axis passing the centroid
      d = mesh.geometry().dim()
      x = SpatialCoordinate(mesh)
      ds = Measure("ds")(subdomain_data=facet_domains)
      dsi = ds(inlet_facet_ID, domain=mesh, subdomain_data=facet_domains)
      # Compute area of boundary tesselation by integrating 1.0 over all facets
      A = assemble(Constant(1.0, name="one")*dsi)
      if MPI.rank(mpi_comm_world()) == 0:
              print 'Inlet area:', A
      inlet_radius = numpy.sqrt(A / pi) # This old estimate is a few % lower because of boundary discretization errors
      inlet_radius2 = A / pi #It should be r**2 in the parabolic eqn below!
      if MPI.rank(mpi_comm_world()) == 0:
          print 'Inlet radius:', inlet_radius
      # Compute barycenter by integrating x components over all facets
      center = [assemble(x[i]*dsi) / A for i in xrange(d)]
      if MPI.rank(mpi_comm_world()) == 0:
              print 'Inlet center:', center
      center = Point(numpy.array(center))
      # Compute average normal (assuming boundary is actually flat)
      n_normal= FacetNormal(mesh)
      ni = numpy.array([assemble(n_normal[i]*dsi) for i in xrange(d)])
      n_len = numpy.sqrt(sum([ni[i]**2 for i in xrange(d)])) # Should always be 1!?
      normal = ni/n_len
      if MPI.rank(mpi_comm_world()) == 0:
              print 'normal vector: ', normal
      alpha_ang = (normal[0]/abs(normal[0]))*acos(normal[0])*180./pi
      beta_ang = (normal[1]/abs(normal[1]))*acos(normal[1])*180./pi
      gamma_ang = -(normal[2]/abs(normal[2]))*acos(normal[2])*180./pi
      #MeshTransformation.rotate(mesh, gamma_ang, 0, center)
      mesh.rotate(gamma_ang, 0, center)

      d = mesh.geometry().dim()
      x = SpatialCoordinate(mesh)
      ds = Measure("ds")(subdomain_data=facet_domains)
      dsi = ds(inlet_facet_ID, domain=mesh, subdomain_data=facet_domains)
      # Compute area of boundary tesselation by integrating 1.0 over all facets
      A = assemble(Constant(1.0, name="one")*dsi)
      inlet_radius = numpy.sqrt(A / pi) # This old estimate is a few % lower because of boundary discretization errors
      inlet_radius2 = A / pi #It should be r**2 in the parabolic eqn below!
      if MPI.rank(mpi_comm_world()) == 0:
          print 'Inlet radius after rotation (1):', inlet_radius
      # Compute barycenter by integrating x components over all facets
      center = [assemble(x[i]*dsi) / A for i in xrange(d)]
      if MPI.rank(mpi_comm_world()) == 0:
              print 'Inlet center after rotation (1):', center
      # Compute barycenter by integrating x components over all facets
      center = [assemble(x[i]*dsi) / A for i in xrange(d)]
      center = Point(numpy.array(center))
      n_normal= FacetNormal(mesh)
      ni = numpy.array([assemble(n_normal[i]*dsi) for i in xrange(d)])
      n_len = numpy.sqrt(sum([ni[i]**2 for i in xrange(d)])) # Should always be 1!?
      normal = ni/n_len
      if MPI.rank(mpi_comm_world()) == 0:
              print 'normal vector after rotation (1): ',normal
      alpha_ang = (normal[0]/abs(normal[0]))*acos(normal[0])*180./pi
      beta_ang = (normal[1]/abs(normal[1]))*acos(normal[1])*180./pi
      gamma_ang = -(normal[2]/abs(normal[2]))*acos(normal[2])*180./pi

      #MeshTransformation.rotate(mesh, gamma_ang, 1, center)
      mesh.rotate(gamma_ang, 1, center)

      d = mesh.geometry().dim()
      x = SpatialCoordinate(mesh)
      ds = Measure("ds")(subdomain_data=facet_domains)
      dsi = ds(inlet_facet_ID, domain=mesh, subdomain_data=facet_domains)
      A = assemble(Constant(1.0, name="one")*dsi)
      if MPI.rank(mpi_comm_world()) == 0:
              print 'Inlet area after rotation:', A
      inlet_radius = numpy.sqrt(A / pi) # This old estimate is a few % lower because of boundary discretization errors
      inlet_radius2 = A / pi #It should be r**2 in the parabolic eqn below!
      if MPI.rank(mpi_comm_world()) == 0:
          print 'Inlet radius after rotation:', inlet_radius
      # Compute barycenter by integrating x components over all facets
      center = [assemble(x[i]*dsi) / A for i in xrange(d)]
      if MPI.rank(mpi_comm_world()) == 0:
              print 'Inlet center after rotation:', center
      n_normal= FacetNormal(mesh)
      ni = numpy.array([assemble(n_normal[i]*dsi) for i in xrange(d)])
      n_len = numpy.sqrt(sum([ni[i]**2 for i in xrange(d)])) # Should always be 1!?
      normal = ni/n_len
      if MPI.rank(mpi_comm_world()) == 0:
              print 'normal vector after rotation: ',normal
 

 # Use dictionary to hold all FunctionSpaces
 #if (jj>0):
 # del VV
 # gc.collect()
 VV = dict((ui, V) for ui in uc_comp)
 VV['p'] = Q

 # Create dictionaries for the solutions at three timesteps
 #if (jj>0):
 #    del q_
 #    gc.collect()
 #    del q_1
 #    gc.collect()
 #    del q_2
 #    gc.collect()
 q_  = dict((ui, Function(VV[ui], name=ui)) for ui in sys_comp)
 q_1 = dict((ui, Function(VV[ui], name=ui + "_1")) for ui in sys_comp)
 q_2 = dict((ui, Function(V, name=ui + "_2")) for ui in u_components)

 # Read in previous solution if restarting
 init_from_restart(**vars())

 # Create vectors of the segregated velocity components
 #if (jj>0):
 #    del u_
 #    gc.collect()
 #    del u_1
 #    gc.collect()
 #    del u_2
 #    gc.collect()
 u_  = as_vector([q_ [ui] for ui in u_components]) # Velocity vector at t
 u_1 = as_vector([q_1[ui] for ui in u_components]) # Velocity vector at t - dt
 u_2 = as_vector([q_2[ui] for ui in u_components]) # Velocity vector at t - 2*dt

 # Adams Bashforth  ion of velocity at t - dt/2
 #if (jj>0):
 # del U_AB
 # gc.collect()
 U_AB = 1.5 * u_1 - 0.5 * u_2

 # Create short forms for accessing the solution vectors
 #if (jj>0):
 #    del x_
 #    gc.collect()
 #    del x_1
 #    gc.collect()
 #    del x_2
 #    gc.collect()
 x_ = dict((ui, q_[ui].vector()) for ui in sys_comp)        # Solution vectors t
 x_1 = dict((ui, q_1[ui].vector()) for ui in sys_comp)      # Solution vectors t - dt
 x_2 = dict((ui, q_2[ui].vector()) for ui in u_components)  # Solution vectors t - 2*dt

 # Create vectors to hold rhs of equations
 #if (jj>0):
 #    del b
 #    gc.collect()
 #    del b_tmp
 #    gc.collect()
 b = dict((ui, Vector(x_[ui])) for ui in sys_comp)      # rhs vectors (final)
 b_tmp = dict((ui, Vector(x_[ui])) for ui in sys_comp)  # rhs temp storage vectors

 # Short forms pressure and scalars
 #if (jj>0):
 # del p_
 # gc.collect()
 # del p_1
 # gc.collect()
 # del dp_
 # gc.collect()
 p_ = q_['p']                # pressure at t
 p_1 = q_1['p']              # pressure at t - dt
 dp_ = Function(Q)           # pressure correction
 for ci in scalar_components:
    exec("{}_   = q_ ['{}']".format(ci, ci))
    exec("{}_1  = q_1['{}']".format(ci, ci))

 print_solve_info = use_krylov_solvers and krylov_solvers['monitor_convergence']

 # Anything problem specific
 #if (jj>0):
 # del n_normal,uv
 # gc.collect()
 vars().update(pre_solve_hook(**vars())) #place it before update, since it needs n_normal and facet_domains. Also before create_bcs
 #n_normal = FacetNormal(mesh) #just update the normals


 # Boundary conditions
 if (flag_ramp):
    t = initial_time_ramp
 t_cycle = t
 if (jj==0): #just once
   bcs = create_bcs(**vars())

 if(0):
  # LES setup
  #exec("from oasis.solvers.NSfracStep.LES.{} import *".format(les_model))
  lesmodel = importlib.import_module('.'.join(('oasis.solvers.NSfracStep.LES', les_model)))
  vars().update({name:lesmodel.__dict__[name] for name in lesmodel.__all__})
 if(1):
  # LES setup
  exec("from solvers.NSfracStep.LES.{} import *".format(les_model))
 if(0):
  vars().update(les_setup(**vars())) #commented Amir
 else:
  nut_ = []

 # Initialize solution
 initialize(**vars())

 #  Fetch linear algebra solvers
 if (jj>0):
  del u_sol
  gc.collect()
  del p_sol
  gc.collect()
  del c_sol
  gc.collect()
 u_sol, p_sol, c_sol = get_solvers(**vars())

 # Get constant body forces
 if (jj>0):
     del f
     gc.collect()
     del b0
     gc.collect()
 f = body_force(**vars())
 assert(isinstance(f, Coefficient))
 b0 = dict((ui, assemble(v * f[i] * dx)) for i, ui in enumerate(u_components))

 # Get scalar sources
 if (jj>0):
  del fs  #do not del b0 (wont be redefined)
  gc.collect()
 fs = scalar_source(**vars())
 for ci in scalar_components:
    assert(isinstance(fs[ci], Coefficient))
    b0[ci] = assemble(v * fs[ci] * dx)

 # Preassemble and allocate
 vars().update(setup(**vars())) #this causes memory leak!




 stop = False
 t = 0.
 t_cycle = t
 v2d_V = dof_to_vertex_map(V_vect)
 v2d_Vs = dof_to_vertex_map(Q)
 while t < (T - tstep * DOLFIN_EPS) and not stop:
    if (t_cycle > Time_last):
        t_cycle = t_cycle - Time_last
    if(0): #update time in BC if time-dependent inflow expression is specified. DONT UPDATE FOR STEADY
      bcs = create_bcs(**vars())
    t += dt
    t_cycle +=dt
    tstep += 1
    inner_iter = 0
    udiff = array([1e8])  # Norm of velocity change over last inner iter
    num_iter = max(iters_on_first_timestep, max_iter) if tstep == 1 else max_iter

    start_timestep_hook(**vars())

    while udiff[0] > max_error and inner_iter < num_iter:
        inner_iter += 1

        t0 = OasisTimer("Tentative velocity")
        if inner_iter == 1:
            if(0):
             les_update(**vars()) #commented amir
            assemble_first_inner_iter(**vars())
    
        udiff[0] = 0.0
        for i, ui in enumerate(u_components):
            t1 = OasisTimer('Solving tentative velocity ' + ui, print_solve_info)
            velocity_tentative_assemble(**vars())
            velocity_tentative_hook(**vars())
            velocity_tentative_solve(**vars())
            t1.stop()

        t0 = OasisTimer("Pressure solve", print_solve_info)
        pressure_assemble(**vars())
        pressure_hook(**vars())
        pressure_solve(**vars())
        t0.stop()

        print_velocity_pressure_info(**vars())

    # Update velocity
    t0 = OasisTimer("Velocity update")
    velocity_update(**vars())
    t0.stop()
    # Solve for scalars
    if len(scalar_components) > 0:
        scalar_assemble(**vars())
        for ci in scalar_components:
            t1 = OasisTimer('Solving scalar {}'.format(ci), print_solve_info)
            scalar_hook(**vars())
            scalar_solve(**vars())
            t1.stop()


    temporal_hook(**vars())

    # Save solution if required and check for killoasis file
    stop = save_solution(**vars())


    # Update to a new timestep
    for ui in u_components:
        x_2[ui].zero()
        x_2[ui].axpy(1.0, x_1[ui])
        x_1[ui].zero()
        x_1[ui].axpy(1.0, x_[ui])
        #new method (Amir):
        #x_2[ui].set_local(x_1[ui].array())
        #x_2[ui].apply("insert")
        #x_1[ui].set_local(x_[ui].array())
        #x_1[ui].apply("insert")

    for ci in scalar_components:
        x_1[ci].zero()
        x_1[ci].axpy(1., x_[ci])

    if MPI.rank(mpi_comm_world()) == 0:
        print 'growth index number:', jj

    # Print some information
    if tstep % print_intermediate_info == 0:
        info_green( 'Time = {0:2.4e}, timestep = {1:6d}, End time = {2:2.4e}'.format(t, tstep, T))
        info_red('Total computing time on previous {0:d} timesteps = {1:f}'.format(
            print_intermediate_info, toc()))
        list_timings(TimingClear_clear, [TimingType_wall])
        tic()

    ##test visc_gamma
    #if (tstep % 20 == 0):
    #    print '------projecting..'
    #    solve(K_test == K_test_rhs, gamma_solution,solver_parameters={'linear_solver': 'gmres'})
    #    file_gamma <<gamma_solution

    # AB projection for pressure on next timestep
    if AB_projection_pressure and t < (T - tstep * DOLFIN_EPS) and not stop:
        x_['p'].axpy(0.5, dp_.vector())
 #get WSS:
 if (jj>0):
  del n_normal2
  del V_boundary
  del u_wss
  del v_wss
  del LHS_wss
  del RHS_wss
  del wss
  del u_soln
  if (Flag_normal_update):
   del my_normal
  del disp_smoothed
  gc.collect()
 if(1):
  n_normal2 = FacetNormal(mesh)
  V_boundary = VectorFunctionSpace(mesh, 'CG', 1)
  u_wss = TrialFunction(V_boundary)
  v_wss = TestFunction(V_boundary)
  wss = Function(V_boundary,name="wss")
  T1 = -visc_nu * 1.06 * dot((grad(u_) + grad(u_).T), n_normal2)
  Tn = dot(T1, n_normal2)
  Tt = T1 - Tn*n_normal2
  LHS_wss = assemble(inner(u_wss, v_wss) * ds, keep_diagonal=True)
  RHS_wss = assemble(inner(Tt, v_wss) * ds)
  LHS_wss.ident_zeros()
  solve(LHS_wss, wss.vector(), RHS_wss,'gmres', 'default')

  u_soln = AssignedVectorFunction(u_)
  u_soln()
  if (Flag_writeVelWSS):
   file_output_vel << u_soln
   file_output_wss << wss

 #mesh movement calculation#--------
 if(1):
  if(Flag_unsteady):#set sigma according to sigma = sigma_0 + sqrt(D*t) diffusion injury sigma propogation
     sigma_temp = 0.1 + math.sqrt(2.1008403361344542e-05 * jj) #such that it goes btwn sigma = 0.1 and 0.15
     #sigma_temp = 0.1 #steady
     for i in xrange(num_nodes):
          myfunc_disp.vector()[i] = math.exp( myfunc_disp_dist.vector()[i] / sigma_temp**2 )
     myfunc_disp.vector().apply('')
  if (Flag_normal_update):
    my_normal = get_normal(mesh)
  for i in xrange(num_nodes):
    normal_mag = ( pow(my_normal.vector()[3*i],2) + pow(my_normal.vector()[3*i+1],2) + pow(my_normal.vector()[3*i+2],2) )**0.5
    if (normal_mag < 1e-6):
       normal_mag = -1.0
    WSS_mag = ( pow(wss.vector()[3*i],2) + pow(wss.vector()[3*i+1],2) + pow(wss.vector()[3*i+2],2) )**0.5
    if (WSS_mag < 1e-18):
        growth_fact = 0.
    else:
        #growth_fact = max(0.0, (WSS_star -WSS_mag)/WSS_star  ) * growth_rate #Zohdi model
        #growth_fact = math.exp(-(1e-4) * WSS_mag) * growth_rate #An exponentially diminishing model (tube)
        growth_fact = math.exp(-(WSS_star_Exp) * WSS_mag) * growth_rate #An exponentially diminishing model (coronary)
    n_v = myfunc_disp.vector()[i] *  -1.0 * growth_fact
    if ( normal_mag > 0  ):#only exterior elements have nonzero normal
      n_v = n_v * my_normal.vector()[3*i] / normal_mag
    else:
      n_v = 0.
    disp_nparray[3*i] = n_v
    n_v = myfunc_disp.vector()[i] *  -1.0 * growth_fact
    if ( normal_mag > 0  ):#only exterior elements have nonzero normal
        n_v = n_v * my_normal.vector()[3*i+1] / normal_mag
    else:
        n_v = 0.
    disp_nparray[3*i+1] = n_v
    n_v = myfunc_disp.vector()[i] *  -1.0 * growth_fact
    if ( normal_mag > 0  ):#only exterior elements have nonzero normal
        n_v = n_v * my_normal.vector()[3*i+2] / normal_mag
    else:
        n_v = 0.
    disp_nparray[3*i+2] = n_v
  disp_vect.vector().set_local(disp_nparray[:])
  disp_vect.vector().apply('')
  disp_smoothed = solve_Laplace(mesh,disp_vect)
  if (flag_smooth_boundary and jj % N_freq_Bsmooth == 0): # take a copy of the mesh before moving.
     #mesh_copy = Mesh(mesh)
     #mesh_copy = mesh
     #mesh_copy = copy_mesh(mesh)
     file_output_temp = File (output_root + 'mesh_temp.pvd')
     file_output_temp << mesh
  ALE.move(mesh,disp_smoothed)
  #mesh = update_mesh(mesh,disp_smoothed)
  my_quality = MeshQuality.radius_ratio_min_max(mesh)
  print '------jj=', jj
  print '------Min,Max quality',my_quality
  if (not flag_smooth_boundary or not jj % N_freq_Bsmooth == 0):
    file_output2 << mesh
  else:
    file_output_temp = File (output_root + 'mesh_moved_temp.pvd')
    file_output_temp << mesh
    #file_output_before_smooth << mesh
  if (flag_smooth_boundary and jj % N_freq_Bsmooth == 0): #Use VTK smoothing to smooth surface
        #move back the mesh
        for i in xrange(num_nodes):
          disp_nparray2[3*i]   =  -disp_smoothed.vector()[3*i]
          disp_nparray2[3*i+1] = -disp_smoothed.vector()[3*i+1]
          disp_nparray2[3*i+2] = -disp_smoothed.vector()[3*i+2]
        disp_vect.vector().set_local(disp_nparray2[:])
        disp_vect.vector().apply('')
        ALE.move(mesh,disp_vect)
        #smooth
        bsmooth_delta_x = numpy.zeros((num_nodes,1))
        bsmooth_delta_y = numpy.zeros((num_nodes,1))
        bsmooth_delta_z = numpy.zeros((num_nodes,1))
        cord_pt = numpy.zeros((3,1))
        cord_pt_temp = numpy.zeros((3,1))
        Mesh_file_vtu = output_root + 'mesh_moved_temp000000.vtu'
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(Mesh_file_vtu )
        reader.Update()
        data_m = reader.GetOutput()
        Mesh_file_temp = output_root + 'mesh_temp000000.vtu'
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(Mesh_file_temp )
        reader.Update()
        data_temp = reader.GetOutput()
        #Add a new data field "NodeID" that saves the index number of each node
        nodeindex = vtk.vtkIntArray()
        nodeindex.SetNumberOfValues(num_nodes)
        nodeindex.SetName("NodeID")
        for k in xrange(num_nodes):
          nodeindex.SetValue(k,k)
        data_m.GetPointData().SetScalars(nodeindex)
        #extract surface
        meshtosurf = vtk.vtkDataSetSurfaceFilter()
        meshtosurf.SetInputData(data_m)
        meshtosurf.Update()
        data_s = meshtosurf.GetOutput()
        num_nodes_s = data_s.GetNumberOfPoints()
        #extract surface
        meshtosurf = vtk.vtkDataSetSurfaceFilter()
        meshtosurf.SetInputData(data_temp)
        meshtosurf.Update()
        data_temp_s = meshtosurf.GetOutput()
        #smooth surface mesh
        if (Flag_smooth_method==1): 
         smoothingFilter = vtk.vtkWindowedSincPolyDataFilter()
         smoothingFilter.SetInputData(data_s)
         smoothingFilter.SetNumberOfIterations(Num_iterations_Windowed)
         smoothingFilter.NormalizeCoordinatesOn()
         #smoothingFilter.SetPassBand(1.) #default 0.1
         smoothingFilter.Update()
         data_s = smoothingFilter.GetOutput()
        else:
         smoother = vtk.vtkSmoothPolyDataFilter()
         smoother.SetInputData(data_s)
         smoother.SetNumberOfIterations(Num_iterations_Bsmooth )
         #smoother.FeatureEdgeSmoothingOn()
         #smoother.SetFeatureAngle(175.) #175
         smoother.Update()
         data_s = smoother.GetOutput()

        nodeid_s = VN.vtk_to_numpy(data_s.GetPointData().GetArray("NodeID"))
        for k in xrange(num_nodes_s):
          data_s.GetPoint(k, cord_pt)
          data_temp_s.GetPoint(k, cord_pt_temp)
          bsmooth_delta_x[ nodeid_s[k] ] = cord_pt[0] - cord_pt_temp[0]
          bsmooth_delta_y[ nodeid_s[k] ] = cord_pt[1] - cord_pt_temp[1]
          bsmooth_delta_z[ nodeid_s[k] ] = cord_pt[2] - cord_pt_temp[2]
    
        bsmooth_delta_x2 =  bsmooth_delta_x.flatten() #convert to n*1 array
        bsmooth_delta_x2 = bsmooth_delta_x2.astype('d') #convert to double
        disp_s1.vector().set_local(bsmooth_delta_x2[v2d_Vs])
        disp_s1.vector().apply('')
        bsmooth_delta_y2 =  bsmooth_delta_y.flatten() #convert to n*1 array
        bsmooth_delta_y2 = bsmooth_delta_y2.astype('d') #convert to double
        disp_s2.vector().set_local(bsmooth_delta_y2[v2d_Vs])
        disp_s2.vector().apply('')
        bsmooth_delta_z2 =  bsmooth_delta_z.flatten() #convert to n*1 array
        bsmooth_delta_z2 = bsmooth_delta_z2.astype('d') #convert to double
        disp_s3.vector().set_local(bsmooth_delta_z2[v2d_Vs])
        disp_s3.vector().apply('')
        for i in xrange(num_nodes):
          #if ( disp_nparray[3*i] > 1e-16 and disp_nparray[3*i+1] > 1e-16 and disp_nparray[3*i+2] > 1e-16):
          if (myfunc_disp.vector()[i] > Tol_smoothing):
              if (Flag_restrict_smoothing):
                disp_mag = ( pow(disp_nparray[3*i],2) + pow(disp_nparray[3*i+1],2) + pow(disp_nparray[3*i+2],2) )**0.5
                if (disp_mag > Thresh_restrict_smooth): #if not then just use the original disp_nparray (without smoothing)
                    disp_nparray[3*i] =    disp_s1.vector()[i]     #bsmooth_delta_x[i]
                    disp_nparray[3*i+1] =  disp_s2.vector()[i]    #bsmooth_delta_y[i]
                    disp_nparray[3*i+2] = disp_s3.vector()[i]  #bsmooth_delta_z[i]
              else:
                disp_nparray[3*i] =    disp_s1.vector()[i]     #bsmooth_delta_x[i]
                disp_nparray[3*i+1] =  disp_s2.vector()[i]    #bsmooth_delta_y[i]
                disp_nparray[3*i+2] = disp_s3.vector()[i]  #bsmooth_delta_z[i]
  
        #disp_nparray_final =   disp_nparray.flatten() #convert to n*1 array
        #disp_nparray_final = disp_nparray_final.astype('d') #convert to double
        #disp_vect.vector().set_local(disp_nparray_final[v2d_V])
        disp_vect.vector().set_local(disp_nparray[:])
        disp_vect.vector().apply('')
        #file_output_TEST = File (output_root + 'disp_vect_temp.pvd')
        #file_output_TEST << disp_vect
        #mesh = Mesh(mesh_copy)
        #disp_smoothed = solve_Laplace(mesh,disp_vect)
        disp_smoothed = solve_Laplace(mesh,disp_vect)
        ALE.move(mesh,disp_smoothed)
        #mesh = update_mesh(mesh,disp_smoothed)
        my_quality = MeshQuality.radius_ratio_min_max(mesh)
        print '------Min,Max quality (boundary_smoothed)',my_quality
        file_output2 << mesh


####################
total_timer.stop()
list_timings(TimingClear_keep, [TimingType_wall])
info_red('Total computing time = {0:f}'.format(total_timer.elapsed()[0]))
oasis_memory('Final memory use ')
total_initial_dolfin_memory = MPI.sum(mpi_comm_world(), initial_memory_use)
info_red('Memory use for importing dolfin = {} MB (RSS)'.format(
    total_initial_dolfin_memory))
info_red('Total memory use of solver = ' +
         str(oasis_memory.memory - total_initial_dolfin_memory) + " MB (RSS)")

# Final hook
theend_hook(**vars())
