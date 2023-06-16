__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-06-25"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from ..NSfracStep import *
import numpy
from numpy import cos, pi, cosh
from os import getcwd
import pickle

import AAA_BC

#What works (backflow): Order=2, beta=2. , dt =0.001/2
#Order=1, beta=100. (50.) dt =0.001
#Order=1, beta=1.  dt =0.001053 /2.,

### AAA P18
restart_folder = False
# Create a mesh
def mesh(**params):
    m = Mesh('/scratch/aa3878/18_oasis/final_longer/P18vel_mesh.xml')
    #m = Mesh('/Users/aa3878/data/nonNewtonian/AAA18/P18-mesh-complete/mesh-surfaces/P18vel_mesh.xml')
    return m


#def problem_parameters(commandline_kwargs, NS_parameters, **NS_namespace):
#if "restart_folder" in commandline_kwargs.keys():
if restart_folder:
        restart_folder = commandline_kwargs["restart_folder"]
        restart_folder = path.join(getcwd(), restart_folder)
        f = open(path.join(restart_folder, 'params.dat'), 'r')
        NS_parameters.update(pickle.load(f))
        NS_parameters['T'] = NS_parameters['T'] + 10 * NS_parameters['dt']
        NS_parameters['restart_folder'] = restart_folder
        globals().update(NS_parameters)

else:
        # Override some problem specific parameters
        NS_parameters.update(
            nu=0.04/1.06,
            T= 1.053*3, #0.84
            dt=0.001053 /2.,
            folder= '../../../../scratch/aa3878/18_oasis/final_longer/nonNewtO1',#'../nonNewtonian/AAA18/results_oasis/O1/nonNewt'
            velocity_degree=1,
            save_step=40,
            save_start=4000,
            checkpoint=100*1e9,
            print_intermediate_info=10,
            nonNewtonian_flag = True,
            backflow_flag = True, #!!!! need to mannualy specify the outlet faces in IPCS_ABCN.py (currently faces 3 and 4 assumed outlet)
            beta_backflow = 0.2, #1.0 , # btwn 0 and 1 (actually beta=2.0 and higher works!)
            Resistance_flag = True, #make sure to specify outlet faces in IPCS_ABCN pressure_solve
            flag_H5 = True, #current MPI does not support h5 file (use Singularity Fenics)
            Res1 = 240., #resistance at the first outlet
            Res2 = 240., #resistance at the second outlet
            flag_wss = True,
            use_krylov_solvers=True
                             )
        NS_parameters['krylov_solvers']['monitor_convergence'] = False
        #parameters["form_compiler"]["quadrature_degree"] = 2 #added by Amir (due to quadrature_degree warning: high number of integation points that happens for velocity_degree=2)
        #set_log_level(ERROR)
        globals().update(NS_parameters)

def create_bcs(facet_domains,t,t_cycle,Time_array,v_IR_array, V, Q, mesh, **NS_namespace):
    # Specify boundary conditions
    
    #bc_file =  '/scratch/aa3878/18_oasis/BCnodeFacets.xml' #specify it in pre_solve_hook function below too.
    #facet_domains = MeshFunction('size_t', mesh,bc_file ) #reading the mesh everytime will be very slow!!! xml file not scaling in parallel!!!!!!!
    parabolic_flag = True
    inlet_flag_flow = True #if True flowrate is input, else avg velocity
    inlet_facet_ID = 1
    if (parabolic_flag):
     if(0): #This does not work (See below)
        class BoundarySource(Expression):
            #def __init__(self, mesh):
            def __init__(self, **kwargs):
                #self.mesh = mesh
                #self._mesh = mesh
                self._mesh = kwargs["mesh"]
            #def eval_cell(self, values, x, ufc_cell):
            def eval(self, values,ufc_Cell):
                cell = Cell(self.mesh, ufc_cell.index)
                n = cell.normal(ufc_cell.local_facet)
                values[0] = n[0]
                values[1] = n[1]
                values[2] = n[2]
            def value_shape(self):
                return (3,)
        
        #create paraboloid profile
        ds = Measure("ds")[facet_domains]
        center = []
        #Area = Constant(assemble(1.0*m2*ds(1), mesh, facet_domains))
        CR = FunctionSpace(mesh, "CR", 1)
        V_vec = VectorFunctionSpace(mesh, "CG",1)
        m = TrialFunction(V_vec)
        inflow_profile = Function(V_vec)
        Area = interpolate(Constant(1.0), CR)
        p = project(Expression(('x[0]', 'x[1]', 'x[2]'),degree=3), V_vec,solver_type='cg')
        #for i in range(3):
        #    #center.append(assemble(p[i]*ds(1), mesh, facet_domains)/Area)
        #    center.append( assemble( p[i]/ Area * ds(1) ) )
        center = numpy.array((0., 0., 0.))
        r = sqrt(Area/DOLFIN_PI)      # radius
        n_normal = FacetNormal(mesh)
        #test = BoundarySource(degree=3,mesh=mesh) #test see if it works
        #assemble(inner(BoundarySource(degree=3,mesh=mesh),m)*ds(1), inflow_profile.vector(), facet_domains)
        #inflow_profile = assemble( inner(BoundarySource(degree=3,mesh=mesh),m)*ds(1) )
        inflow_profile = BoundarySource(degree=3,mesh=mesh)
        #N = int(len(inflow_profile.vector()))
        N = int(len(inflow_profile))
        for i in range(int(N/3)):
            #n1 = inflow_profile.vector()[i]
            #n2 = inflow_profile.vector()[int(N/3)+i]
            #n3 = inflow_profile.vector()[int(2*N/3)+i]
            n1 = inflow_profile[i]
            n2 = inflow_profile[int(N/3)+i]
            n3 = inflow_profile[int(2*N/3)+i]
            norm = 1. #numpy.sqrt(n1**2+n2**2+n3**2)
            if(norm > 0):
                x = p.vector()[i]
                y = p.vector()[N/3+i]
                z = p.vector()[2*N/3+i]
                d_c = numpy.sqrt((center[0]-x)**2+(center[1]-y)**2+(center[2]-z)**2)/r
                #inflow_profile.vector()[i]   = n1/norm*(1-d_c**2)
                #inflow_profile.vector()[N/3+i]   = n2/norm*(1-d_c**2)
                #inflow_profile.vector()[2*N/3+i] = n3/norm*(1-d_c**2)
                inflow_profile[i]   = n1/norm*(1-d_c**2)
                inflow_profile[N/3+i]   = n2/norm*(1-d_c**2)
                inflow_profile[2*N/3+i] = n3/norm*(1-d_c**2)
   
   
     #inflow = Expression('-5.0*sin(2.0*t*pi)',t=0,degree=3) # inflow.t needs to be updated to change time
     #inflow.t = t #update the t  variable in inflow
     v_inlet_BC = numpy.interp(t_cycle, Time_array, v_IR_array)
     if(1): #This works!! (from Womersley.py)
         d = mesh.geometry().dim()
         x = SpatialCoordinate(mesh)
         ds = Measure("ds")[facet_domains]
         dsi = ds(inlet_facet_ID, domain=mesh, subdomain_data=facet_domains)
         # Compute area of boundary tesselation by integrating 1.0 over all facets
         A = assemble(Constant(1.0, name="one")*dsi)
         if MPI.rank(mpi_comm_world()) == 0:
                 print 'Inlet area:', A
         if (inlet_flag_flow):
             v_inlet_BC = v_inlet_BC / A
         inlet_radius = numpy.sqrt(A / pi) # This old estimate is a few % lower because of boundary discretization errors
         inlet_radius2 = A / pi #It should be r**2 in the parabolic eqn below!
         if MPI.rank(mpi_comm_world()) == 0:
             print 'Inlet radius:', inlet_radius
        # Compute barycenter by integrating x components over all facets
         center = [assemble(x[i]*dsi) / A for i in xrange(d)]
         if MPI.rank(mpi_comm_world()) == 0:
                 print 'Inlet center:', center
         # Compute average normal (assuming boundary is actually flat)
         n_normal= FacetNormal(mesh)
         ni = numpy.array([assemble(n_normal[i]*dsi) for i in xrange(d)])
         n_len = numpy.sqrt(sum([ni[i]**2 for i in xrange(d)])) # Should always be 1!?
         normal = ni/n_len
         if(0):
          bc_u0 = Expression("n0 * u * (1 - (pow(cent0-x[0], 2) + pow(cent1-x[1], 2) + pow(cent2-x[2], 2)   ) / r)",n0=normal[0], r=inlet_radius2, u=-2.0*v_inlet_BC, cent0=center[0],cent1=center[1],cent2=center[2], degree=3)
          bc_u1 = Expression("n0 * u * (1 - (pow(cent0-x[0], 2) + pow(cent1-x[1], 2)+ pow(cent2-x[2], 2)  ) / r)",n0=normal[1], r=inlet_radius2, u=-2.0*v_inlet_BC, cent0=center[0],cent1=center[1],cent2=center[2], degree=3)
          bc_u2 = Expression("n0 * u * (1 - (pow(cent0-x[0], 2) + pow(cent1-x[1], 2)+ pow(cent2-x[2], 2) ) / r)",n0=normal[2], r=inlet_radius2, u=-2.0*v_inlet_BC, cent0=center[0],cent1=center[1],cent2=center[2], degree=3)
         else:#we know that the inlet face is in Z direction
           bc_u0 = Expression("n0 * u * (1 - (pow(cent0-x[0], 2) + pow(cent1-x[1], 2) ) / r)",n0=normal[0], r=inlet_radius2, u=-2.0*v_inlet_BC, cent0=center[0],cent1=center[1],cent2=center[2], degree=3)
           bc_u1 = Expression("n0 * u * (1 - (pow(cent0-x[0], 2) + pow(cent1-x[1], 2) ) / r)",n0=normal[1], r=inlet_radius2, u=-2.0*v_inlet_BC, cent0=center[0],cent1=center[1],cent2=center[2], degree=3)
           bc_u2 = Expression("n0 * u * (1 - (pow(cent0-x[0], 2) + pow(cent1-x[1], 2) ) / r)",n0=normal[2], r=inlet_radius2, u=-2.0*v_inlet_BC, cent0=center[0],cent1=center[1],cent2=center[2], degree=3)
    if (parabolic_flag):
        bc_in_1  = DirichletBC(V, bc_u0 , facet_domains,inlet_facet_ID) #inlet for u_x
        bc_in_2  = DirichletBC(V, bc_u1 , facet_domains,inlet_facet_ID) #inlet for u_y
        bc_in_3  = DirichletBC(V, bc_u2 , facet_domains,inlet_facet_ID) #inlet for u_z
    else:
        inflow = Expression('-vel_IR',vel_IR=0,degree=3)
        inflow.vel_IR = v_inlet_BC
        bc_in_1  = DirichletBC(V, 0. , facet_domains,inlet_facet_ID) #inlet for u_x
        bc_in_2  = DirichletBC(V, 0. , facet_domains,inlet_facet_ID) #inlet for u_y
        bc_in_3  = DirichletBC(V,  inflow , facet_domains,inlet_facet_ID) #inlet for u_z
    bc_wall_1  = DirichletBC(V, 0. , facet_domains,2) #wall for u_x
    bc_wall_2  = DirichletBC(V, 0. , facet_domains,2) #wall for u_y
    bc_wall_3  = DirichletBC(V, 0. , facet_domains,2) #wall for u_z
    bcp1  = DirichletBC(Q, 0. , facet_domains,3) #outlet p
    bcp2  = DirichletBC(Q, 0. , facet_domains,4) #outlet p
    if MPI.rank(mpi_comm_world()) == 0:
      print '-----t = ', t
      print '-----t_cycle = ', t_cycle
    return dict(u0=[bc_in_1, bc_wall_1],
                u1=[bc_in_2, bc_wall_2],
                u2=[bc_in_3, bc_wall_3],
                p=[bcp1, bcp2])
    #return dict(u0=[bc_in_1, bc_wall_1],
    #           u1=[bc_in_2, bc_wall_2],
    #           u2=[bc_in_3, bc_wall_3],
    #           p =[]) # no pressure BC

def pre_solve_hook(mesh, velocity_degree, u_,
                   AssignedVectorFunction, **NS_namespace):
    bc_file =   '/scratch/aa3878/18_oasis/final_longer/BCnodeFacets.xml'
    #bc_file =  '/Users/aa3878/data/nonNewtonian/AAA18/P18-mesh-complete/mesh-surfaces/BCnodeFacets.xml'
    facet_domains = MeshFunction('size_t', mesh,bc_file )
    #normal
    n_normal = FacetNormal(mesh)
    #Time depndent inlet BC
    #read BC from AAA_BC.py
    Time_array = AAA_BC.time_BC()
    v_IR_array = AAA_BC.Vel_IR_BC()
    Time_last = Time_array[-1]
    
    return dict(uv=AssignedVectorFunction(u_),  facet_domains= facet_domains, n_normal=n_normal,Time_array=Time_array,v_IR_array=v_IR_array,Time_last= Time_last )


def temporal_hook(tstep, uv, p_, plot_interval, **NS_namespace):
    if(0):
     if tstep % plot_interval == 0:
        uv()
        plot(uv, title='Velocity')
        plot(p_, title='Pressure')


def theend_hook(p_, uv, **NS_namespace):
   if(0):
    uv()
    plot(uv, title='Velocity')
    plot(p_, title='Pressure')
