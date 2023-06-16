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

### AAA P18
restart_folder = False
# Create a mesh
def mesh(**params):
    #m = Mesh('/scratch/aa3878/tube_oasis/P18vel_mesh.xml')
    #m = Mesh('/Users/aa3878/data/stenosis_ALE/tube_mesh/mesh-complete-mesh-complete/mesh-surfaces/tube_mesh.xml')
    #m = Mesh('/Users/aa3878/data/stenosis_ALE/Symon-models/C0_pz-mesh-complete/mesh-surfaces/coronary_mesh.xml')
    #m = Mesh('/Users/aa3878/data/stenosis_ALE/Symon-models/iso-mesh-complete/mesh-surfaces/coronary_mesh.xml')
    m =  Mesh('/Users/aa3878/data/stenosis_ALE/tube_final/results/meshindep/mesh90-mesh-complete/mesh-surfaces/tube_mesh.xml')
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
            max_iter=2,
            solver="IPCS_ABCN_1outlet",
            nu= 0.04/1.06, #4.0/1.06, #mg/ mm^3 and mg /mm^1 / s^1
            T= 0.12 + 0.03, #0.04 - 0.01, #0.84
            dt= 0.001,  #0.001/2., #assume Q = 500 mm^3/s => v = 131.6 mm/s (dt=0.001 works for O1)
            folder= '../stenosis_ALE/Symon-models/oasis_resuts',
            velocity_degree=1,
            save_step= 15, #1e9, #save in the code
            save_start=0, #2000,#4000,
            checkpoint=100*1e9,
            print_intermediate_info=10,
            nonNewtonian_flag = False,
            backflow_flag = False, #!!!! need to mannualy specify the outlet faces in IPCS_ABCN.py (currently faces 3 and 4 assumed outlet)
            beta_backflow = 0.02,#0.2, #1.0 , # btwn 0 and 1 (actually beta=2.0 and higher works!)
            Resistance_flag = True, #make sure to specify outlet faces in IPCS_ABCN pressure_solve
            flag_H5 =  True, #current MPI does not support h5 file (use Singularity Fenics)
            Res1 = 0., #240., #resistance at the first outlet
            Res2 = 0., #240., #resistance at the second outlet
            flag_wss = True,
            use_krylov_solvers=True,
            krylov_solvers=dict(
                                 monitor_convergence=False,
                                 report=False,
                                 error_on_nonconvergence=False,
                                 nonzero_initial_guess=True,
                                 maximum_iterations=200,
                                 relative_tolerance=1e-8,
                                 absolute_tolerance=1e-8),
                
           krylov_solvers_P=dict(
                                 monitor_convergence=False,
                                 report=False,
                                 error_on_nonconvergence=False,
                                 nonzero_initial_guess=True,
                                 maximum_iterations=200,
                                 relative_tolerance=1e-8,
                                 absolute_tolerance=1e-8),
                             )
        NS_parameters['krylov_solvers']['monitor_convergence'] = True
        NS_parameters['krylov_solvers']['report'] = True
        # NS_parameters['krylov_solvers']['error_on_nonconvergence'] = True
        NS_parameters['krylov_solvers_P']['monitor_convergence'] = True
        NS_parameters['krylov_solvers_P']['report'] = True
        #NS_parameters['krylov_solvers_P']['error_on_nonconvergence'] = True
        #set_log_level(ERROR)
        globals().update(NS_parameters)


# Changed by Mostafa: V_inletBC_x,V_inletBC_y,V_inletBC_z
def create_bcs(t,t_cycle,Time_array,v_IR_array, V, Q, facet_domains, mesh, **NS_namespace):
    # Specify boundary conditions
    
    #bc_file =  '/scratch/aa3878/18_oasis/BCnodeFacets.xml' #specify it in pre_solve_hook function below too.
    #bc_file =  '/home/mm4238/vul_plaque_project/biochem_tr_Budof/1A_2012_LAD/BCnodeFacets.xml'
    #facet_domains = MeshFunction('size_t', mesh,bc_file )
    parabolic_flag = True
    inlet_flag_flow = True #if True flowrate is input, else avg velocity
    inlet_facet_ID = 1
    #v_inlet_BC = numpy.interp(t_cycle, Time_array, v_IR_array)
    v_inlet_BC = 1.314 #cc/s  #500.0 #steady flow rate (mm^3/s)
    if (parabolic_flag):
        #Rotating the mesh so that the average normal vector aligned with z-axis passing the centroid
        d = mesh.geometry().dim()
        x = SpatialCoordinate(mesh)
        ds = Measure("ds")(subdomain_data=facet_domains)
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
        center = Point(numpy.array(center))
        # Compute average normal (assuming boundary is actually flat)
        n_normal= FacetNormal(mesh)
        ni = numpy.array([assemble(n_normal[i]*dsi) for i in xrange(d)])
        n_len = numpy.sqrt(sum([ni[i]**2 for i in xrange(d)])) # Should always be 1!?
        normal = ni/n_len
    
        bc_u2 = Expression("n0 * u * (1 - (pow(cent0-x[0], 2) + pow(cent1-x[1], 2) ) / r)",n0=normal[2], r=inlet_radius2, u=-2.0*v_inlet_BC, cent0=center[0],cent1=center[1],cent2=center[2], degree=3)

    if (parabolic_flag):
    
        #V_inletBC_x = Function(V,'/scratch/mm4238/utilities/BC_xml/bctxml_x.xml')
        #V_inletBC_y = Function(V,'/scratch/mm4238/utilities/BC_xml/bctxml_y.xml')
        #V_inletBC_z = Function(V,'/scratch/mm4238/utilities/BC_xml/bctxml_z.xml')
        bc_in_1  = DirichletBC(V, 0. , facet_domains,inlet_facet_ID) #inlet for u_x
        bc_in_2  = DirichletBC(V, 0. , facet_domains,inlet_facet_ID) #inlet for u_y
        bc_in_3  = DirichletBC(V, bc_u2 , facet_domains,inlet_facet_ID) #inlet for u_z
        #bc_in_1  = DirichletBC(V, bc_u0 , facet_domains,inlet_facet_ID) #inlet for u_x
        #bc_in_2  = DirichletBC(V, bc_u1 , facet_domains,inlet_facet_ID) #inlet for u_y
        #bc_in_3  = DirichletBC(V, bc_u2 , facet_domains,inlet_facet_ID) #inlet for u_z
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
    #bcp2  = DirichletBC(Q, 0. , facet_domains,4) #outlet p
    if MPI.rank(mpi_comm_world()) == 0:
        print '-----t = ', t
        print '-----t_cycle = ', t_cycle
    return dict(u0=[bc_in_1, bc_wall_1],
                    u1=[bc_in_2, bc_wall_2],
                    u2=[bc_in_3, bc_wall_3],
                    p=[bcp1])
    #return dict(u0=[bc_in_1, bc_wall_1],
    #           u1=[bc_in_2, bc_wall_2],
    #           u2=[bc_in_3, bc_wall_3],
    #           p =[]) # no pressure BC



def pre_solve_hook(mesh,facet_domains, velocity_degree, u_,
                   AssignedVectorFunction, **NS_namespace):
    #bc_file =   '/scratch/aa3878/tube_oasis/BCnodeFacets.xml'
    #if (jj==0):
    #  bc_file = '/Users/aa3878/data/stenosis_ALE/Symon-models/C0_pz-mesh-complete/mesh-surfaces/BCnodeFacets.xml'
    #  facet_domains = MeshFunction('size_t', mesh,bc_file )
    #normal
    n_normal = FacetNormal(mesh)
    #Time depndent inlet BC
    #read BC from AAA_BC.py
    Time_array = AAA_BC.time_BC()
    v_IR_array = AAA_BC.Vel_IR_BC()
    Time_last = Time_array[-1]
    if (True):
       #return dict(uv=AssignedVectorFunction(u_),  facet_domains= facet_domains, n_normal=n_normal,Time_array=Time_array,v_IR_array=v_IR_array,Time_last= Time_last )
       return dict(uv=AssignedVectorFunction(u_), facet_domains= facet_domains, n_normal=n_normal,Time_array=Time_array,v_IR_array=v_IR_array,Time_last= Time_last )
    else:
      return dict(uv=AssignedVectorFunction(u_),
                  n_normal=n_normal)


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
