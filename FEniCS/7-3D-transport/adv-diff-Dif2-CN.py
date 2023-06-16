from dolfin import *

#import vtk

import numpy as np

#import pylab

#import scipy

import time


#from vtk.util import numpy_support as VN

# TO RUN: mpirun -np #of_procs python filename.py

set_log_level(30)






stabilized = True
aneurysm_bc_type = 'neumann'

#!!!!!!! Uses 2 Diffusion coeficients. A very high value outside the region of interest to damp the soln near the outlets

parameters['form_compiler']['representation'] = 'quadrature'
parameters['form_compiler']['optimize'] = True
parameters['form_compiler']['cpp_optimize'] = True


Flag_XML = False #If True reads XML if not h5

root_dir = '/home/sci/amir.arzani/Python_tutorials/Fenics/3D_transport/'
results_dir = root_dir + 'results/'
mesh_filename = root_dir + 'P3vel_mesh.h5' #your velocity mesh in xml format
bc_file = root_dir + 'BCnodeFacets.h5' # the BCnodeFacets file where it stores the tags of BC IDs.
velocity_filename = root_dir + 'vel/P3_velocity.h5' 

file_conc_output = XDMFFile(results_dir + 'conc.xdmf') # the output name for your results.
file_conc_output.parameters["flush_output"] = True




velocity_start = 0 # the index for your first velocity file; so for example if velocity_filename above is = myvel_  and velocity_start =0 your file is: myvel_0.xml or in h5 file
velocity_stop = 49 #the last index of your vel file (end of cardiac cycle)
velocity_interval = 1 # difference between indices of two files


max_save_interval = 5  #This is how often (time-step) you want to write the max concentration (you need to monitor it for divergence)
solution_save_interval = 200 #how often (how many time-steps) do you want tosave the results
basis_order = 1 #1st order or 2nd baseis order. Start with 1 and then also try 2.

D = 4e-4 #cm^2/s #Diff coeff. In cardiovascular mass transport typically between 1e-4 cm^2/s - 1e-6 cm^2s. The higher it is the easier the code converge; if your unit is not cm, make sure you CONVERT!!!!

q_val = -D*10. #flux BC value  at the wall; This is arbitraty and for our BC here (constant Neuman at the wall) it will just scale the results

t_start = 0.0 # start time usually 0.
t_stop = 5 * 0.9 #(T=0.9) # Total time you run. You need to run at least 10 cardiac cycles for thin boundary layers to converge; so if T=0.8; it becomes 8.0


n_tsteps = 20000  #total number of time-steps you set it to get the desired dt below
tsteps_per_file = 82 #number of time steps btwn files; You should know the delta_t between two of your velocity files, so based on dt below you can calculate how many time-steps you have btwn two vel files.


dt = (t_stop - t_start) / n_tsteps  #Choose n_tsteps such that this is at least 0.0001 s or smaller

#!!!!!! Define the boundary condition face IDs below at DirichletBC( !!!!!!!!

############### Deefine parameters above here#############



D_multiple = D

if MPI.rank(MPI.comm_world) == 0:
    print ('reading mesh')

if (Flag_XML):
 mesh = Mesh(mesh_filename )  #xml
else:
 mesh = Mesh()
 mesh_in = HDF5File(MPI.comm_world, mesh_filename, 'r')
 mesh_in.read(mesh, 'mesh', False)
 mesh_in.close()
 velocity_in = HDF5File(MPI.comm_world, velocity_filename, 'r')
 velocity_prefix = '/velocity/vector_'





n_normal = FacetNormal(mesh)

 





#velocity_in = HDF5File(MPI.comm_world, velocity_filename, 'r')
#velocity_prefix = '/velocity/vector_'

#Initial Condition from file
IC_flag = False  #if True uses an I.C file
IC_file = root_dir + 'vel/concentration_5000.h5'
IC_start_index = 5000









h = 2*Circumradius(mesh)  # CellSize(mesh)







# Create FunctionSpaces

Q = FunctionSpace(mesh, 'CG', basis_order)
V = VectorFunctionSpace(mesh, 'CG', 1)



# Create mappings

# v2d_Q = dof_to_vertex_map(Q)
v2d_V = dof_to_vertex_map(V)



# Define functions

v = TestFunction(Q)
c = TrialFunction(Q)

c_prev = Function(Q)
c_sol = Function(Q)


velocity = Function(V)






flux = TrialFunction(Q)
phi = TestFunction(Q)
flux_sol = Function(Q)



if MPI.rank(MPI.comm_world) == 0:
    print ('reading BCmesh')

if (Flag_XML):
 facet_domains = MeshFunction('size_t', mesh,bc_file )
else:
 facet_domains = MeshFunction('size_t', mesh,mesh.topology().dim()-1)
 facet_domains_in = HDF5File(MPI.comm_world, bc_file, 'r')
 facet_domains_in.read(facet_domains, 'mesh_function')
 facet_domains_in.close()

#out_BC = File(results_dir + 'BC.pvd') #Write the BC and check to make sure the IDs match what is below

#out_BC << facet_domains




###Define DirichletBC here (set their IDs from the mesh)
#!!! Also set the id of the no-slip Wall for Neumann flux BC below: F += q * v * ds(2)
#1st try: just zero Dirichlet at the inlet so bc = [bc1]
#2nd tr: if outlet you have backflow and get divergence also set outlet to zero Dirichlet, so bc=[bc1,bc2]

bc1 = DirichletBC(Q, 0.0, facet_domains, 1) #inlet  Dirichlet.
#bc2 = DirichletBC(Q, 0.0, facet_domains, 3) #outlet1
#bc3 = DirichletBC(Q, 0.0, facet_domains, 4) #outlet2
#bc = [bc1,bc2,bc3]
bc = [bc1]





c_mid = 0.5 * (c + c_prev)


ds = Measure("ds")(subdomain_data=facet_domains)


#x_r = Expression('x[0]', element = Q.ufl_element())

#x_z = Expression('x[1]', element = Q.ufl_element())


F = v * (c - c_prev) / dt * dx \
    + v * dot(velocity ,grad(c_mid) ) * dx \
    + D_multiple * dot(grad(v), grad(c_mid)) * dx



F += q_val * v * ds(2)



if stabilized:
    res = (c - c_prev) / dt \
        + div(velocity * c_mid) \
            - div(D_multiple * grad(c_mid))



#     velocity_mag = inner(velocity, velocity)**.5

#     Pe = velocity_mag * h / 2. / D

#     tau_m = h / 2. / velocity_mag * Min(1., Pe / 3. / basis_order**2)


    tau_m = (4. / dt**2 \
         + dot(velocity, velocity) / h**2 \
         + 9. * basis_order**4 * D_multiple**2 / h**4)**(-.5)
    
    
    F += tau_m * res * inner(grad(v), velocity) * dx



a = lhs(F)

L = rhs(F)








tstep = 0

# Set initial condition
if (IC_flag == True):
 IC_in =  HDF5File(MPI.rank(MPI.comm_world), IC_file, 'r')
 IC_in.read(c_prev, '/concentration/vector_0')
 IC_in.close()
 tstep = IC_start_index
 t_start = IC_start_index * dt
 file_initial =  0       #(IC_tart_index * dt / T ) e.g if 7.4 then file initial = 0.4 * 50 (file initial = btwn 0 and 49 here)
 file_final = file_initial + velocity_interval
 if file_final > velocity_stop:
     file_final = velocity_start
 if MPI.rank(MPI.comm_world) == 0:
    print ('Using I.C.')
else:
    c_prev.assign(interpolate(Constant(0.0), Q)) #IC the same as inlet BC





t = t_start

file_initial = velocity_start

file_final = file_initial + velocity_interval

if file_final > velocity_stop:
    file_final = velocity_start



#out = HDF5File(MPI.comm_world,results_dir + 'concentration.h5','w')
#out_velocity = HDF5File(MPI.comm_world,results_dir + 'velocity.h5','w')

#out_max_vals = open(results_dir + 'max_vals.dat', 'w')




#out_bcs << facet_domains






#prm = parameters["krylov_solver"]
#prm["nonzero_initial_guess"] = True



# Time-stepping

start_time = time.clock()
problem = LinearVariationalProblem(a, L, c_sol, bc)
solver = LinearVariationalSolver(problem)
solver.parameters["linear_solver"] ="bicgstab"
#solver.parameters["preconditioner"] ="petsc_amg"
solver.parameters["preconditioner"] ="default"
solver.parameters['krylov_solver']['nonzero_initial_guess'] = True


#prm = parameters['krylov_solver'] # short form
#prm['absolute_tolerance'] = 1E-4  #default 1e-10
#prm['relative_tolerance'] = 1E-5 #default is 1e-6

#file_conc_output = File(results_dir + 'sn_conc.pvd')

if (not Flag_XML):
    vel_initial = Function(V)
    vel_final = Function(V)


if MPI.rank(MPI.comm_world) == 0:
    print ('Entering time loop',flush=True)

while tstep < n_tsteps:

     if (Flag_XML):
       vel_initial = Function(VectorFunctionSpace(mesh, 'CG', 1),velocity_filename +str(file_initial)+'.xml')
       vel_final = Function(VectorFunctionSpace(mesh, 'CG', 1),velocity_filename +str(file_final)+'.xml')
     else:
       velocity_in.read(vel_initial, velocity_prefix + str(file_initial))
       velocity_in.read(vel_final, velocity_prefix + str(file_final))
     if MPI.rank(MPI.comm_world) == 0:
        print ('Velocity read',flush=True)


     for tstep_inter in range(tsteps_per_file):
        tstep += 1
        if tstep > n_tsteps:
            break
        t += dt

           

        alpha = (tstep_inter * 1.0) / (tsteps_per_file * 1.0)

        velocity.vector().set_local((1. - alpha) * vel_initial.vector().get_local()+ alpha *  vel_final.vector().get_local())
        velocity.vector().apply('')

        #solve(a == L, c_sol, bc, solver_parameters={'linear_solver': 'bicgstab', 'preconditioner': 'hypre_euclid'}) #or gmres
        #solve(a == L, c_sol, bc, solver_parameters={'linear_solver': 'bicgstab', 'preconditioner': 'petsc_amg'}) #This one works!
        solver.solve()
       
        c_prev.assign(c_sol)
        #tmp = c_prev.vector().get_local()
        #tmp[tmp < 0.] = 0.
        #c_prev.vector().set_local(tmp)
        #c_prev.vector().apply('')



        # Plot and/or save solution

        if tstep % max_save_interval == 0:
            if aneurysm_bc_type == 'dirichlet':

                flux_form = -D * inner(grad(c_sol), n)

                LHS_flux = assemble(flux * phi * ds, keep_diagonal=True)

                RHS_flux = assemble(flux_form * phi * ds)

                LHS_flux.ident_zeros()

                solve(LHS_flux, flux_sol.vector(), RHS_flux)

                max_val = MPI.max(MPI.comm_world,
                                  np.amax(flux_sol.vector().get_local()))

                min_val = MPI.min(MPI.comm_world,
                                  np.amin(flux_sol.vector().get_local()))

                if MPI.rank(MPI.comm_world) == 0:

                    print ('time =', t, 'of', t_stop, '...', \
                          'Max=', max_val, \
                          'Min=', min_val, \
                          'Time elapsed:', (time.clock() - start_time) / 3600., \
                          'Time remaining:', 1. / 3600. * (n_tsteps-tstep) \
                                         * (time.clock()-start_time) / tstep, flush=True)

            #     out_max_vals.write('%g %g %g\n' % (t/1.0, max(flux_sol.vector().array()), min(flux_sol.vector().array())))

            else:

                max_val = MPI.max(MPI.comm_world,np.amax(c_sol.vector().get_local()))
                min_val = MPI.min(MPI.comm_world, np.amin(c_sol.vector().get_local()))


                if MPI.rank(MPI.comm_world) == 0:
                    print ('time =', t, 'of', t_stop, '...', \
                          'Max=', max_val, \
                          'Min=', min_val, \
                          'Time elapsed:', (time.clock() - start_time) / 3600., \
                          'Time remaining:', 1. / 3600. * (n_tsteps-tstep) \
                                         * (time.clock()-start_time) / tstep, flush=True)

                 #   out_max_vals.write('%g %g\n' % (t/1.0, max(c_sol.vector().array())))

        if ( tstep % solution_save_interval == 0  ) :
           #file_conc_output << c_sol
           file_conc_output.write(c_sol, t)

           if aneurysm_bc_type == 'dirichlet':
                flux_form = -D * inner(grad(c_sol), n)
                LHS_flux = assemble(flux * phi * ds, keep_diagonal=True)
                RHS_flux = assemble(flux_form * phi * ds)
                LHS_flux.ident_zeros()
                solve(LHS_flux, flux_sol.vector(), RHS_flux)
                out_flux.write(flux_sol,'flux_sol', tstep)




     file_initial = file_final

     file_final = file_initial + velocity_interval

     if file_final > velocity_stop:
        file_final = velocity_start



if MPI.rank(MPI.comm_world) == 0:
    print ('done',flush=True)

