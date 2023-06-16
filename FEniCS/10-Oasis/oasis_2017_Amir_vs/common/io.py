__author__ = "Mikael Mortensen <mikaem@math.uio.no>"
__date__ = "2013-11-26"
__copyright__ = "Copyright (C) 2013 " + __author__
__license__ = "GNU Lesser GPL version 3 or any later version"

from os import makedirs, getcwd, listdir, remove, system, path
import pickle
import six
#from dolfin import (MPI, Function, XDMFFile, HDF5File, info_red, VectorFunctionSpace, mpi_comm_world, FunctionAssigner,File)
from dolfin import * #changed by Amir to calculate wss
#import vtk #do we really need vtk? (commented for testing Singularity)

__all__ = ["create_initial_folders", "save_solution", "save_tstep_solution_h5",
           "save_checkpoint_solution_h5", "check_if_kill", "check_if_reset_statistics",
           "init_from_restart"]


def create_initial_folders(folder, restart_folder, sys_comp, tstep, info_red,
                           scalar_components, output_timeseries_as_vector,
                           **NS_namespace):
    """Create necessary folders."""
    info_red("Creating initial folders")
    # To avoid writing over old data create a new folder for each run
    if MPI.rank(mpi_comm_world()) == 0:
        try:
            makedirs(folder)
        except OSError:
            pass

    MPI.barrier(mpi_comm_world())
    newfolder = path.join(folder, 'data')
    if restart_folder:
        newfolder = path.join(newfolder, restart_folder.split('/')[-2])
    else:
        if not path.exists(newfolder):
            newfolder = path.join(newfolder, '1')
        else:
            previous = listdir(newfolder)
            previous = max(map(eval, previous)) if previous else 0
            newfolder = path.join(newfolder, str(previous + 1))

    MPI.barrier(mpi_comm_world())
    if MPI.rank(mpi_comm_world()) == 0:
        if not restart_folder:
            #makedirs(path.join(newfolder, "Voluviz"))
            #makedirs(path.join(newfolder, "Stats"))
            #makedirs(path.join(newfolder, "VTK"))
            makedirs(path.join(newfolder, "Timeseries"))
            makedirs(path.join(newfolder, "Checkpoint"))

    tstepfolder = path.join(newfolder, "Timeseries")
    tstepfiles = {}
    comps = sys_comp
    if output_timeseries_as_vector:
        comps = ['p', 'u'] + scalar_components

    for ui in comps:
        tstepfiles[ui] = XDMFFile(mpi_comm_world(), path.join(
            tstepfolder, ui + '_from_tstep_{}.xdmf'.format(tstep)))
        tstepfiles[ui].parameters["rewrite_function_mesh"] = False
        tstepfiles[ui].parameters["flush_output"] = True
    
    #tstepfiles_wss = XDMFFile(mpi_comm_world(), path.join(tstepfolder, 'wss_from_tstep_{}.xdmf'.format(tstep)))
    tstepfiles_wss = XDMFFile(mpi_comm_world(), path.join(tstepfolder, 'wss_from_tstep_0.xdmf'))
    tstepfiles_wss.parameters["rewrite_function_mesh"] = False
    tstepfiles_wss.parameters["flush_output"] = True
    return newfolder, tstepfiles,tstepfiles_wss


def save_solution(save_start,mesh,visc_nu,flag_wss,flag_H5,tstep, t, q_, q_1, folder, newfolder, save_step, checkpoint,
                  NS_parameters, tstepfiles,tstepfiles_wss, u_, u_components, scalar_components,
                  output_timeseries_as_vector, constrained_domain,
                  AssignedVectorFunction, **NS_namespace):
    """Called at end of timestep. Check for kill and save solution if required."""
    NS_parameters.update(t=t, tstep=tstep)
    if (tstep >= save_start):
     if (tstep % save_step == 0):
      if (flag_wss):#save wss results
          n_normal = FacetNormal(mesh)
          V_boundary = VectorFunctionSpace(mesh, 'CG', 1)
          u_wss = TrialFunction(V_boundary)
          v_wss = TestFunction(V_boundary)
          wss = Function(V_boundary,name="wss")
          T1 = -visc_nu * 1.06 * dot((grad(u_) + grad(u_).T), n_normal)
          Tn = dot(T1, n_normal)
          Tt = T1 - Tn*n_normal
          LHS_wss = assemble(inner(u_wss, v_wss) * ds, keep_diagonal=True)
          RHS_wss = assemble(inner(Tt, v_wss) * ds)
          LHS_wss.ident_zeros()
          solve(LHS_wss, wss.vector(), RHS_wss,'gmres', 'default')
          if(0):#testing viscossity value
              V_visc = FunctionSpace(mesh, 'CG', 1)
              u_vis = TrialFunction(V_visc)
              v_vis = TestFunction(V_visc)
              visc = Function(V_visc,name="mu")
              solve( u_vis*v_vis*dx == 1.06*visc_nu*v_vis*dx, visc,solver_parameters={'linear_solver': 'gmres'})
              file_output = File(newfolder +'/Timeseries/'+  'visc_result_' + str(tstep)+ '.pvd')
              file_output <<  visc
              
      if (flag_H5):
          save_tstep_solution_h5(tstep, q_, u_, newfolder, tstepfiles, constrained_domain,
                               output_timeseries_as_vector, u_components, AssignedVectorFunction,
                               scalar_components, NS_parameters)
          if (flag_wss):
            tstepfiles_wss.write(wss, float(tstep))
      else:
         u_soln = AssignedVectorFunction(u_)
         u_soln()
         file_output = File(newfolder +'/Timeseries/'+  'velocity_result_' + str(tstep)+ '.pvd')
         file_output <<  u_soln
         file_outputP = File(newfolder + '/Timeseries/'+ 'pressure_result_' + str(tstep)+ '.pvd')
         file_outputP << q_['p']
         if (flag_wss):
          file_output_w = File(newfolder +'/Timeseries/'+  'WSS_result_' + str(tstep)+ '.pvd')
          file_output_w <<  wss

    killoasis = check_if_kill(folder)
    if tstep % checkpoint == 0 or killoasis:
        if (flag_H5):
          save_checkpoint_solution_h5(tstep, q_, q_1, newfolder, u_components,
                                    NS_parameters)

    return killoasis


def save_tstep_solution_h5(tstep, q_, u_, newfolder, tstepfiles, constrained_domain,
                           output_timeseries_as_vector, u_components, AssignedVectorFunction,
                           scalar_components, NS_parameters):
    """Store solution on current timestep to XDMF file."""
    timefolder = path.join(newfolder, 'Timeseries')
    if output_timeseries_as_vector:
        # project or store velocity to vector function space
        for comp, tstepfile in six.iteritems(tstepfiles):
            if comp == "u":
                V = q_['u0'].function_space()
                # First time around create vector function and assigners
                if not hasattr(tstepfile, 'uv'):
                    tstepfile.uv = AssignedVectorFunction(u_)

                # Assign solution to vector
                tstepfile.uv()

                # Store solution vector
                tstepfile.write(tstepfile.uv, float(tstep))

            elif comp in q_:
                tstepfile.write(q_[comp], float(tstep))

            else:
                tstepfile.write(tstepfile.function, float(tstep))

    else:
        for comp, tstepfile in six.iteritems(tstepfiles):
            tstepfile << (q_[comp], float(tstep))

    if MPI.rank(mpi_comm_world()) == 0:
        if not path.exists(path.join(timefolder, "params.dat")):
            f = open(path.join(timefolder, 'params.dat'), 'wb')
            pickle.dump(NS_parameters,  f)


def save_checkpoint_solution_h5(tstep, q_, q_1, newfolder, u_components,
                                NS_parameters):
    """Overwrite solution in Checkpoint folder.

    For safety reasons, in case the solver is interrupted, take backup of
    solution first.

    Must be restarted using the same mesh-partitioning. This will be fixed
    soon. (MM)

    """
    checkpointfolder = path.join(newfolder, "Checkpoint")
    NS_parameters["num_processes"] = MPI.size(mpi_comm_world())
    if MPI.rank(mpi_comm_world()) == 0:
        if path.exists(path.join(checkpointfolder, "params.dat")):
            system('cp {0} {1}'.format(path.join(checkpointfolder, "params.dat"),
                                       path.join(checkpointfolder, "params_old.dat")))
        f = open(path.join(checkpointfolder, "params.dat"), 'wb')
        pickle.dump(NS_parameters,  f)

    MPI.barrier(mpi_comm_world())
    for ui in q_:
        h5file = path.join(checkpointfolder, ui + '.h5')
        oldfile = path.join(checkpointfolder, ui + '_old.h5')
        # For safety reasons...
        if path.exists(h5file):
            if MPI.rank(mpi_comm_world()) == 0:
                system('cp {0} {1}'.format(h5file, oldfile))
        MPI.barrier(mpi_comm_world())
        ###
        newfile = HDF5File(mpi_comm_world(), h5file, 'w')
        newfile.flush()
        newfile.write(q_[ui].vector(), '/current')
        if ui in u_components:
            newfile.write(q_1[ui].vector(), '/previous')
        if path.exists(oldfile):
            if MPI.rank(mpi_comm_world()) == 0:
                system('rm {0}'.format(oldfile))
        MPI.barrier(mpi_comm_world())
    if MPI.rank(mpi_comm_world()) == 0 and path.exists(path.join(checkpointfolder, "params_old.dat")):
        system('rm {0}'.format(path.join(checkpointfolder, "params_old.dat")))


def check_if_kill(folder):
    """Check if user has put a file named killoasis in folder."""
    found = 0
    if 'killoasis' in listdir(folder):
        found = 1
    collective = MPI.sum(mpi_comm_world(), found)
    if collective > 0:
        if MPI.rank(mpi_comm_world()) == 0:
            remove(path.join(folder, 'killoasis'))
            info_red('killoasis Found! Stopping simulations cleanly...')
        return True
    else:
        return False


def check_if_reset_statistics(folder):
    """Check if user has put a file named resetoasis in folder."""
    found = 0
    if 'resetoasis' in listdir(folder):
        found = 1
    collective = MPI.sum(mpi_comm_world(), found)
    if collective > 0:
        if MPI.rank(mpi_comm_world()) == 0:
            remove(path.join(folder, 'resetoasis'))
            info_red('resetoasis Found!')
        return True
    else:
        return False


def init_from_restart(restart_folder, sys_comp, uc_comp, u_components,
                      q_, q_1, q_2, **NS_namespace):
    """Initialize solution from checkpoint files """
    if restart_folder:
        for ui in sys_comp:
            filename = path.join(restart_folder, ui + '.h5')
            hdf5_file = HDF5File(mpi_comm_world(), filename, "r")
            hdf5_file.read(q_[ui].vector(), "/current", False)
            q_[ui].vector().apply('insert')
            # Check for the solution at a previous timestep as well
            if ui in uc_comp:
                q_1[ui].vector().zero()
                q_1[ui].vector().axpy(1., q_[ui].vector())
                q_1[ui].vector().apply('insert')
                if ui in u_components:
                    hdf5_file.read(q_2[ui].vector(), "/previous", False)
                    q_2[ui].vector().apply('insert')
