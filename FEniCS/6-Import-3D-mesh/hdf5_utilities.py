from dolfin import *
import numpy as np
import vtk
from vtk.util import numpy_support as VN

#For creating files to be used in FENICS with paralell
def xml_mesh_to_hdf5(xml_filename, hdf5_filename):
    mesh = Mesh(xml_filename)
    mesh_out = HDF5File(MPI.comm_world, hdf5_filename, 'w')
    mesh_out.write(mesh, 'mesh')
    mesh_out.close()

#For BC:
def xml_mesh_function_to_hdf5(mesh_filename, # Must be .h5
                         xml_filename,
                         hdf5_filename,
                         function_name='mesh_function'):
    mesh = Mesh()
    mesh_file = HDF5File(MPI.comm_world, mesh_filename, 'r')
    mesh_file.read(mesh, 'mesh', False)
    mesh_file.close()

    facet_domains = MeshFunction('size_t', mesh, xml_filename)

    fout = HDF5File(MPI.comm_world, hdf5_filename, 'w')
    fout.write(facet_domains, function_name)
    fout.close()
#For vel:
def bin_to_hdf5(mesh_filename,
                hdf5_filename,
                bin_root,
                bin_start,
                bin_stop,
                bin_interval,
                hdf5_start=0,
                hdf5_interval=1,
                fieldname='velocity',
                is_vector=True):
    mesh = Mesh()
    mesh_file = HDF5File(MPI.comm_world, mesh_filename, 'r')
    mesh_file.read(mesh, 'mesh', False)
    mesh_file.close()

    if is_vector:
        V = VectorFunctionSpace(mesh, 'CG', 1)
    else:
        V = FunctionSpace(mesh, 'CG', 1)

    d2v = dof_to_vertex_map(V)
    v = Function(V)

    hdf5_out = HDF5File(MPI.comm_world, hdf5_filename, 'w')

    hdf5_index = hdf5_start
    for i in range(bin_start, bin_stop + 1, bin_interval):
        bin_filename = bin_root + str(i) + '.bin'
        print ('Writing', bin_filename, 'to', hdf5_filename)
        function_in = np.fromfile(bin_filename)
        v.vector()[:] = function_in[d2v]
        hdf5_out.write(v, fieldname, hdf5_index)
        hdf5_index += hdf5_interval
    hdf5_out.close()

#For vel:
def vtk_to_hdf5(mesh_filename,
                hdf5_filename,
                bin_root,
                bin_start,
                bin_stop,
                bin_interval,
                hdf5_start=0,
                hdf5_interval=1,
                fieldname='velocity',
                is_vector=True):
    mesh = Mesh()
    mesh_file = HDF5File(MPI.comm_world, mesh_filename, 'r')
    mesh_file.read(mesh, 'mesh', False)
    mesh_file.close()
    
    if is_vector:
        V = VectorFunctionSpace(mesh, 'CG', 1)
    else:
        V = FunctionSpace(mesh, 'CG', 1)

    d2v = dof_to_vertex_map(V)
    v = Function(V)

    hdf5_out = HDF5File(MPI.comm_world, hdf5_filename, 'w')
    
    hdf5_index = hdf5_start
    for i in range(bin_start, bin_stop + 1, bin_interval):
        bin_filename = bin_root + str(i) + '.vtu'
        print ('Writing', bin_filename, 'to', hdf5_filename)
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(bin_filename)
        reader.Update()
        data = reader.GetOutput()
        VEL  = data.GetPointData().GetArray('velocity')
        w_final = VN.vtk_to_numpy(VEL)
        w_final = w_final.flatten() #convert to n*1 array
        w_final = w_final.astype('d') #convert to double
        v.vector()[:] = w_final[d2v]
        hdf5_out.write(v, fieldname, hdf5_index)
        hdf5_index += hdf5_interval
    hdf5_out.close()

def hd5f_to_vtk(mesh_filename,hdf5_filename_vel,hdf5_start,hdf5_stop,hdf5_interval,VTKresults_dir,is_vector,basis_order):

    mesh = Mesh()
    mesh_file = HDF5File(MPI.comm_world, mesh_filename, 'r')
    mesh_file.read(mesh, 'mesh', False)
    mesh_file.close()
    if is_vector:
        V = VectorFunctionSpace(mesh, 'CG', 1)
    else:
        V = FunctionSpace(mesh, 'CG', basis_order)
        #V = FunctionSpace(mesh, 'DG', 0)  #For diffusion coefficient
        print ('concentration!')
    #d2v = dof_to_vertex_map(V)
    #v = Function(V)
    myvelocity = Function(V)


    if (is_vector):
      velocity_prefix = '/velocity/vector_0'
    else:
      velocity_prefix = '/concentration/vector_0'
#velocity_prefix = 'D'

    for i in range(hdf5_start, hdf5_stop + 1, hdf5_interval):
      hdf5_filename_temporal = hdf5_filename_vel + str(i) + '.h5'
      #hdf5_filename_temporal = hdf5_filename_vel
      velocity_in = HDF5File(MPI.comm_world, hdf5_filename_temporal, 'r')
      print ('Reading ', hdf5_filename_temporal)
      velocity_in.read(myvelocity, velocity_prefix)
      if (is_vector):
        out_file = File(VTKresults_dir + 'velocity_' + str(i) + '.pvd')
      else:
        out_file = File(VTKresults_dir + 'concentration_' + str(i) + '.pvd')
#out_file = File(VTKresults_dir + 'D' + str(i) + '.pvd')
      out_file << myvelocity
      velocity_in.close()




if __name__ == '__main__':
    old_root = '/home/sci/amir.arzani/Python_tutorials/Fenics/3D_transport/'
    vel_root_in = '/home/sci/amir.arzani/Python_tutorials/Fenics/3D_transport/vel/'
    vel_root_out = vel_root_in 
    xml_mesh_filename = old_root + 'Volume_mesh.xml'
    xml_filename_BC =  old_root + 'BCnodeFacets.xml'
    hdf5_mesh_filename =  old_root + 'P3vel_mesh.h5'
    
    hdf5_filename_vel = '/home/sci/amir.arzani/Python_tutorials/Fenics/3D_transport/vel/'
    hdf5_start =10100
    hdf5_stop = 15000
    hdf5_interval = 100
    VTKresults_dir = '/home/sci/amir.arzani/Python_tutorials/Fenics/3D_transport/vel/'
    basis_order = 1
    
 
 
 
 
    xml_mesh_to_hdf5(xml_mesh_filename, hdf5_mesh_filename)
    
    #bin_to_hdf5(hdf5_mesh_filename,
    #            new_root + '02_Velocity_Data/velocity.h5',
    #            old_root + '02_Adv_Diff/01_Velocity_Fields/velocity_field.',
    #            199,
    #            298,
    #            1)
    
    #BC:
    xml_mesh_function_to_hdf5(hdf5_mesh_filename, xml_filename_BC,old_root + 'BCnodeFacets.h5', function_name='mesh_function')
    vtk_to_hdf5(hdf5_mesh_filename, vel_root_out + 'P3_velocity.h5', vel_root_in + 'all_results_',10100,15000,100 )
    #VECTOR = False
    #hd5f_to_vtk(hdf5_mesh_filename,hdf5_filename_vel,hdf5_start,hdf5_stop,hdf5_interval,VTKresults_dir,VECTOR, basis_order)




