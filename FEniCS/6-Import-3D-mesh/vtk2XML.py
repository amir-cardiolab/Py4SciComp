from dolfin import *
import vtk
import math
from vtk.util import numpy_support as VN
#converts vtu to xml. For vtk remove XML in vtkunstructured reader

N_files = 51
delta_file = 20
file_start = 3000
root_dir = '/Users/symon/Data/Research/Residence_Time/ICA42/'
#Mesh_file_xml = root_dir + 'P18actext-mesh-complete/mesh-surfaces/P18highres_mesh.xml'
Mesh_file_xml = root_dir + 'Mesh/Carotid-mesh.xml'
#velocity_filename = root_dir + 'vel/P18vel_advdif-'
velocity_filename = root_dir + 'Velocity/ICA42_vel_005_'
array_fieldname = 'velocity'
#output_name = root_dir + 'vel/P18vel_advdif-'
output_name = root_dir + 'Velocity/ICA42_vel_005_'

file_str = file_start

print ('reading..' , Mesh_file_xml)
bmesh = Mesh(Mesh_file_xml)  #xml
Vb_vector_CG1 = VectorFunctionSpace(bmesh, 'CG', 1)
v2d_V = dof_to_vertex_map(Vb_vector_CG1)
Velocity_f = Function(Vb_vector_CG1)

for i in xrange(N_files): #xrange instead of range since it uses a lot less memory

 print ('processing..', velocity_filename +str(file_str))
 reader = vtk.vtkUnstructuredGridReader() #what format to read
 reader.SetFileName(velocity_filename +str(file_str)+'.vtk') #what is the name
 reader.Update() #just reads
 data = reader.GetOutput() #saves read data in a file
 #print data
 
 VEL  = data.GetPointData().GetArray(array_fieldname) #saves node values (here we have vectors) GetArray would probably be for that reason! NOT really! data was read
 #print VEL
 
 vel_array = VN.vtk_to_numpy(VEL) #change format from vtk to numpy
 #print vel_array

 vel_array_final =  vel_array.flatten() #convert to n*1 array
 #print vel_array_final
 #vel_array_final = vel_array_final.astype('d') #convert to double (It already works without this line!)
 Velocity_f.vector().set_local(vel_array_final[v2d_V]) #put values on 
 output_filename_xml = output_name + str(file_str)+'.xml'
 File(output_filename_xml) <<  Velocity_f

 
 file_str = file_str + delta_file




print ('done')
                                
                                   
                                   



