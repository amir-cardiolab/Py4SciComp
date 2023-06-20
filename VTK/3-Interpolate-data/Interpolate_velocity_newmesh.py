import numpy
import sys
import vtk
import glob
import re
import math
from vtk.util import numpy_support as VN
def interp(input_filename, output_filename, fieldname, T_first, T_delta, T,input_new_mesh,input_file_wall):

 print ('Loading', input_new_mesh)
 reader = vtk.vtkXMLUnstructuredGridReader()
 reader.SetFileName(input_new_mesh)
 reader.Update()
 data_iso = reader.GetOutput()
 n_points_iso = data_iso.GetNumberOfPoints()
 t_index = T_first - T_delta
 print ('Loading no-slip wall', input_file_wall)
 reader = vtk.vtkXMLPolyDataReader()
 reader.SetFileName(input_file_wall)
 reader.Update()
 data_wall = reader.GetOutput()
 n_points_wall = data_wall.GetNumberOfPoints()
 wall_IDs = VN.vtk_to_numpy(data_wall.GetPointData().GetArray('GlobalNodeID'))
 data_iso_IDs = VN.vtk_to_numpy(data_iso.GetPointData().GetArray('GlobalNodeID'))

 Wall_ID_check = numpy.zeros( numpy.max(data_iso_IDs) +1 ) #1 if that ID is on wall. plus one because array starts at 0 and ID at 1

 for i in range(n_points_wall):
     Wall_ID_check[wall_IDs[i]] = 1 #tag with one if on wall
 
 ind = T_first
 for t in range(T):
     input_filename2 = input_filename + str(ind) + '.vtu'
     print ('Loading', input_filename2)
     reader = vtk.vtkXMLUnstructuredGridReader()
     reader.SetFileName(input_filename2)
     reader.Update()
     data = reader.GetOutput()
     n_points = data.GetNumberOfPoints()
     print ('n_points old, n_points new' ,n_points, n_points_iso)
     # data.GetPointData().RemoveArray('velocity')
     Vel = VN.vtk_to_numpy(data.GetPointData().GetArray(fieldname))
     if t==0:  #do the interpolation once
         VTKpoints = vtk.vtkPoints()
         for i in range(n_points_iso):
             pt_iso  =  data_iso.GetPoint(i)
             VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])
         point_data = vtk.vtkUnstructuredGrid()
         point_data.SetPoints(VTKpoints)
         #find the cell
         #cellLocator = vtk.vtkCellLocator()
         #cellLocator.SetDataSet(data)
         #cellLocator.BuildLocator()
         #cellID =cellLocator.FindCell(pt_iso)
     probe = vtk.vtkProbeFilter()
     probe.SetInputData(point_data)
     probe.SetSourceData(data)
     probe.Update()
     array = probe.GetOutput().GetPointData().GetArray(fieldname)
     Vel_interped  = VN.vtk_to_numpy(array)

     for j in range(n_points_iso):
        if (  Wall_ID_check[ data_iso_IDs[j]]  ==1): #then on boundary
          Vel_interped[j,0] = 0.0
          Vel_interped[j,1] = 0.0
          Vel_interped[j,2] = 0.0
     #write output
     output_vtk = VN.numpy_to_vtk(Vel_interped)
     output_vtk.SetName(fieldname)
     data_iso.GetPointData().AddArray(output_vtk)
     output_filename2 =  output_root + output_name +  str(t_index) + '.vtu'
     myoutput = vtk.vtkXMLDataSetWriter()
     myoutput.SetInputData(data_iso)
     myoutput.SetFileName(output_filename2)
     myoutput.Write()
     ind = ind + T_delta


 print ('Done')



if __name__ == "__main__":
 

 fieldname = 'velocity'
 T_first = 10100 
 T_delta = 20
 T = 1 # The total number of files
 input_filename = '/Users/amir/Data/Berkeley_IMAC/carotid/P3/09-29-2015-085351/4-procs_case/all_results_'
 output_root = './'
 input_new_mesh ='/Users/amir/Data/Berkeley_IMAC/carotid/P3_advdiff/mesh-complete2/mesh-complete2.mesh.vtu'
 input_file_wall ='/Users/amir/Data/Berkeley_IMAC/carotid/P3_advdiff/mesh-complete2/mesh-surfaces/wall.vtp' #No slip wall for new mesh
 output_name = 'carotid_vel_interpolated_'
 


interp(input_filename, output_root + output_name, fieldname, T_first, T_delta, T,input_new_mesh,input_file_wall)
  

