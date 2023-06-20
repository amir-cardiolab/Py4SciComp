import numpy
import sys
import vtk
import glob
import re
import math
from vtk.util import numpy_support as VN
import struct

def seed(input_filename, output_filename, Normal_wall_distance, Flip,Write_bin):
     print ('Loading', input_filename)
    # reader = vtk.vtkDataSetReader()
     reader = vtk.vtkXMLUnstructuredGridReader()
     reader.SetFileName(input_filename)
     reader.Update()
     data = reader.GetOutput()

     meshtosurf = vtk.vtkDataSetSurfaceFilter()  #extract surface
     meshtosurf.SetInputData(data)
     meshtosurf.Update()
     data = meshtosurf.GetOutput()  
    
     normals = vtk.vtkPolyDataNormals()
     normals.SetInputData(data)
     normals.SetFeatureAngle(91);
     normals.SetSplitting(0); #otherwise duplicates sharp nodes
     normals.ConsistencyOn()
     if (Flip==1):
       normals.FlipNormalsOn()
     normals.Update()
     data = normals.GetOutput()
     normal_array = VN.vtk_to_numpy(data.GetPointData().GetArray('Normals'))
     n_points = data.GetNumberOfPoints()
     print ('total points:', n_points)
     
     IC = numpy.zeros(3)
     ID = numpy.zeros(n_points)
     Points = vtk.vtkPoints()
     #Also write to binary for flowVC
     if (Write_bin==1):
       file_bin = open( 'nearwall_tracers.bin', 'ab')
       data_b = struct.pack('i', n_points)
       file_bin.write(data_b) #number of pts
     for i in range(n_points):
       data.GetPoint(i, IC)
       ID[i] = i
       X  = IC[0] +  normal_array[i,0]*Normal_wall_distance
       Y  = IC[1] +  normal_array[i,1]*Normal_wall_distance
       Z  = IC[2] + normal_array[i,2]*Normal_wall_distance
       Points.InsertNextPoint(X, Y, Z)
       if (Write_bin==1):
        file_bin.write(X)
        file_bin.write(Y)
        file_bin.write(Z)

     if (Write_bin==1):
       file_bin.close()
     polydata = vtk.vtkPolyData()
     polydata.SetPoints(Points)
     writer = vtk.vtkPolyDataWriter()
     writer.SetFileName(output_filename)
     writer.SetInputData(polydata)
     writer.Write()




     print ('Done!')


if __name__ == "__main__":

    input_filename = '/Users/amir/Data/Berkeley_IMAC/carotid/P3/09-29-2015-085351/4-procs_case/all_results_10100.vtu'
    output =  'nearwall_tracers.vtk'
    Normal_wall_distance = 0.01 #distance for placing near-wall tracers normal to wall
    Flip = 0 #if 1 then flips normals
    Write_bin = 0 #if 1 also writes in bin format for flowVC
    seed(input_filename,output, Normal_wall_distance, Flip,Write_bin )


