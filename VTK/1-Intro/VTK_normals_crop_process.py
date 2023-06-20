import numpy
import sys
import vtk
import glob
import re
import math
from vtk.util import numpy_support as VN

def vtk_intro(input_filename, output_filename, fieldname):


     print('Loading', input_filename)
     reader = vtk.vtkXMLUnstructuredGridReader() #VTU
     #reader = vtk.vtkUnstructuredGridReader() #VTK

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
     normals.Update()
     data = normals.GetOutput()

     #data.GetPointData().RemoveArray('wss')  #we can remove arrays if we want 
     n_points = data.GetNumberOfPoints()
    
 

     print('total points on surface:', n_points)


     #We can read arrays in the data, process, and save them 
     P = VN.vtk_to_numpy(data.GetPointData().GetArray(fieldname )) #Read the array fieldname from file
     P_dyn = numpy.zeros((n_points, 1))  #Pressure squared (processed data)
     for i in range(n_points):
        P_dyn[i] = P[i]**2

     theta_vtk = VN.numpy_to_vtk(P_dyn)
     theta_vtk.SetName('P_squared')  
     data.GetPointData().AddArray(theta_vtk)

     #define a plane for cropping
     plane1 = vtk.vtkPlane()    # the plane with inside out on
     plane1.SetOrigin(-3.1,-2.0,-10.9)
     plane1.SetNormal(0.35,-0.25,0.9)
  

     #crop the data with the plane
     clipper = vtk.vtkClipPolyData()
     clipper.SetInputData(data)
     clipper.SetClipFunction(plane1)
     clipper.InsideOutOn()  #Which side? Other side --> clipper.InsideOutOff() 
     clipper.Update()
     data_clipped = clipper.GetOutput()

     #If you want to process the gradient tensor:
     #wss_grad_vector = VN.vtk_to_numpy(data_clipped .GetPointData().GetArray('wssVector_grad'))
     
     
     #Save the final outcome 
     myoutput = vtk.vtkXMLDataSetWriter() 
     myoutput.SetInputData(data_clipped)
     myoutput.SetFileName(output_filename)
     myoutput.Write()
        
     



     print ('Done!')


if __name__ == "__main__":
    input_filename = '/Users/amir/Data/Berkeley_IMAC/carotid/P3/09-29-2015-085351/4-procs_case/all_results_10100.vtu'
    output_name = 'carotid_data_processed.vtk'
    fieldname = 'pressure'
    vtk_intro(input_filename, output_name, fieldname)
  

