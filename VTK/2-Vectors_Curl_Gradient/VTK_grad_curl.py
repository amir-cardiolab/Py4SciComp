import numpy
import sys
import vtk
import glob
import re
import math
from vtk.util import numpy_support as VN
#Time avg of theta gradient
def myVorticity(input_filename,output_filename, T_first, T_delta, T,fieldname):

   

   t_index = T_first - T_delta
   for t in range(T):
     t_index = t_index + T_delta
     input_filename2 = input_filename + str(t_index) + '.vtu'
     print ('Loading', input_filename2)
     reader = vtk.vtkXMLUnstructuredGridReader()
     reader.SetFileName(input_filename2)
     reader.Update()
     data = reader.GetOutput()

     
     gradientFilter = vtk.vtkGradientFilter()
     gradientFilter.SetInputData(data)
     gradientFilter.SetInputArrayToProcess(0,0,0,0,fieldname)
     gradientFilter.SetResultArrayName('gradu')
     gradientFilter.ComputeVorticityOn()  #Gets us the curl
     gradientFilter.Update()
     data_grad = gradientFilter.GetOutput()


     #If you want to process the gradient tensor:
     #wss_grad_vector = VN.vtk_to_numpy(data_grad .GetPointData().GetArray('gradu'))
     #wss_grad_vector[i,0] #This is \partial u / \partial x
     #wss_grad_vector[i,1] #This is \partial u / \partial y
     #wss_grad_vector[i,2] #This is \partial u / \partial z
     #wss_grad_vector[i,3] #This is \partial v / \partial x
     #wss_grad_vector[i,4] #This is \partial v / \partial y
     #wss_grad_vector[i,5] #This is \partial v / \partial z
     #wss_grad_vector[i,6] #This is \partial w / \partial x
     #wss_grad_vector[i,7] #This is \partial w / \partial y
     #wss_grad_vector[i,8] #This is \partial w / \partial z


     #write results
     myoutput = vtk.vtkXMLDataSetWriter() 
     myoutput.SetInputData(data_grad)
     output_filename2 = output_filename + str(t_index) + '.vtu'
     myoutput.SetFileName(output_filename2)
     myoutput.Write()

     print ('Done!')


if __name__ == "__main__":
    fieldname = 'velocity'
    input_filename = '/Users/amir/Data/Berkeley_IMAC/carotid/P3/09-29-2015-085351/4-procs_case/all_results_'
    output_filename = 'output_grad_curl_'
    T_first = 10100  #index of first file
    T_delta = 1 #index delta btwn files
    T = 1    #total number of files
    myVorticity(input_filename, output_filename , T_first, T_delta, T,fieldname)
  

