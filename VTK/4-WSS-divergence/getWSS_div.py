import numpy
import sys
import vtk
import glob
import re
import math
from vtk.util import numpy_support as VN
def wss_theta(input_filename, output_filename, fieldname, T_first, T_delta, T, Flag_save_time_average ):




 t_index = T_first - T_delta
 for t in range(T):
     t_index = t_index + T_delta  
     input_filename2 = input_filename + str(t_index) + '.vtu'
     print ('Loading', input_filename2)
     reader =  vtk.vtkXMLUnstructuredGridReader()  #vtk.vtkPolyDataReader()
     reader.SetFileName(input_filename2)
     reader.Update()
     data = reader.GetOutput()
     meshtosurf = vtk.vtkDataSetSurfaceFilter()  #extract surface
     meshtosurf.SetInputData(data)
     meshtosurf.Update()
     data = meshtosurf.GetOutput()  

     n_points = data.GetNumberOfPoints()



     wss_grad_matrix = numpy.zeros((3, 3)) 
     wss_div = numpy.zeros((n_points, 1))
     wss_div_Tavg = numpy.zeros((n_points, 1))

     WSS = VN.vtk_to_numpy(data.GetPointData().GetArray(fieldname))

     
     if (Flag_normalized):
      for i in range(n_points):
       WSS_mag = math.sqrt((WSS[i,0]**2)+(WSS[i,1]**2)+(WSS[i,2]**2))
       WSS[i,0] = WSS[i,0] / WSS_mag
       WSS[i,1] = WSS[i,1] / WSS_mag
       WSS[i,2] = WSS[i,2] / WSS_mag
      output_vtk = VN.numpy_to_vtk(WSS)
      output_vtk.SetName(fieldname)
      data.GetPointData().AddArray(output_vtk)

     #wss vector gradient
     gradientFilter = vtk.vtkGradientFilter()
     gradientFilter.SetInputData(data)
     gradientFilter.SetInputArrayToProcess(0,0,0,0,fieldname)
     gradientFilter.SetResultArrayName('wssVector_grad')
     gradientFilter.Update()
     data_grad = gradientFilter.GetOutput()
     wss_grad_vector = VN.vtk_to_numpy(data_grad.GetPointData().GetArray('wssVector_grad'))
     for i in range(n_points):
       wss_grad_matrix[0,0] = wss_grad_vector[i,0]
       wss_grad_matrix[0,1] = wss_grad_vector[i,1]
       wss_grad_matrix[0,2] = wss_grad_vector[i,2]
       wss_grad_matrix[1,0] = wss_grad_vector[i,3]
       wss_grad_matrix[1,1] = wss_grad_vector[i,4]
       wss_grad_matrix[1,2] = wss_grad_vector[i,5]
       wss_grad_matrix[2,0] = wss_grad_vector[i,6]
       wss_grad_matrix[2,1] = wss_grad_vector[i,7]
       wss_grad_matrix[2,2] = wss_grad_vector[i,8]
       wss_div[i] =  wss_grad_matrix[0,0] + wss_grad_matrix[1,1] + wss_grad_matrix[2,2] 
       wss_div_Tavg[i] = wss_div_Tavg[i] +  wss_div[i] #Time average
     output_vtk = VN.numpy_to_vtk(wss_div)
     output_vtk.SetName('wss_div')
     data.GetPointData().AddArray(output_vtk)
     #data.GetPointData().RemoveArray('wss')
     output_filename2 = output_filename + str(t_index) + '.vtk'
     myoutput = vtk.vtkXMLDataSetWriter()
     myoutput.SetInputData(data)
     myoutput.SetFileName(output_filename2)
     myoutput.Write()


 if (Flag_save_time_average):
  for i in range(n_points):
    wss_div_Tavg[i] = wss_div_Tavg[i] / T 
  print ('Getting WSS_div Time averaged..') 	
  data.GetPointData().RemoveArray('wss_div')
  output_vtk = VN.numpy_to_vtk(wss_div_Tavg)
  output_vtk.SetName('wss_div_Tavg')
  data.GetPointData().AddArray(output_vtk)
  output_filename2 = output_filename + 'Tavg.vtk'
  myoutput = vtk.vtkXMLDataSetWriter()
  myoutput.SetInputData(data)
  myoutput.SetFileName(output_filename2)
  myoutput.Write() 
 
 print ('Done')



if __name__ == "__main__":
 


  fieldname = 'WSS'
  T_first = 10100 
  T_delta = 20
  T = 1 #number of files
  input_filename = '/Users/amir/Data/Berkeley_IMAC/carotid/P3/09-29-2015-085351/4-procs_case/all_results_'
  output_root = './'
  output_name = 'all_results_WSSdiv_'
  Flag_save_time_average = False
  Flag_normalized = True #If True, calculates divergence based on normalized WSS vector



  wss_theta(input_filename, output_root + output_name, fieldname, T_first, T_delta, T, Flag_save_time_average  )
  

