import vtk
import numpy as np

def main():

	### Read the reference wall mesh

	wall_ref_rootdir = '/Users/af2289/data/task2/mesh/z-inlet/mesh-surfaces/'
	wall_ref = wall_ref_rootdir + 'Wall.vtp'
	reader = vtk.vtkXMLPolyDataReader()
	reader.SetFileName(wall_ref)
	reader.Update()
	data_ref = reader.GetOutput()
	n_cells = data_ref.GetNumberOfCells()

	### Read the coordinates of the final step flowVC outcome

	file_rootdir = '/Users/af2289/data/task4/results/trachea_traj/'
	file = file_rootdir + 'trachea_1_5e-5_NS.100.vtk'
	file_output = file_rootdir + 'dc.vtp'
	reader = vtk.vtkPolyDataReader()
	reader.SetFileName(file)
	reader.Update()
	data = reader.GetOutput()
	n_nodes = data.GetNumberOfPoints()
	n_deposited = 5000 # Should be found with pre-processing!

	### Find number of particles deposited in each element

	n_dc = vtk.vtkIntArray()
	n_dc.SetName('deposition number')
	for i in range(n_cells):
		n_dc.InsertValue(i, 0)
	dc = vtk.vtkFloatArray()
	dc.SetName('dc')
	closestPoint = np.zeros(3)
	cell = vtk.vtkGenericCell()
	cellId = vtk.mutable(0)
	subId = vtk.mutable(0)
	closestPointDist_squared = vtk.mutable(0)
	VTKpoints = vtk.vtkPoints()
	Locator = vtk.vtkCellLocator()
	Locator.SetDataSet(data_ref)
	Locator.BuildLocator()

	printProgressBar(0, n_nodes, prefix = 'Progress:', suffix = 'Complete', length = 50)
	for i in range(n_nodes):
		pt = data.GetPoint(i)
		Locator.FindClosestPoint(pt, closestPoint, cellId, subId, closestPointDist_squared)
		n_dc.InsertValue(cellId, n_dc.GetValue(cellId) + 1)
		printProgressBar(i + 1, n_nodes, prefix = 'Progress:', suffix = 'Complete', length = 50)

	### Find surface area and each element area of the lung
	### Calculate non-dimensional deposition concentration

	mesh_element = vtk.vtkMeshQuality()
	mesh_element.SetInputData(data_ref)
	mesh_element.SetTriangleQualityMeasureToArea()
	mesh_element.Update()
	qualityArray = mesh_element.GetOutput().GetCellData().GetArray('Quality')

	mesh_surface = vtk.vtkMassProperties()
	mesh_surface.SetInputData(data_ref)
	total_area = mesh_surface.GetSurfaceArea()
	
	for i in range(n_cells):
		tmp = (float(n_dc.GetValue(i)) / n_deposited) / (qualityArray.GetValue(i) / total_area)
		dc.InsertValue(i, tmp)

	### Save the final data
	data_ref.GetCellData().AddArray(n_dc)
	data_ref.GetCellData().AddArray(dc)
	data_ref.GetCellData().RemoveArray('quality')
	myoutput = vtk.vtkPolyDataWriter()
	myoutput.SetInputData(data_ref)
	myoutput.SetFileName(wall_ref_rootdir + 'dc.vtk')
	myoutput.Write()	
	print('done!')

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '*'):
	'''
	 Call in a loop to create terminal progress bar
	 @params:
		  iteration   - Required  : current iteration (Int)
		  total       - Required  : total iterations (Int)
		  prefix      - Optional  : prefix string (Str)
		  suffix      - Optional  : suffix string (Str)
		  decimals    - Optional  : positive number of decimals in percent complete (Int)
		  length      - Optional  : character length of bar (Int)
		  fill        - Optional  : bar fill character (Str)
	'''    
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
	# Print New Line on Complete
	if iteration == total: 
	  print()

if __name__ == '__main__':
	 main()