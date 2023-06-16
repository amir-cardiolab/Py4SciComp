import vtk as vtk
import numpy as np
from os import getcwd

#Acknowledgement:  Kirk Hanson and Debanjan Mukherjee


def vtk_to_dolfin_xml_3D(a_vtkFileName, \
                          a_OutputFileName, \
                          a_IdVariable='Ids', \
                          a_FileType='vtu', \
                          a_UseID=False):

  #
  # read the file using fileType
  #
  if a_FileType == 'vtu':
    reader = vtk.vtkXMLUnstructuredGridReader()
  elif a_FileType == 'vtp':
    reader = vtk.vtkPolyDataReader()

  reader.SetFileName(a_vtkFileName)
  reader.Update()

  data = reader.GetOutput()

  #
  # number of nodes, and number of cells
  #
  numNodes    = data.GetNumberOfPoints()
  numElements = data.GetNumberOfCells()

  print ('Nodes:', numNodes)
  print ('Elements:', numElements)
  
  #
  # use global cell ID or node ID
  #
  if a_UseID:
    globCellID  = data.GetCellData().GetArray(a_IDVariable)
    globNodeID  = data.GetPointData().GetArray(a_IDVariable)

  #
  # open the xml file for writing
  #
  outFileObj = open(a_OutputFileName,'w')

  # 
  # header for the dolfin xml format
  #
  outFileObj.write('<?xml version="1.0"?>\n')
  outFileObj.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')

  #
  # header for cell types based on topological dimension
  #
  cellType  = "tetrahedron"
  dim       = 3
  outFileObj.write('  <mesh celltype="%s" dim="%d">\n' % (cellType, dim))

  #
  # header for vertices of the cells
  #
  outFileObj.write('    <vertices size="%d">\n' % (numNodes))

  coordinates = np.zeros(dim)

  for nodeID in range(numNodes):

    data.GetPoint(nodeID, coordinates)
    if a_UseID:
      nodeCount = globNodeID.GetValue(nodeID)
    else:
      nodeCount = nodeID

    # THE FOLLOWING LINE SHOULD ALSO BE DIMENSION INEDEPENDENT
    outFileObj.write('      <vertex index="%d" x="%0.16g" y="%0.16g" z="%0.16g" />\n' \
                    % (nodeCount, coordinates[0], coordinates[1], coordinates[2]))

  # 
  # finish writing the nodal coordinates
  #
  outFileObj.write('    </vertices>\n')
  
  #
  # beginning to write the cell connectivities
  #
  print (numElements, 'elements, finding connectivity')

  #
  # header for the cell connectivity data in the xml file
  #
  outFileObj.write('    <cells size="%d">\n' % numElements)

  if cellType == "tetrahedron":
    connectivity = np.zeros(4, dtype=np.int32)
  elif cellType == "triangle":
    connectivity = np.zeros(3, dtype=np.int32)

  for cellID in range(numElements):
    
    nodesInCell = vtk.vtkIdList()
    data.GetCellPoints(cellID, nodesInCell)
    
    for j in range(4): # CHANGE THIS TO MAKE DIMENSION INDEPENDENT
      if a_UseID:
        connectivity[j] = globNodeID.GetValue(nodesInCell.GetId(j))
        cellCount       = globCellID.GetValue(i)
      else:
        connectivity[j] = nodesInCell.GetId(j)
        cellCount       = cellID
            
    connectivity = np.sort(connectivity)

    # CHANGE TIS TO MAKE DIMENSION/CELLTYPE INDEPENDENT
    outFileObj.write('      <%s index="%d" v0="%d" v1="%d" v2="%d" v3="%d" />\n' \
            % (cellType, cellID, connectivity[0], connectivity[1], connectivity[2], connectivity[3]))

  #
  # finish writing the cell connectivity
  #
  outFileObj.write('    </cells>\n  </mesh>\n</dolfin>')
  outFileObj.close()

# This code is for surface mesh 
def vtp_to_dolfin_xml(a_vtpFilename,  
                      a_outputFilename,
                      a_IDVariable='Ids',
                      a_geometricDimension=3,
                      a_topologicalDimension=2,
                      a_isXMLRead=False,
                      a_Test=False):
    
    #
    # read file based on file extension 
    #
    if a_isXMLRead:
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkPolyDataReader()

    reader.SetFileName(a_vtpFilename)
    reader.Update()
    data = reader.GetOutput()

    #
    # number of node, and number of cells
    #
    n_nodes     = data.GetNumberOfPoints()
    n_elements  = data.GetNumberOfCells()
    
    print ('Nodes:', n_nodes)
    print ('Elements:', n_elements)

    #
    # use the global cell id and global node id
    # 
    if a_Test:
        globCellID  = data.GetCellData().GetArray(a_IDVariable)
        globNodeID  = data.GetPointData().GetArray(a_IDVariable)

    #
    # open xml file for writing
    #
    fout = open(a_outputFilename, 'w')

    # 
    # header for the dolfin xml format
    #
    fout.write('<?xml version="1.0"?>\n')
    fout.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')

    #
    # header for cell types based on topological dimension
    #
    if a_topologicalDimension == 2:
        fout.write('  <mesh celltype="triangle" dim="%d">\n' \
                   % (a_geometricDimension))
    else:
        fout.write('  <mesh celltype="interval" dim="%d">\n' \
                   % (a_geometricDimension))

    print (n_nodes, 'nodes, finding coordinates')

    #
    # header for the nodal vertices data in the xml file
    #
    fout.write('    <vertices size="%d">\n' % n_nodes)

    #
    # beginning to write the vertices (coordinates)
    #
    coordinates = np.zeros(3)

    for i in range(n_nodes):

        data.GetPoint(i, coordinates)
        if a_Test:
            nodeID  = globNodeID.GetValue(i)
        else:
            nodeID  = i

        if a_geometricDimension == 3:
            fout.write( \
                '      <vertex index="%d" x="%.16g" y="%.16g" z="%.16g" />\n' \
                % (nodeID, coordinates[0], coordinates[1], coordinates[2]))
        elif a_geometricDimension == 2:
            fout.write('      <vertex index="%d" x="%.16g" y="%.16g" />\n'
                       % (nodeID, coordinates[0], coordinates[1]))
        else:
            fout.write('      <vertex index="%d" x="%.16g" />\n'
                       % (nodeID, coordinates[0], coordinates[1]))
    
    # 
    # finish writing the nodal coordinates
    #
    fout.write('    </vertices>\n')

    #
    # beginning to write the cell connectivities
    #
    print (n_elements, 'elements, finding connectivity')

    #
    # header for the cell connectivity data in the xml file
    #
    fout.write('    <cells size="%d">\n' % n_elements)

    #
    # beginning to write the connectivity data
    #
    if a_topologicalDimension == 2:
        connectivity = np.zeros(3, dtype=np.int32) # triangle surface elements
    elif a_topologicalDimension == 1:
        connectivity = np.zeros(2, dtype=np.int32) # line segments

    for i in range(n_elements):

        nodesInCell = vtk.vtkIdList()
        data.GetCellPoints(i, nodesInCell)

        if a_topologicalDimension == 2:
            for j in range(3):
                if a_Test:
                    connectivity[j] = globNodeID.GetValue(nodesInCell.GetId(j))
                    cellID          = globCellID.GetValue(i)
                else:
                    connectivity[j] = nodesInCell.GetId(j)
                    cellID          = i
            
            connectivity = np.sort(connectivity)
            fout.write( \
                    '      <triangle index="%d" v0="%d" v1="%d" v2="%d" />\n' \
                    % (cellID, connectivity[0], connectivity[1], connectivity[2]))
        else:
            for j in range(2):
                if a_Test:
                    connectivity[j] = globNodeID.GetValue(nodesInCell.GetId(j))
                    cellID          = globCellID.GetValue(i)
                else:
                    connectivity[j] = nodesInCell.GetId(j)
                    cellID          = i

            fout.write( \
                    '      <triangle index="%d" v0="%d" v1="%d" />\n' \
                    % (cellID, connectivity[0], connectivity[1]))

    #
    # finish writing the cell connectivity
    #
    fout.write('    </cells>\n  </mesh>\n</dolfin>')
    fout.close()


def node_list_to_facet_function(a_InputFileName,    # the complete mesh
                                a_NodeListFiles,
                                a_OutputFileName,
                                a_MeshType='vtu',
                                a_GeoDimension=3,
                                a_IsXMLRead=True):

    numBoundaries = len(a_NodeListFiles)

    print (numBoundaries)

    boundaryList = []

    for i in range(numBoundaries):
        boundaryList.append(np.loadtxt(a_NodeListFiles[i], dtype=np.int32))

    if a_IsXMLRead:
        if a_MeshType == 'vtp':
            reader = vtk.vtkXMLPolyDataReader()
        elif a_MeshType == 'vtu':
            reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        if a_MeshType == 'vtp':
            reader = vtk.vtkPolyDataReader()
        elif a_MeshType == 'vtu':
            reader = vtk.vtkUnstructuredGridReader()

    reader.SetFileName(a_InputFileName)
    reader.Update()

    polyData = reader.GetOutput()

    numNodes    = polyData.GetNumberOfPoints()
    numElements = polyData.GetNumberOfCells()

    outFileObj  = open(a_OutputFileName, 'w')

    outFileObj.write('<?xml version="1.0"?>\n')
    outFileObj.write('<dolfin xmlns:dolfin="http://fenicsproject.org">\n')
    outFileObj.write('  <mesh_function>\n')

    #
    # in the following function dim = 2 for 3D face functions
    # and 4*numElements for tetrahedral elements with 4 nodes each
    #
    outFileObj.write('    <mesh_value_collection name="f" type="uint"'
               + ' dim="2" size="%d">\n' % (4*numElements))

    #
    # now loop over all elements 
    #
    for i in range(numElements):

        temp = vtk.vtkIdList()

        polyData.GetCellPoints(i, temp)

        numPointsInCells = temp.GetNumberOfIds()

        tags            = np.zeros(numPointsInCells, dtype=np.int32)
        connectivity    = np.zeros(numPointsInCells, dtype=np.int32)

        for j in range(numPointsInCells):
            connectivity[j] = temp.GetId(j)

        connectivity = np.sort(connectivity)

        for k in range(numBoundaries):
            
            #
            # mark nodes on the surface with True/1, and ones 
            # not on surface with False/0
            #
            nodeOnBoundary = [n in boundaryList[k] for n in connectivity]
            
            #
            # tag the node not on surface
            #
            if sum(nodeOnBoundary) == (numPointsInCells - 1):
                oppositeNode = nodeOnBoundary.index(False)
                #print i, oppositeNode, k
                tags[oppositeNode] = k + 1

        for j in range(numPointsInCells):
            outFileObj.write('      <value cell_index="%d" local_entity="%d" value="%d" />\n' 
                        % (i, j, tags[j]))

    outFileObj.write('    </mesh_value_collection>\n')
    outFileObj.write('  </mesh_function>\n')
    outFileObj.write('</dolfin>')
    outFileObj.close()

def getNodeListFromSurface(FROM_SIMVASCULAR,a_InputFileName,
                            a_OutputFileName,
                            a_IdVariable='Ids',
                            a_IsXMLRead=True):

    if a_IsXMLRead:
        reader = vtk.vtkXMLPolyDataReader()
    else:
        reader = vtk.vtkPolyDataReader()

    if (FROM_SIMVASCULAR==1):
       a_IdVariable = 'GlobalNodeID'

    reader.SetFileName(a_InputFileName)
    reader.Update()

    polyData    = reader.GetOutput()
    numNodes    = polyData.GetNumberOfPoints()
    nodeIDs     = polyData.GetPointData().GetArray(a_IdVariable)
    outFileObj  = open(a_OutputFileName, 'w')
    nodeIDArray = np.zeros(numNodes, dtype=np.int32)

    for i in range(numNodes):
        nodeIDArray[i] = nodeIDs.GetValue(i)
        if (FROM_SIMVASCULAR==1):
            nodeIDArray[i] =  nodeIDArray[i] - 1  #Simvascular Ids start from 1
        outFileObj.write('%d \n' % nodeIDArray[i])

    outFileObj.close()

if __name__ == '__main__':

    #rootDir = getcwd()
    #vtp_to_dolfin_xml('inlet.vtk','inlet.xml')
    #vtp_to_dolfin_xml('wall.vtk','wall.xml')
    #vtp_to_dolfin_xml('outlet1.vtk','outlet1.xml')
    #vtp_to_dolfin_xml('outlet2.vtk','outlet2.xml')
    #-------Use functions below here:
    
    FROM_SIMVASCULAR = 1 #if 1 then from simvascular face vtk files (still need to extract surface so it is polydata). Simvascular starts the ID's from 1. But VTK generateIDs starts from 0
    
    getNodeListFromSurface(FROM_SIMVASCULAR,'/Users/amir/Data/Berkeley_IMAC/carotid/P3/mesh-complete/mesh-surfaces/inlet.vtp','inletNodes.dat') #Extract the surface so it is polydata. !!!!!! Must use extract cells by region not clip!! (will lose Ids)
    getNodeListFromSurface(FROM_SIMVASCULAR,'/Users/amir/Data/Berkeley_IMAC/carotid/P3/mesh-complete/mesh-surfaces/wall.vtp','wallNodes.dat')
    getNodeListFromSurface(FROM_SIMVASCULAR,'/Users/amir/Data/Berkeley_IMAC/carotid/P3/mesh-complete/mesh-surfaces/ext.vtp','outletNodes1.dat')
    getNodeListFromSurface(FROM_SIMVASCULAR,'/Users/amir/Data/Berkeley_IMAC/carotid/P3/mesh-complete/mesh-surfaces/int.vtp','outletNodes2.dat')
    nodesFiles = ['inletNodes.dat','wallNodes.dat','outletNodes1.dat','outletNodes2.dat' ]  #This will be the same order of tags you can use in FEniCS:   1, 2, 3, 4   
     
   # node_list_to_facet_function('fullVascularMesh3D.vtk', nodesFiles, 'nodeFacets.xml')
    #node_list_to_facet_function('m.vtk', nodesFiles, 'nodeFacets.xml')
    node_list_to_facet_function('/Users/amir/Data/Berkeley_IMAC/carotid/P3/mesh-complete/mesh-complete.mesh.vtu', nodesFiles, 'BCnodeFacets.xml')
#   vtk_to_dolfin_xml_3D('/Users/amir/data/Thrombus/P41/vel/Patient41Rest_vel.3020.vtk','/Users/amir/data/oldAAAwss/P41/Patient41Rest_mesh.xml')

    vtk_to_dolfin_xml_3D('/Users/amir/Data/Berkeley_IMAC/carotid/P3/mesh-complete/mesh-complete.mesh.vtu','Volume_mesh.xml')

#INFO!!!!: To get vtk files for faces. First do GenerateIds filter in paraview and then extract surface (extract surface renames GlobalNodeIDS). Use this new array as Ids.





