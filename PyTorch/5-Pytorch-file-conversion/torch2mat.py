import numpy as np
import torch
import vtk
from vtk.util import numpy_support as VN
from torch import nn
import torch.nn.functional as F
#from model import *
#dev = "cuda:0" if torch.cuda.is_available() else "cpu"
import matplotlib.pyplot as plt
from scipy.io import savemat

#convert the trained pt networks and input/label data to mat format for Matlab


class DeepAutoencoder_deeper(torch.nn.Module):  #https://www.geeksforgeeks.org/implement-deep-autoencoder-in-pytorch-for-image-reconstruction/
    def __init__(self):
        super().__init__()        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28 , 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )
          
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 28 * 28),
            #torch.nn.Sigmoid()
        )
  
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def init_normal(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight)
        elif type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight)

class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()

        # Convolution 1;  #in_challens = 1 for greyscale;  3 for RGB
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.relu1 = nn.ReLU()

        self.cnn11 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=5)
        self.relu11 = nn.ReLU()

        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.relu2 = nn.ReLU()

        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected 1 (readout). #? out_channel ( kernel_size * kernel_size)
        #self.fc1 = nn.Linear(32 * 4 * 4, 128)   
        self.fc1 = nn.Linear(288, 512)   
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 28*28) 

        #self.soft = nn.Softmax() #Dont use with crossentropyloss

    def forward(self, x):
        # Convolution 1
        #print(x.shape) 
        out = self.cnn1(x)
        #print(out.shape)
        out = self.relu1(out)
        out = self.cnn11(out)
        out = self.relu1(out)


        #print(out.shape)
        # Max pool 1
        out = self.maxpool1(out)
        #print(out.shape)

        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)

        #print(out.shape)
        # Max pool 2 
        out = self.maxpool2(out)

        #print(out.shape)

        # Resize
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)

        #print(out.shape)
        # Linear function (readout)
        out = self.fc1(out)
        out  = self.relu3(out)
        out = self.fc2(out)

        #out = F.log_softmax(out) #binary classification 
        #out = self.soft(out)
        #out  = self.relu3(out)

        return out



class DeepAutoencoder(torch.nn.Module):  #https://www.geeksforgeeks.org/implement-deep-autoencoder-in-pytorch-for-image-reconstruction/
    def __init__(self):
        super().__init__()        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28 , 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )
          
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(32, 44),
            torch.nn.ReLU(),
            torch.nn.Linear(44, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 80),
            torch.nn.ReLU(),
            torch.nn.Linear(80, 28*28),
            #torch.nn.Sigmoid()
        )
  
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




dev = "cpu"

fieldname_perm = "Vel_mag"
fieldname_u = "Vel_mag"

File_input_train =  "/Users/amir/Data/porous-ML/generalize/stenosis_superresolution/fenics/results/vtk_files/u_stenosis_lres__"
File_input_label = "/Users/amir/Data/porous-ML/generalize/stenosis_superresolution/fenics/results/vtk_files/u_stenosis_hres__"
num_files = 400
File_ood = "/Users/amir/Data/porous-ML/generalize/stenosis_superresolution/fenics/results/vtk_files/OODu_stenosis_lres__"
File_ood_label =  "/Users/amir/Data/porous-ML/generalize/stenosis_superresolution/fenics/results/vtk_files/OODu_stenosis_hres__"
num_files_ood = 100

Output_file = "superresolution_train_test_image2image.mat"
nn_file = "results/superresolution_stenosis.pt"

Flag_CNN = False

nPt = 28
#nPt = 264 #trying hres
my_eps = 0.0
xStart =  1.2 + my_eps
xEnd =  1.35 - my_eps
yStart = 0.05 + my_eps
yEnd = 0.2 - my_eps
Norm_factor_perm =  2.
Norm_factor_out = 2.

#strcutured grid:
x = np.linspace(xStart, xEnd, nPt)
y = np.linspace(yStart, yEnd, nPt)
x, y = np.meshgrid(x, y)
n_points = nPt * nPt


k_all =  np.zeros((num_files, nPt,nPt))
label_all= np.zeros((num_files,nPt*nPt)) 
k_ood =  np.zeros((num_files_ood, nPt,nPt))
label_ood= np.zeros((num_files_ood,nPt*nPt)) 

for i in range(num_files):

    
    mesh_file = File_input_train + str(i) +".vtu"
    #print ('Loading', mesh_file)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    data_vtk_in = reader.GetOutput()
    
    mesh_file_u = File_input_label  + str(i) +".vtu"
    #print ('Loading', mesh_file_u)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_file_u)
    reader.Update()
    data_vtk_out = reader.GetOutput()
    #Vel =  VN.vtk_to_numpy(data_vtk_out.GetPointData().GetArray( fieldname_u ))

    if (i ==0):
        VTKpoints = vtk.vtkPoints()
        n = 0
        for j in range(nPt):
            for k in  range(nPt):
              VTKpoints.InsertPoint(n, x[j,k], y[j,k], 0.)
              n = n + 1
        point_data = vtk.vtkUnstructuredGrid()
        point_data.SetPoints(VTKpoints)
    probe = vtk.vtkProbeFilter()
    probe2 = vtk.vtkProbeFilter()
    probe.SetInputData(point_data)
    probe2.SetInputData(point_data)
    probe.SetSourceData(data_vtk_in)
    probe2.SetSourceData(data_vtk_out)
    probe.Update()
    probe2.Update()
    array = probe.GetOutput().GetPointData().GetArray(fieldname_perm)
    array2 = probe2.GetOutput().GetPointData().GetArray(fieldname_u)
    perm_interped = VN.vtk_to_numpy(array)
    Vel_interped = VN.vtk_to_numpy(array2)
    image_input  = perm_interped.reshape(nPt,nPt) / Norm_factor_perm
    k_all[i,:,:] = image_input  
    label_all[i,:] = Vel_interped  / Norm_factor_out 


for i in range(num_files_ood):

    
    mesh_file = File_ood + str(i) +".vtu"
    #print ('Loading', mesh_file)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    data_vtk_in = reader.GetOutput()
    
    mesh_file_u = File_ood_label + str(i) +".vtu"
    #print ('Loading', mesh_file_u)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_file_u)
    reader.Update()
    data_vtk_out = reader.GetOutput()
    #Vel =  VN.vtk_to_numpy(data_vtk_out.GetPointData().GetArray( fieldname_u ))

    if (i ==0):
        VTKpoints = vtk.vtkPoints()
        n = 0
        for j in range(nPt):
            for k in  range(nPt):
              VTKpoints.InsertPoint(n, x[j,k], y[j,k], 0.)
              n = n + 1
        point_data = vtk.vtkUnstructuredGrid()
        point_data.SetPoints(VTKpoints)
    
    probe = vtk.vtkProbeFilter()
    probe2 = vtk.vtkProbeFilter()
    probe.SetInputData(point_data)
    probe2.SetInputData(point_data)
    probe.SetSourceData(data_vtk_in)
    probe2.SetSourceData(data_vtk_out)
    probe.Update()
    probe2.Update()
    array = probe.GetOutput().GetPointData().GetArray(fieldname_perm)
    array2 = probe2.GetOutput().GetPointData().GetArray(fieldname_u)
    perm_interped = VN.vtk_to_numpy(array)
    Vel_interped = VN.vtk_to_numpy(array2)
    image_input  = perm_interped.reshape(nPt,nPt) / Norm_factor_perm
    k_ood[i,:,:] = image_input  
    
    label_ood[i,:] =  Vel_interped  / Norm_factor_out 



#######################################

k_all = np.expand_dims(k_all, axis=1)
input_image = torch.Tensor(k_all).to(dev)


print(torch.Tensor.size(input_image))
#plt.imshow(input_image.data.numpy()[2,0,:,:], cmap='rainbow', interpolation='nearest')
#plt.show()


output_label = torch.Tensor(label_all).to(dev)


if (Flag_CNN):
    m = myCNN().to(dev)
else:
    #m = DeepAutoencoder().to(dev)
    m = DeepAutoencoder_deeper().to(dev)




m.load_state_dict(torch.load(nn_file,
                             map_location=torch.device(dev)))
m.eval()

y_predict= m(input_image).cpu().detach().numpy() # m(input_image).cpu().detach().numpy().reshape(-1)
y_true = output_label.cpu().detach().numpy()


print (np.shape(y_predict))
print (np.shape(y_true))


k_ood = np.expand_dims(k_ood, axis=1)
test_image = torch.Tensor(k_ood).to(dev)
output_label_ood  = torch.Tensor(label_ood).to(dev)


y_predict_ood = m(test_image).cpu().detach().numpy() #m(test_image).cpu().detach().numpy().reshape(-1)
y_true_ood = output_label_ood.cpu().detach().numpy()



N_train = np.size(y_true)
print ('Number of training data',N_train)
N_test = np.size(y_true_ood)
print ('Number of testing data',N_test)

input_images_train = np.zeros((num_files,nPt,nPt))
input_images_ood = np.zeros((num_files_ood,nPt,nPt))

input_images_train[:,:,:] = input_image.cpu().detach().numpy()[:,0,:,:]
input_images_ood[:,:,:] = test_image.cpu().detach().numpy()[:,0,:,:]

mdic = {"input_images_train": input_images_train, "input_images_ood": input_images_ood, "y_true_ood": y_true_ood, "y_predict": y_predict, "y_true": y_true, "y_predict_ood": y_predict_ood }
savemat(Output_file, mdic)

if(1):
 Error_training = abs( y_predict[:] - y_true[:] )
 Error_ood = abs( y_predict_ood[:] - y_true_ood[:] )
 print ('MAE training data',np.mean(Error_training))
 print ('MAE testing data',np.mean(Error_ood))

 print ('MAE training data max error',np.max(Error_training))
 print ('MAE testing data max error',np.max(Error_ood))

 print ('shape',np.shape(Error_ood))
 print ('shape',np.shape(Error_training))


 plt.figure()
 #plt.plot(y_predict,'-', label='NN solution', alpha=1.0,zorder=0)
 #plt.plot( y_true,'r--', label='True solution', alpha=1.0,zorder=0)
 plt.plot(Error_training,'-', label='Error_training', alpha=1.0,zorder=0)
 plt.plot(Error_ood,'r--', label='Error_ood', alpha=1.0,zorder=0)
 #plt.plot(y_predict_ood,'g-', label='NN solution OOD', alpha=1.0,zorder=0)
 #plt.plot( y_true_ood,'g--', label='True solution OOD', alpha=1.0,zorder=0)
 plt.legend(loc='best')
 plt.show()

print('Done!')

