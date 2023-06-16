import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as dsets 
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
import numpy as np
import vtk
import sys
from vtk.util import numpy_support as VN
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image


#obtained from: https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_convolutional_neuralnetwork/

#another example: https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/

#another example for selecting the CNN numbers: https://www.kaggle.com/code/shtrausslearning/pytorch-cnn-binary-image-classification

#An example of simple setting of conv2d dimensions and number of layers 

'''
STEP 3: CREATE MODEL CLASS
'''


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




#https://analyticsindiamag.com/how-to-implement-convolutional-autoencoder-in-pytorch-with-cuda/
class autoencoder(nn.Module):  
    def __init__(self):
        super(autoencoder, self).__init__()
        #Encoder
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 2)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 4, kernel_size = 5, stride = 1)
        self.maxpool = nn.MaxPool2d(2,2)
        #Encoder
        self.tconv1 = nn.ConvTranspose2d(in_channels = 4, out_channels = 16, kernel_size = 5, stride = 1, padding = 2)
        self.tconv2 = nn.ConvTranspose2d(in_channels = 16, out_channels = 1, kernel_size = 5, stride = 1)
        self.flatten = nn.Flatten()
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.tconv1(x))
        x = F.sigmoid(self.tconv2(x))

        return x




class DeepAutoencoder_original(torch.nn.Module):  #https://www.geeksforgeeks.org/implement-deep-autoencoder-in-pytorch-for-image-reconstruction/
    def __init__(self):
        super().__init__()        
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28 , 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8)
        )
          
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 28 * 28),
            #torch.nn.Sigmoid()
        )
  
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



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


def data_transform(a):
    d_mean = 0.1
    d_std = 0.3
    #d_mean = np.mean(a)
    #d_std = np.std(a)
    return (a-d_mean)/d_std


def get_image(path,flag):
    #with open(os.path.abspath(path), 'rb') as f:
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if (flag):
                return img.convert('L') #grey
            else:
                return img.convert('RGB')
            

# resize and take the center part of image to what our model expects
def get_input_transform():
    if (Flag_grey):
     normalize = transforms.Normalize(mean=0.1,
                                    std= 0.3)
    else:       
     normalize = transforms.Normalize(mean=[0.4, 0.4, 0.4],
                                    std=[0.2, 0.2, 0.2])       
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])    

    return transf


def get_input_tensors(img):
    transf = get_input_transform()
    # unsqeeze converts single image to batch of 1
    return transf(img).unsqueeze(0)

######### Read the training/test data abd classify them ###############


Flag_grey = True 

Flag_CNN = False



output_file = "results/superresolution_stenosis.pt"

file_prefix_in = "/Users/amir/Data/porous-ML/generalize/stenosis_superresolution/fenics/results/vtk_files/u_stenosis_lres__"
file_prefix_out = "/Users/amir/Data/porous-ML/generalize/stenosis_superresolution/fenics/results/vtk_files/u_stenosis_hres__"

fieldname_perm = "Vel_mag"
fieldname_u = "Vel_mag"
ind_start = 0
num_files = 400 #Total number of files
num_testing = 80   #number of files used for validation
num_training = num_files - num_testing 
nPt = 28
#nPt = 264 #trying hres
my_eps = 0.0
xStart =  1.2 + my_eps
xEnd =  1.35 - my_eps
yStart = 0.05 + my_eps
yEnd = 0.2 - my_eps
Norm_factor_perm =  2.
Norm_factor_out = 2.

#rng = np.random.default_rng(12345)  #testing

#all_indices = np.arange(0,num_files)
#test_indices = np.random.choice(all_indices , num_testing,replace=False)


if(Flag_grey):
 k_all =  np.zeros((num_files, nPt,nPt)) # all input data
 k_train = np.zeros((num_training, nPt,nPt)) # training data input
 k_test = np.zeros((num_testing, nPt,nPt)) # testing data input
else: #color
 k_all =  np.zeros((num_files,3, nPt,nPt)) # all input data
 k_train = np.zeros((num_training,3, nPt,nPt)) # training data input
 k_test = np.zeros((num_testing,3, nPt,nPt)) # testing data input


label_all= np.zeros((num_files,nPt*nPt)) #alldata label  
label_train = np.zeros((num_training,nPt*nPt)) #training data label  
label_test = np.zeros((num_testing,nPt*nPt)) #testing data label 


#strcutured grid:
x = np.linspace(xStart, xEnd, nPt)
y = np.linspace(yStart, yEnd, nPt)
x, y = np.meshgrid(x, y)
n_points = nPt * nPt


for i in range(num_files):


    mesh_file = file_prefix_in + str(i) +".vtu"
    #print ('Loading', mesh_file)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_file)
    reader.Update()
    data_vtk_in = reader.GetOutput()
    
    mesh_file_u = file_prefix_out + str(i) +".vtu"
    #print ('Loading', mesh_file_u)
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(mesh_file_u)
    reader.Update()
    data_vtk_out = reader.GetOutput()
    #Vel =  VN.vtk_to_numpy(data_vtk_out.GetPointData().GetArray( fieldname_u ))

    #n_points_mesh = data_vtk.GetNumberOfPoints()
    if (i ==0 ):
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
    #Vel_interped  = Vel_interped.reshape(nPt,nPt) / Norm_factor_out

    if(0):
        image_input = data_transform(image_input) 
    k_all[i,:,:] = image_input  
    
    label_all[i,:] = Vel_interped  / Norm_factor_out 



#plt.imshow(k_all[2229,:,:], cmap='rainbow', interpolation='nearest',vmin=0.5, vmax=1)
if(0):
 plt.imshow(k_all[2,:,:], cmap='rainbow', interpolation='nearest')
 plt.show()
 temp = label_all[2,:]
 plt.imshow(temp.reshape(nPt,nPt), cmap='rainbow', interpolation='nearest')
 plt.show()


#plt.plot(label_all) 
#plt.show()


K_train, K_test, label_train, label_test = train_test_split(k_all, label_all, test_size=num_testing , random_state=443)


#need the size to be [1,total_number_of_samples, npt,mpt]  1: channel (greyscale) #(batch_size, channels, width, height)
if (Flag_grey):
 K_train = np.expand_dims(K_train, axis=1)
 K_test = np.expand_dims(K_test, axis=1)
 #label_train = np.expand_dims(label_train, axis=1)
 #label_test = np.expand_dims(label_test, axis=1)

print('train shape',np.shape(K_train))
print('label shape',np.shape(label_train))
print('test shape',np.shape(K_test))
print('label test shape',np.shape(label_test))


Flag_cuda = False

if Flag_cuda:
 device = torch.device("cuda")
else:
 device = torch.device("cpu")
 #device = torch.device("mps")
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



'''
STEP 1: LOADING DATASET
'''
K_train = torch.Tensor(K_train).to(device)
K_test = torch.Tensor(K_test).to(device)
label_train = torch.Tensor(label_train).to(device)
label_test = torch.Tensor(label_test).to(device)
if (Flag_cuda):
 K_train = k_train.type(torch.cuda.FloatTensor)
 K_test = k_test.type(torch.cuda.FloatTensor)
 label_train = label_train.type(torch.cuda.FloatTensor)
 label_test = label_test.type(torch.cuda.FloatTensor)





batch_size = 64 #256 # 512 # 64 #32
#n_iters = 3000 * 50
#num_epochs = n_iters / (len(train_dataset) / batch_size)
#num_epochs = int(num_epochs)
num_epochs = 5000 #2000 * 2


train_dataset = TensorDataset(K_train,label_train)
test_dataset = TensorDataset(K_test,label_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=num_testing , 
                                          shuffle=False)







###################
'''
STEP 4: INSTANTIATE MODEL CLASS
'''

if (Flag_CNN):
    model = myCNN()
else:
    #model = DeepAutoencoder()
    model =  DeepAutoencoder_deeper()

#######################
#  USE GPU FOR MODEL  #
#######################


model.to(device)
model.apply(init_normal)



'''
STEP 5: INSTANTIATE LOSS CLASS
'''
criterion = nn.MSELoss()
#criterion = nn.L1Loss()
#criterion = nn.NLLLoss() #for when last layer is softmax https://neptune.ai/blog/pytorch-loss-functions


'''
STEP 6: INSTANTIATE OPTIMIZER CLASS
'''
learning_rate = 5e-5 / 2



#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.001/1000.) #"Improving Generalization Performance by Switching from Adam to SGD" by Nitish Shirish Keskar and Richard Socher
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate )

'''
STEP 7: TRAIN THE MODEL
'''


for epoch in range(num_epochs):
    model.train()
    training_loss = []
    iter = 0
    for i, (images, labels) in enumerate(train_loader):

        #######################
        #  USE GPU FOR MODEL  #
        #######################
        images = images.requires_grad_().to(device)
        labels = labels.to(device)

        # Clear gradients w.r.t. parameters
        #optimizer.zero_grad()
        model.zero_grad()

        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        training_loss.append(loss.item())

        iter += 1

    if epoch % 5 ==0:
            # Print Loss
            avg_loss = np.average(training_loss)
            print('epoch {}, Iteration: {}. Loss: {}'.format(epoch, iter, avg_loss))
            sys.stdout.flush()


    if (epoch % 400 == 0):
                torch.save(model.state_dict(),output_file)


    if epoch % 20 == 0 :
            model.eval()
            # Calculate Accuracy         
            #correct = 0
            total = 0
            loss_test_total = 0.
            # Iterate through test dataset
            for images, labels in test_loader:
                #######################
                #  USE GPU FOR MODEL  #
                #######################
                
                images = images.requires_grad_() #.to(device)
                #labels = labels.to(device)


                

                with torch.no_grad():
                    output_test = model(images)



                loss_test = criterion(output_test,labels)
                # Forward pass only to get logits/output
                #outputs = model(images)

                # Get predictions from the maximum value
                #_, predicted = torch.max(outputs.data, 1)


            


                output_test = output_test.cpu().data.numpy() #need to convert to cpu before converting to numpy
                labels = labels.cpu().data.numpy()

                #print('shape output', np.shape(output_test))
                #print('shape labels', np.shape(labels))

                #print('output', output_test)
                #print('labels', labels)

               


                #acc = ( output_test.cpu().reshape(-1).detach().numpy().round() == labels.cpu().reshape(-1).detach().numpy().round() ).mean() #https://medium.com/analytics-vidhya/pytorch-for-deep-learning-binary-classification-logistic-regression-382abd97fb43
                loss_test_total = loss_test_total + loss_test

                total +=1 


            # Print Loss
            loss_test_total  =  loss_test_total  / total 
            print('---Validation: epoch: {}. total {}, loss_test {}'.format(epoch, total,loss_test_total.item()))
            sys.stdout.flush()

            



torch.save(model.state_dict(),output_file)
print("Done!")

#softmax train/eval issues:   https://discuss.pytorch.org/t/how-to-prevent-very-large-values-in-final-linear-layer/147054/5
#https://discuss.pytorch.org/t/why-softmax-is-not-applied-on-the-network-output-during-the-evaluation/33448

#do sigmoid for evaluation: https://towardsdatascience.com/pytorch-tabular-binary-classification-a0368da5bb89


#Outcome:
#Train1: Very good validation error 





