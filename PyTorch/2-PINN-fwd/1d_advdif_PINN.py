import torch
import numpy as np
#import foamFileOperation
from matplotlib import pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
#from torchvision import datasets, transforms
import csv
from torch.utils.data import DataLoader, TensorDataset,RandomSampler
from math import exp, sqrt,pi
import time
import math

#Solve 1D linear steady advection-diffusion eqn with Vel and Diff give:   Vel*  \partial C / \partialx = Diff * C''
#Note: Making wider and reducing learning rate makes it work for higher Pe numbers

def mytrain(device,x_in,xb,cb,batchsize,learning_rate,epochs,path,Flag_batch,C_analytical,Vel,Diff,Flag_BC_exact ):
	if (Flag_batch): 
	 x = torch.Tensor(x_in)
	 dataset = (x) #TensorDataset(x)
	 dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True,num_workers = 0,drop_last = False )
	else:
	 x = torch.Tensor(x_in)  
	h_nD = 30
	h_n = 10 * 4  #20
	input_n = 1 # this is what our answer is a function of. In the original example 3 : x,y,scale 
	class Swish(nn.Module):
		def __init__(self, inplace=True):
			super(Swish, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			if self.inplace:
				x.mul_(torch.sigmoid(x))
				return x
			else:
				return x * torch.sigmoid(x)
	
	class Net2(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			if (Flag_BC_exact):
				output = output*x*(x-1) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
				#output = output*x*(1-x) + torch.exp(math.log(0.1)*x) #Do it exponentially? Not as good
			return  output

	
	################################################################

	net2 = Net2().to(device)


	###### Initialize the neural network using a standard method ##############
	def init_normal(m):
		if type(m) == nn.Linear:
			nn.init.kaiming_normal_(m.weight)

	# use the modules apply function to recursively apply the initialization
	net2.apply(init_normal)

	############################################################

	optimizer2 = optim.Adam(net2.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)


	###### Definte the PDE and physics loss here ##############
	def criterion(x):

		#print (x)
		x = torch.Tensor(x).to(device)
		#x = torch.FloatTensor(x).to(device)
		#x= torch.from_numpy(x).to(device)

		x.requires_grad = True
		
		#net_in = torch.cat((x),1)
		net_in = x
		C = net2(net_in)
		C = C.view(len(C),-1)
		if(0):
			C = C*(x-1) + 0.1 #Enfore BC


		
		c_x = torch.autograd.grad(C,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		c_xx = torch.autograd.grad(c_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		
		loss_1 = Vel * c_x - Diff * c_xx




		# MSE LOSS
		loss_f = nn.MSELoss()

		#Note our target is zero. It is residual so we use zeros_like
		loss = loss_f(loss_1,torch.zeros_like(loss_1)) 

		return loss

	###### Define boundary conditions ##############
	def Loss_BC(xb,cb):
		xb = torch.FloatTensor(xb).to(device)
		cb = torch.FloatTensor(cb).to(device)

		#xb.requires_grad = True
		
		#net_in = torch.cat((xb),1)
		out = net2(xb)
		cNN = out.view(len(out), -1)
		#cNN = cNN*(1.-xb) + cb    #cNN*xb*(1-xb) + cb
		loss_f = nn.MSELoss()
		loss_bc = loss_f(cNN, cb)
		return loss_bc


	######## Main loop ###########

	tic = time.time()

	if(Flag_batch):# This one uses dataloader
		for epoch in range(epochs):
			for batch_idx, (x_in) in enumerate(dataloader):
		
				net2.zero_grad()
				loss_eqn = criterion(x_in)
				loss_bc = Loss_BC(xb,cb)
				if (Flag_BC_exact):
					loss = loss_eqn #+ loss_bc
				else:
					loss = loss_eqn + loss_bc
				loss.backward()
		
				optimizer2.step() 
				if batch_idx % 10 ==0:
					print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}'.format(
						epoch, batch_idx * len(x), len(dataloader.dataset),
						100. * batch_idx / len(dataloader), loss.item()))
					
				#if epoch %100 == 0:
				#	torch.save(net2.state_dict(),path+"geo_para_axisy_sigma"+str(sigma)+"_epoch"+str(epoch)+"hard_u.pt")
	else:
		for epoch in range(epochs):
			#zero gradient
			#net1.zero_grad()
			##Closure function for LBFGS loop:
			#def closure():
			net2.zero_grad()
			loss_eqn = criterion(x)
			loss_bc = Loss_BC(xb,cb)
			if (Flag_BC_exact):
				loss = loss_eqn #+ loss_bc
			else:
				loss = loss_eqn + loss_bc
			loss.backward()
			#return loss
			#loss = closure()
			#optimizer2.step(closure)
			#optimizer3.step(closure)
			#optimizer4.step(closure)
			optimizer2.step() 
			if epoch % 5 ==0:
				print('Train Epoch: {} \tLoss: {:.10f}'.format(
					epoch, loss.item()))
				

	toc = time.time()
	elapseTime = toc - tic
	print ("elapse time = ", elapseTime)
	###################
	#plot
	output = net2(x)  #evaluate model
	C_Result = output.data.numpy()
	plt.figure()
	plt.plot(x.detach().numpy(), C_analytical[:], '--', label='True data', alpha=0.5) #analytical
	plt.plot(x.detach().numpy() , C_Result, 'go', label='Predicted', alpha=0.5) #PINN
	plt.legend(loc='best')
	plt.show()

	return net2


#######################################################
#Main code:
device = torch.device("cpu")
epochs  = 6000 

Flag_batch =False  #Use batch or not 
Flag_Chebyshev = False #Use Chebyshev pts for more accurcy in BL region
Flag_BC_exact = True #If True enforces BC exactly HELPS ALOT here!!!


## Parameters###
Vel = 1.0
Diff =  0.01 

nPt = 100 
xStart = 0.
xEnd = 1.

if(Flag_Chebyshev): #!!!Not a very good idea (makes even the simpler case worse)
 x = np.polynomial.chebyshev.chebpts1(2*nPt)
 x = x[nPt:]
 if(0):#Mannually place more pts at the BL 
    x = np.linspace(0.95, xEnd, nPt)
    x[1] = 0.2
    x[2] = 0.5
 x[0] = 0.
 x[-1] = xEnd
 x = np.reshape(x, (nPt,1))
else:
 x = np.linspace(xStart, xEnd, nPt)
 x = np.reshape(x, (nPt,1))


print('shape of x',x.shape)

#boundary pt and boundary condition
#X_BC_loc = 1.
#C_BC = 1.
#xb = np.array([X_BC_loc],dtype=np.float32)
#cb = np.array([C_BC ], dtype=np.float32)
C_BC1 = 1.
C_BC2 = 0.1
xb = np.array([0.,1.],dtype=np.float32)
cb = np.array([C_BC1,C_BC2 ], dtype=np.float32)
xb= xb.reshape(-1, 1) #need to reshape to get 2D array
cb= cb.reshape(-1, 1) #need to reshape to get 2D array
#xb = np.transpose(xb)  #transpose because of the order that NN expects instances of training data
#cb = np.transpose(cb)


batchsize = 32
learning_rate = 1e-3 






path = "Results/"

#Analytical soln
A = (C_BC2 - C_BC1) / (exp(Vel/Diff) - 1)
B = C_BC1 - A
C_analytical = A*np.exp(Vel/Diff*x[:] ) + B



#path = pre+"aneurysmsigma01scalepara_100pt-tmp_"+str(ii)
net2_final = mytrain(device,x,xb,cb,batchsize,learning_rate,epochs,path,Flag_batch,C_analytical,Vel,Diff,Flag_BC_exact )
#tic = time.time()

#elapseTime = toc - tic
#print ("elapse time in serial = ", elapseTime)

 








