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
import vtk
from vtk.util import numpy_support as VN

#plot the loss on CPU (first load the net)



def create_vtk(x,y):

	x = torch.Tensor(x).to(device)
	y = torch.Tensor(y).to(device)
	h_nD = 64  #for BC net
	h_D = 128 # for distance net
	h_n = 128 #for u,v,p
	input_n = 2 # this is what our answer is a function of. In the original example 3 : x,y,scale 



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

	class MySquared(nn.Module):
		def __init__(self, inplace=True):
			super(MySquared, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			return torch.square(x)

	class Net1_dist(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net1_dist, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n ,h_D),
				#nn.ReLU(),
				#nn.BatchNorm1d(h_n),
				Swish(),
			
				nn.Linear(h_D,h_D),
				#nn.ReLU(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				
				nn.Linear(h_D,h_D),
				#nn.ReLU(),
				#nn.BatchNorm1d(h_n),
				Swish(),
			

				nn.Linear(h_D,h_D),

				#nn.ReLU(),
				#nn.BatchNorm1d(h_n),
				Swish(),
			

				nn.Linear(h_D,h_D),

				#nn.ReLU(),
				#nn.BatchNorm1d(h_n),
				Swish(),
			

				nn.Linear(h_D,h_D),

				#nn.BatchNorm1d(h_n),
				Swish(),
			
				nn.Linear(h_D,h_D),

				#nn.BatchNorm1d(h_n),
				Swish(),
			
				nn.Linear(h_D,h_D),

				#nn.BatchNorm1d(h_n),
				Swish(),
			
				nn.Linear(h_D,h_D),



				nn.Linear(h_D,1),

				#nn.ReLU(), # make sure output is positive (does not work with PINN!)
				#nn.Sigmoid(), # make sure output is positive
				MySquared(),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output
	class Net1_bc_u(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net1_bc_u, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n ,h_nD),
				#nn.ReLU(),
				Swish(),
				nn.Linear(h_nD,h_nD),
				#nn.ReLU(),
				Swish(),
				nn.Linear(h_nD,h_nD),

				Swish(),
				nn.Linear(h_nD,h_nD),

				Swish(),
				nn.Linear(h_nD,h_nD),


				Swish(),
				nn.Linear(h_nD,h_nD),


				#nn.ReLU(),
				Swish(),

				nn.Linear(h_nD,1),

				nn.ReLU(), # make sure output is positive
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output
	class Net1_bc_v(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net1_bc_v, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n ,h_nD),
				#nn.ReLU(),
				Swish(),
				nn.Linear(h_nD,h_nD),
				#nn.ReLU(),
				Swish(),
				nn.Linear(h_nD,h_nD),

				Swish(),
				nn.Linear(h_nD,h_nD),

				Swish(),
				nn.Linear(h_nD,h_nD),

				Swish(),
				nn.Linear(h_nD,1),

				nn.ReLU(), # make sure output is positive
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			return  output


	def WSS(x,y):
		x.requires_grad = True
		y.requires_grad = True


		net_in = torch.cat((x,y),1)
		u = net2_u(net_in)  #evaluate model
		u = u.view(len(u),-1)
	
		u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

		return Diff*rho * u_y  #shear stress

	


	class Net2_u(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_u, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),


				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),


				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		#def forward(self,x):
		def forward(self,x):	
			output = self.main(x)
			#output_bc = net1_bc_u(x)
			#output_dist = net1_dist(x)
			if (Flag_BC_exact):
				output = output*(x- xStart) *(y- yStart) * (y- yEnd ) + U_BC_in + (y- yStart) * (y- yEnd )  #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
			#return  output * (y_in-yStart) * (y_in- yStart_up)
			#return output * dist_bc + v_bc
			#return output *output_dist * Dist_net_scale + output_bc
			return output

	class Net2_v(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_v, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),


				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		#def forward(self,x):
		def forward(self,x):	
			output = self.main(x)
			#output_bc = net1_bc_v(x)
			#output_dist = net1_dist(x)
			if (Flag_BC_exact):
				output = output*(x- xStart) *(x- xEnd )*(y- yStart) *(y- yEnd ) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
			#return  output * (y_in-yStart) * (y_in- yStart_up)
			#return output * dist_bc + v_bc
			#return output *output_dist * Dist_net_scale + output_bc
			return output

	class Net2_p(nn.Module):

		#The __init__ function stack the layers of the 
		#network Sequentially 
		def __init__(self):
			super(Net2_p, self).__init__()
			self.main = nn.Sequential(
				nn.Linear(input_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),
				#nn.BatchNorm1d(h_n),
				Swish(),
				nn.Linear(h_n,h_n),
				#nn.Tanh(),
				#nn.Sigmoid(),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),
				nn.Linear(h_n,h_n),

				#nn.BatchNorm1d(h_n),

				Swish(),

				nn.Linear(h_n,1),
			)
		#This function defines the forward rule of
		#output respect to input.
		def forward(self,x):
			output = self.main(x)
			#print('shape of xnet',x.shape) #Resuklts: shape of xnet torch.Size([batchsize, 2]) 
			if (Flag_BC_exact):
				output = output*(x- xStart) *(x- xEnd )*(y- yStart) *(y- yEnd ) + (-0.9*x + 1.) #modify output to satisfy BC automatically #PINN-transfer-learning-BC-20
			#return  (1-x[:,0]) * output[:,0]  #Enforce P=0 at x=1 #Shape of output torch.Size([batchsize, 1])
			return  output


	net2_u = Net2_u().to(device)
	net2_v = Net2_v().to(device)
	net2_p = Net2_p().to(device)
	#net1_dist = Net1_dist().to(device)
	#net1_bc_u = Net1_bc_u().to(device)
	#net1_bc_v = Net1_bc_v().to(device)


	def criterion_plot(x,y):

		#print (x)
		#x = torch.Tensor(x).to(device)
		#y = torch.Tensor(y).to(device)
		#t = torch.Tensor(t).to(device)

		#x = torch.FloatTensor(x).to(device)
		#x= torch.from_numpy(x).to(device)

		x.requires_grad = True
		y.requires_grad = True
		
		
		#net_in = torch.cat((x),1)
		net_in = torch.cat((x,y),1)
		u = net2_u(net_in)
		u = u.view(len(u),-1)
		v = net2_v(net_in)
		v = v.view(len(v),-1)
		P = net2_p(net_in)
		P = P.view(len(P),-1)

		#u = u * t + V_IC #Enforce I.C???
		#v = v * t + V_IC #Enforce I.C???

	
		
		u_x = torch.autograd.grad(u,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_xx = torch.autograd.grad(u_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		u_y = torch.autograd.grad(u,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		u_yy = torch.autograd.grad(u_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		v_x = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		v_xx = torch.autograd.grad(v_x,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		v_y = torch.autograd.grad(v,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]
		v_yy = torch.autograd.grad(v_y,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

		P_x = torch.autograd.grad(P,x,grad_outputs=torch.ones_like(x),create_graph = True,only_inputs=True)[0]
		P_y = torch.autograd.grad(P,y,grad_outputs=torch.ones_like(y),create_graph = True,only_inputs=True)[0]

		#u_t = torch.autograd.grad(u,t,grad_outputs=torch.ones_like(t),create_graph = True,only_inputs=True)[0]
		#v_t = torch.autograd.grad(v,t,grad_outputs=torch.ones_like(t),create_graph = True,only_inputs=True)[0]
		
		
		XX_scale = U_scale * (X_scale**2)
		YY_scale = U_scale * (Y_scale**2)
		UU_scale  = U_scale **2
	
		loss_2 = u*u_x / X_scale + v*u_y / Y_scale - Diff*( u_xx/XX_scale  + u_yy /YY_scale  )+ 1/rho* (P_x / (X_scale*UU_scale)   )  #X-dir
		loss_1 = u*v_x / X_scale + v*v_y / Y_scale - Diff*( v_xx/ XX_scale + v_yy / YY_scale )+ 1/rho*(P_y / (Y_scale*UU_scale)   ) #Y-dir
		loss_3 = (u_x / X_scale + v_y / Y_scale) #continuity



		# MSE LOSS
		loss_f = nn.MSELoss()

		#Note our target is zero. It is residual so we use zeros_like
		#loss = loss_f(loss_1,torch.zeros_like(loss_1))+ loss_f(loss_2,torch.zeros_like(loss_2))+loss_f(loss_3,torch.zeros_like(loss_3))

		final_loss = abs(loss_1) +  abs(loss_2) + abs(loss_3 ) #for plotting
		return final_loss, abs(loss_2), abs(loss_3 )

	print('load the network')
	net2_u.load_state_dict(torch.load(path+ NN_filename_prefix + "u.pt"))
	net2_v.load_state_dict(torch.load(path+ NN_filename_prefix + "v.pt"))
	net2_p.load_state_dict(torch.load(path+ NN_filename_prefix + "p.pt"))
	#net2_u.load_state_dict(torch.load(path+"stenBatch_u"+ ".pt"))
	#net2_v.load_state_dict(torch.load(path+"stenBatch_v"+ ".pt"))
	#net2_p.load_state_dict(torch.load(path+"stenBatch_p"+ ".pt"))


	net2_u.eval()
	net2_v.eval()
	net2_p.eval()

	if(Flag_plot): #test to make sure loads properly
		net_in = torch.cat((x.requires_grad_(),y.requires_grad_()),1)
		output_u = net2_u(net_in)  #evaluate model
		output_u = output_u.data.numpy() 
		
		plt.figure()
		plt.subplot(2, 1, 1)
		plt.scatter(x.detach().numpy(), y.detach().numpy(), c = output_u , cmap = 'rainbow')
		#plt.scatter(x.detach().numpy(), y.detach().numpy(), c = output_u ,vmin=0, vmax=0.58, cmap = 'rainbow')
		plt.title('NN results, u')
		plt.colorbar()
		plt.show()
		if(1): #plot vector field
			output_v = net2_v(net_in)  #evaluate model
			output_v = output_v.data.numpy() 
			# Normalize the arrows:
			U = output_u / np.sqrt(output_u**2 + output_v**2);
			V = output_v / np.sqrt(output_u**2 + output_v **2);
			plt.figure()
			fig, ax = plt.subplots(figsize=(9,9))
			skip=(slice(None,None,5),slice(None,None,5)) #plot every 5 pts
			#ax.quiver(x.detach().numpy(), y.detach().numpy(), output_u , output_v,scale=5)
			ax.quiver(x.detach().numpy()[skip], y.detach().numpy()[skip], U[skip], V[skip],scale=50)#a smaller scale parameter makes the arrow longer.
			plt.title('NN results, Vel vector')
			plt.show()
			#Streamline
			#plt.figure() 
			#fig2, ax2 = plt.subplots()
			#xs = np.linspace(xStart, xEnd, nPt)
			#ys = np.linspace(yStart_up, yEnd_up, nPt)
			#xs, ys = np.meshgrid(xs, ys)
			#ax2.streamplot(xs, ys, output_u , output_v,density = 0.5)
			#plt.title('NN results, Vel SL')
			#plt.show()

	if(Flag_plot): #test to make sure BC loads properly
		net_in = torch.cat((x.requires_grad_(),y.requires_grad_()),1)
		outputbc_u = net1_bc_u(net_in)  #evaluate model
		outputbc_u = outputbc_u.data.numpy() 
		plt.figure()
		plt.subplot(2, 1, 1)
		plt.scatter(x.detach().numpy(), y.detach().numpy(), c = outputbc_u , cmap = 'rainbow')
		#plt.scatter(x.detach().numpy(), y.detach().numpy(), c = outputbc_u ,vmin=0, vmax=0.5, cmap = 'rainbow')
		plt.title('NN results, BC u')
		plt.colorbar()
		plt.show()

	if(Flag_plot): #test to make sure distance loads properly
		net_in = torch.cat((x.requires_grad_(),y.requires_grad_()),1)
		outputbc_u = net1_dist(net_in)  #evaluate model
		outputbc_u = outputbc_u.data.numpy() 
		plt.figure()
		plt.subplot(2, 1, 1)
		plt.scatter(x.detach().numpy(), y.detach().numpy(), c = outputbc_u , cmap = 'rainbow')
		#plt.scatter(x.detach().numpy(), y.detach().numpy(), c = outputbc_u ,vmin=0, vmax=0.01, cmap = 'rainbow')
		plt.title('NN results, Distance')
		plt.colorbar()
		plt.show()

	if(Flag_plot): #Calculate WSS at the bottom wall
		xw = np.linspace(xStart + delta_wall , xEnd, nPt)
		yw = np.linspace(yStart , yStart, nPt)
		xw = np.reshape(xw, (np.size(xw[:]),1))
		yw = np.reshape(yw, (np.size(yw[:]),1))
		xw = torch.Tensor(xw).to(device)
		yw = torch.Tensor(yw).to(device)

		wss = WSS(xw,yw)
		wss = wss.data.numpy()

		plt.figure()
		plt.plot(xw.detach().numpy() , wss[0:nPt], 'go', label='Predict-WSS', alpha=0.5) #PINN
		plt.legend(loc='best')
		plt.show()

	if(Flag_plot): #Calculate near-wall velocity
		xw = np.linspace(xStart + delta_wall , xEnd, nPt)
		yw = np.linspace(yStart + 0.02 , yStart + 0.02, nPt)
		xw = np.reshape(xw, (np.size(xw[:]),1))
		yw = np.reshape(yw, (np.size(yw[:]),1))
		xw = torch.Tensor(xw).to(device)
		yw = torch.Tensor(yw).to(device)

		net_in = torch.cat((xw,yw),1)
		output_u = net2_u(net_in)  #evaluate model
		output_u = output_u.data.numpy() 

		plt.figure()
		plt.plot(xw.detach().numpy() , output_u[0:nPt], 'go', label='Near-wall vel', alpha=0.5) #PINN
		plt.legend(loc='best')
		plt.show()
		
	############### Convert network to VTK #################################
	print ('Loading', mesh_file)
	reader = vtk.vtkXMLUnstructuredGridReader()
	reader.SetFileName(mesh_file)
	reader.Update()
	data_vtk = reader.GetOutput()


	net_in = torch.cat((x.requires_grad_(),y.requires_grad_()),1)
	output_u = net2_u(net_in)  #evaluate model
	output_u = output_u.data.numpy() 
	output_v = net2_v(net_in)  #evaluate model
	output_v = output_v.data.numpy() 

	Velocity = np.zeros((n_points, 3)) #Velocity vector
	Velocity[:,0] = output_u[:,0] * U_scale
	Velocity[:,1] = output_v[:,0] * U_scale

	if(Flag_save_BC):
		outputbc_u = net1_bc_u(net_in)  #evaluate model
		outputbc_u = outputbc_u.data.numpy() 
		output_dist = net1_dist(net_in)  #evaluate model
		output_dist = output_dist.data.numpy() 
		BC_net = np.zeros((n_points, 1)) 
		BC_net[:,0] = outputbc_u[:,0]
		Dist_net = np.zeros((n_points, 1)) 
		Dist_net[:,0] = output_dist[:,0]

		#outputbc_v = net1_bc_v(net_in)  #evaluate model
		#outputbc_v = outputbc_v.data.numpy() 
		#BC_net_v = np.zeros((n_points, 1)) 
		#BC_net_v[:,0] = outputbc_v[:,0]
		
	Loss_plot, Loss_u, Loss_div = criterion_plot(x,y)
	Loss_plot = Loss_plot.data.numpy()
	Loss_net = np.zeros((n_points, 1)) 
	Loss_net[:,0] = Loss_plot[:,0]

	#Save VTK
	theta_vtk = VN.numpy_to_vtk(Velocity)
	theta_vtk.SetName('Vel_PINN')   #TAWSS vector
	data_vtk.GetPointData().AddArray(theta_vtk)

	#theta_vtk = VN.numpy_to_vtk(BC_net)
	#theta_vtk.SetName('BC_net')   #TAWSS vector
	#data_vtk.GetPointData().AddArray(theta_vtk)
	#theta_vtk = VN.numpy_to_vtk(Dist_net)
	#theta_vtk.SetName('dist_net')   
	#data_vtk.GetPointData().AddArray(theta_vtk)
	theta_vtk = VN.numpy_to_vtk(Loss_net)
	theta_vtk.SetName('loss_net')   
	data_vtk.GetPointData().AddArray(theta_vtk)

	output_p = net2_p(net_in)  #evaluate model
	output_p = output_p.data.numpy() 
	theta_vtk = VN.numpy_to_vtk(output_p)
	theta_vtk.SetName('P_PINN')   
	data_vtk.GetPointData().AddArray(theta_vtk)
	Loss_plot_u = Loss_u.data.numpy()
	Loss_net_u = np.zeros((n_points, 1)) 
	Loss_net_u[:,0] = Loss_plot_u[:,0]
	Loss_plot_div = Loss_div.data.numpy()
	Loss_net_div = np.zeros((n_points, 1)) 
	Loss_net_div[:,0] = Loss_plot_div[:,0]
	theta_vtk = VN.numpy_to_vtk(Loss_net_u)
	theta_vtk.SetName('loss_net_u')   
	data_vtk.GetPointData().AddArray(theta_vtk)
	theta_vtk = VN.numpy_to_vtk(Loss_net_div)
	theta_vtk.SetName('loss_net_div')   
	data_vtk.GetPointData().AddArray(theta_vtk)


	myoutput = vtk.vtkDataSetWriter()
	myoutput.SetInputData(data_vtk)
	myoutput.SetFileName(output_filename)
	myoutput.Write()


	print ('Done!' )


############## Set parameters here (make sure you are calling the appropriate network in the code. Network code needs to be compied here)
Diff = 0.001
rho = 1.
Flag_BC_exact = False 
device = torch.device("cpu")

Flag_plot = False #True: for also plotting in python
Flag_save_BC = False

#!!! Need to make the vtk mesh with Paraview 4.0 if we get Python VTK reading error

Flag_physical = True #IF True use the physical mesh, not the normalized dimension mesh

if (Flag_physical):
	mesh_file = "/home/aa3878/Data/ML/Amir/stenosis/sten_mesh_physical000000.vtu"
else:
	mesh_file = "/home/aa3878/Data/ML/Amir/stenosis/sten_mesh000000.vtu"
#output_filename = "/home/aa3878/Data/ML/Amir/stenosis/Results/stenosis_PINN_physical.vtk"
output_filename = "/home/aa3878/Data/ML/Amir/stenosis/Results/stenosis_PINN_data.vtk"

NN_filename_prefix = "sten_data_" # "sten_"  

X_scale = 2. 
Y_scale = 1.
U_scale = 1.0
U_BC_in = 0.5

if (not Flag_physical):
	X_scale = 1. 
	Y_scale = 1.



path = "Results/"


print ('Loading', mesh_file)
reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(mesh_file)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print ('n_points of the mesh:' ,n_points)
x_vtk_mesh = np.zeros((n_points,1))
y_vtk_mesh = np.zeros((n_points,1))
VTKpoints = vtk.vtkPoints()
for i in range(n_points):
	pt_iso  =  data_vtk.GetPoint(i)
	x_vtk_mesh[i] = pt_iso[0]	
	y_vtk_mesh[i] = pt_iso[1]
	VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)

print ('Net input max x:', np.max(x_vtk_mesh))
print ('Net input max y:', np.max(y_vtk_mesh))

x  = np.reshape(x_vtk_mesh , (np.size(x_vtk_mesh [:]),1)) / X_scale
y  = np.reshape(y_vtk_mesh , (np.size(y_vtk_mesh [:]),1)) / Y_scale

create_vtk(x,y)




