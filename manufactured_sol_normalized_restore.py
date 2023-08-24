
import deepxde as dde
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import torch

# Exponential Coefficients in Diffusivity Function to be Determined in Inverse Problem
a = dde.Variable(0.479)
b = dde.Variable(0.481)
c = dde.Variable(0.934)
d0 = dde.Variable(14.4)

def get_derivative(x,y):
        return dde.grad.jacobian(y, x, i=0, j=0)

def pde(x, y):
    y1 = 0.25 + y*(0.75-0.25)
    fac = 99.996996996997/(600**2)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_x = dde.grad.jacobian(y,x,i=0,j=0)
    dy_xx = dde.grad.jacobian(d0*torch.exp(a*y1 + b*y1*y1 + c*y1*y1*y1)*dy_x,x,i=0,j=0)#Added variable Diffusivity in Fick's Second Law
    return (dy_t-fac*dy_xx)

def ZeroBC(x):
    return np.zeros((len(x),1))


def initial_function(x):
	a = [  78.00381484,   79.1421714 ,   -6.59283738,  115.41147907, 15.65909495,   65.99942936,   -5.92257449,    0.59154341,39.67688763,   40.61773514,   -0.98953989,   15.39118452, 21.61617447,  -51.21424868,    5.70258697,   -0.35526709, 11.49888749,   40.2872876 ,   -0.99821871,   23.8835267 ,  2.58645872,   40.6934871 ,   -0.97806312,  -71.74983323, 13.90591678,  -28.31102516,    1.21773476,   50.18835433, 2.48766649,   46.54523698,  -37.47353693,   12.77153302, -27.7348712 ,  -39.86578307,    1.04389541,  -52.51373979, -13.93288233,  -49.36908251,  -37.44117445,  -39.05337158, -9.71739626,  -13.55821678,   30.80322409,   -8.00642238, 33.89143859,  120.36539567,   14.21916177, -123.55440799, 9.07902025]
	res = a[0]
	for i in range(1,len(a),4):
        	res += a[i]*np.sin(a[i+1]*x[:,0:1] )+a[i+2]*np.cos(a[i+3]*x[:,0:1] )
	return res
        
def midpoint_constraint(inputs,outputs,X):
        y = 0.25 + outputs*(0.75-0.25)
        return 12.012012012012011*torch.exp(y+0.5*y*y+0.25*y*y*y)-d0*torch.exp(a*y + b*y*y + c*y*y*y)
       
        


start = time.time()

geom = dde.geometry.Interval(0, 1.0)
timedomain = dde.geometry.TimeDomain(0.15015465929907804,1.0)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

bc = dde.icbc.NeumannBC(geomtime, ZeroBC, lambda _, on_boundary: on_boundary)#Here we need to have Homogeneous Neumann BCs
ic = dde.icbc.IC(geomtime, initial_function, lambda _, on_initial: on_initial)#Here the initial condition is applied

#Reading the Data
df = pd.read_fwf('RawData_Scaled.txt',header=None)
df = df[0].str.split(' ', expand=True)
df.columns = ['Position','Time','Composition']
observe_x = np.vstack((df['Position'].to_numpy(),df['Time'].to_numpy())).T
observe_x = observe_x.astype(float)
observe_x[:,0] = observe_x[:,0]/np.max(observe_x[:,0])
observe_x[:,1] = observe_x[:,1]/np.max(observe_x[:,1])
comp_y = df['Composition'].to_numpy()
comp_y = comp_y.astype(float)
comp_y = comp_y.reshape((len(comp_y),1))
comp_y = (comp_y-0.25) / (0.75-0.25)
observe_y = dde.icbc.PointSetBC(observe_x,comp_y, component=0)
constraint = dde.OperatorBC(geom,midpoint_constraint,lambda x,_: dde.utils.isclose(x[0],0.5))
data = dde.data.TimePDE(
    geomtime,
    pde,
    [bc,ic,observe_y,constraint],
    num_domain=1000,
    num_boundary=20,
    num_initial=1000,
    anchors=observe_x
    #num_test=10000
)
    
layer_size = [2] + [50] * 8 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)
#net.apply_output_transform(modify_output)
model = dde.Model(data, net)
weights = [1]+[10,1,1,1]
model.compile(
    "adam",lr = 0.001,  external_trainable_variables=[a,b,c,d0],loss_weights = weights
)
model.restore("NormalizedDiffusionModelLateStart-1000000.pt", verbose=1)
variable = dde.callbacks.VariableValue([a,b,c,d0], period=1000,filename="Trained_Variable.dat")
losshistory, train_state = model.train(iterations=1000000,callbacks=[variable],model_save_path="NormalizedDiffusionModelLateStartRestore")
end = time.time()
print(f'Elapsed Time : {(end - start)/60.0} mins')
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

#Get the prediction from model and compare it with final composition
comp_NN = model.predict(observe_x)
plt.plot(observe_x[:,0:1],comp_NN,'r-',label='NNSol')
plt.plot(observe_x[:,0:1],comp_y,label='FDMSol')
plt.title('FinalCompositionComparison - FDM Vs NN')
plt.legend()
plt.savefig('FinalProfile.png')
plt.show()
f = model.predict(observe_x, operator=get_derivative)
np.savetxt(f"Derivative.dat",np.hstack((observe_x,f)))

#Get Initial Profile Comparative Plot
PosInit = np.vstack((observe_x[:,0],np.full(observe_x[:,0].size,0.0))).T
CompInit = model.predict(PosInit)
plt.plot(PosInit[:,0:1],initial_function(PosInit),'k-',label=' tanh Profile')
plt.plot(PosInit[:,0:1],CompInit,'c-',label='InitProfile After Epoches')
plt.legend()
plt.savefig('Fitted_Initial_Profile.png')
plt.show()

            

