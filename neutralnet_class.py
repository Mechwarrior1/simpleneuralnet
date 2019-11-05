from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
from __future__ import absolute_import, division, print_function, unicode_literals

#functions
def sigmoid(x,deriv=False): 
    sigmoid = 1 / (1 + np.exp(-x))
    if deriv==False:
        return sigmoid
    elif deriv==True:
        return sigmoid*(1-sigmoid)

def tanh(x,deriv=False): 
    if deriv==False:
        return np.tanh(x)
    elif deriv==True:
        return 1-np.tanh(x)**2

def relu(x,deriv=False):
    if deriv==False:
        return np.maximum(x,0)
    elif deriv==True:
        return np.maximum(x,0)

def softmax(x,deriv=False):
    expA = np.exp(x)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y


class neural_net():
    '''
    hidden layer takes sigmoid, tanh and relu
    final layer is softmax
    hidden_nodes=(3,3)
    '''
    def __init__ (self,funcs,hidden_nodes):
        self.funcs = funcs
        self.hidden_nodes=hidden_nodes
        self.matrix_size={}
        self.weights={}
        self.bias={}
        self.Z_matrix={}
        self.delta={}
        self.weights_best={}
        self.bias_best={}
        self.function_call={'sigmoid':sigmoid,'tanh':tanh,'relu':relu, 'softmax':softmax}
    
    def forward(self,X1=None): #forward propagation
        result1=0
        if isinstance(X1,np.ndarray)!=True:
            print('input for forward propagation must be np.ndarray')
        for i,func in enumerate(self.funcs):
            #print(func)
            if i == 0:
                weights1=self.weights['W0']
                dot_product=np.dot(X1,weights1)+self.bias['B0']
                #
                self.Z_matrix['Z'+str(i)]=self.function_call[func](dot_product,False)
            else:
                Z_matrix1=self.Z_matrix['Z'+str(i-1)]
                weights1=self.weights['W'+str(i)]
                dot_product=np.dot(Z_matrix1,weights1)+self.bias['B'+str(i)]
                #
                result1=self.function_call[func](dot_product,False)
                self.Z_matrix['Z'+str(i)]=result1
        return result1
    
    def fit(self,X1,T1,iter1=10000,rate=1e-5):
        '''
        builds the matrix for weights
        and trains
        '''
        for func in self.funcs:
            if func in self.function_call.keys():
                pass
            else:
                return(func+" not found, functions available:"+str(list(self.function_call.keys())))
        self.matrix_size['K0']=T1.shape[1]
        self.matrix_size['D0']=X1.shape[1]
        self.matrix_size['N0']=X1.shape[0]
        num_nodes=len(self.funcs)
        #create dimensions for matrix
        print("Building matrix dimensions")
        for i,nodes in enumerate(self.hidden_nodes):
            if i >= (len(self.hidden_nodes)-1): #skip last M, using K for the number of catagory after softmax
                pass
            else:
                self.matrix_size["M"+str(i)]=nodes
        #print(num_nodes)
        print("Building weights and bias matrix")
        for i, nodes in enumerate(self.funcs):#initiate matrix for weights and bias
            #print('weights '+str(i))
            if i == 0:
                self.weights['W'+str(i)]= np.random.randn(self.matrix_size['D0'], self.matrix_size['M0'])
                self.bias['B'+str(i)]=np.random.randn(self.matrix_size['M0'])
            elif i >= num_nodes-1:
                # print('last')
                self.weights['W'+str(num_nodes-1)]= np.random.randn(self.matrix_size['M'+str(num_nodes-2)], self.matrix_size['K0'])
                self.bias['B'+str(num_nodes-1)]=np.random.randn(self.matrix_size['K0'])
            else:
                self.weights['W'+str(i)]= np.random.randn(self.matrix_size['M'+str(i-1)], self.matrix_size['M'+str(i)])
                self.bias['B'+str(i)]=np.random.randn(self.matrix_size['M'+str(i)])
        cost_plot=[]
        accuracy2=0
        for epoch in range(iter1):
            if epoch % 200 == 0:
                Y1=self.forward(X1)
                # for val in self.Z_matrix.values(): #check Z1
                #     print(type(val)) 
                #print(Y1[0:5])
                if T1.shape[1]>1:
                    cost1=np.sum((T1)*np.log(Y1))#cross entropy
                    accuracy1=np.sum(np.argmax(T1,axis=1)==np.argmax(Y1,axis=1))/self.matrix_size['N0']
                else:
                    cost1=np.sum(T1*np.log(Y1)+(1-T1)*np.log(1-Y1)) 
                    accuracy1=np.sum(T1==Y1)/self.matrix_size['N0']
                cost_plot.append(cost1)
                
                print("Epoch: ",epoch, "| Cost: ",'%.2f' % cost1,", | Classification accuracy: ",'%.2f' % (100*accuracy1),"%")
                if accuracy1>accuracy2:
                    accuracy2=accuracy1
                    self.weights_best=self.weights
                    self.bias_best=self.bias
                #return(Y1)
            self.delta['D0']=(T1-Y1)
            for i in range(num_nodes): #-1 as softmax is calculated with logloss at the start
                #print('deriv'+str(i))
                i2=num_nodes-i-1 #Delta counts forward, the rest backwards
                delta1=self.delta['D'+str(i)] #D0 is for 1st calculation, 
                weights1=self.weights['W'+str(i2)] #while the rest uses the last matrix (W2 if 3 layers counting output layer)
                function1=self.function_call[self.funcs[i2]]
                if i2 != 0:
                    Z_matrix1=self.Z_matrix['Z'+str(i2-1)]
                else:
                    Z_matrix1=np.array([0,0]) #just a random matrix, but it will not be used when i2 == 0
                #print('D'+str(i),', W'+str(i2),', Z'+str(i2-1),', Func: '+self.funcs[i2])
                if i2 == 0: #need to skip last function
                    #print('weights: ',weights1.shape,', X1: ',X1.shape,', delta: ',delta1.shape)
                    self.weights['W'+str(i2)] += rate*np.dot(X1.T,delta1)
                    self.bias['B'+str(i2)] += rate*delta1.sum(axis=0)
                else:
                    #print('weights: ','W'+str(i2),weights1.shape,', delta: ','D'+str(i),delta1.shape,', Z_matrix: ','Z'+str(i2-1),Z_matrix1.shape,", Z': "+self.funcs[i2])
                    self.delta['D'+str(i+1)] = np.dot(delta1,weights1.T)*function1(Z_matrix1,True)
                    self.weights['W'+str(i2)] += rate*np.dot(Z_matrix1.T,delta1)
                    self.bias['B'+str(i2)] += rate*delta1.sum(axis=0)
        plt.plot(cost_plot)
        plt.show()
    
#test data, based on lazy programmer deep learning class
Nclass=500
D=2

X11 = np.random.randn(Nclass, D) + np.array([0, -2])
X12 = np.random.randn(Nclass, D) + np.array([2, 2])
X13 = np.random.randn(Nclass, D) + np.array([-2, 2])
X1 = np.vstack([X11, X12, X13])

Y1 = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
N1 = len(Y1) #creating targets, first 500 is 0, next 500 is 1, last 500 is 2
# turn Y into an indicator matrix for training
T1 = np.zeros((N1, 3))
for i in range(N1):
    T1[i, Y1[i]] = 1

########################### test 0
# ANN=neural_net(('sigmoid','tanh','softmax'),(3,6,3))
# ANN.fit(X1,T1,iter1=2,rate=1e-6)

########################### test 1

ANN=neural_net(('relu','sigmoid','softmax'),(3,7,3))
ANN.fit(X1,T1,iter1=300000,rate=1e-6)


#ANN.weights_best['W0']-ANN.weights['W0']
#ANN.bias_best

#original plot
plt.scatter(X1[:,0], X1[:,1], c=Y1, s=100, alpha=0.5)
plt.show()

#error plot
Y2=ANN.forward(X1)
Y3=(np.argmax(T1,axis=1)==np.argmax(Y2,axis=1))
plt.scatter(X1[:,0], X1[:,1], c=Y3, s=100, alpha=0.5)

########################## test 2
ANN=neural_net(('sigmoid','tanh','tanh','sigmoid','softmax'),(5,5,6,8,3))
ANN.fit(X1,T1,iter1=300000,rate=1e-6)
#error plot
Y2=ANN.forward(X1)
Y3=(np.argmax(T1,axis=1)==np.argmax(Y2,axis=1))
plt.scatter(X1[:,0], X1[:,1], c=Y3, s=100, alpha=0.5)

################ 3D plot for test 2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X1[:,0], X1[:,1], Y1,c='Black')

line = np.linspace(-7, 7, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
Yhat= ANN.forward(Xgrid)
#output, hidden = forward(X1, W1, b1, W2, b2)
Yhat=np.argmax(Yhat,axis=1).squeeze()#squeeze converts to single array
ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], np.array(Yhat),alpha=0.7,  #
    linewidth=0.2, antialiased=True)
