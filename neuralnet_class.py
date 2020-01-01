import numpy as np
import copy
from matplotlib import pyplot as plt
from pandas import DataFrame
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
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
        return x*np.maximum(x,0)

def softmax(x,deriv=False):
    expA = np.exp(x)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y

class neural_net():
    '''
    hidden=(sigmoid,tanh,softmax), softmax is last layer
    hidden_nodes=(3,3)
    '''
    
    def __init__ (self,funcs,hidden_nodes,iter1=10000,rate1=1e-6):
        self.funcs = funcs
        self.hidden_nodes=hidden_nodes
        self.matrix_size={}
        self.weights={}
        self.bias={}
        self.Z_matrix={}
        self.delta={}
        self.weights_best={}
        self.bias_best={}
        self.function_call={'sigmoid':sigmoid,'tanh':tanh,'relu':relu, 'softmax':softmax,'linear':lambda x, y: x}
        self.iter1=iter1
        self.rate1=rate1
        self.adam_cache={}
    
    def predict(self,X1=None): #forward propagation
        #print(isinstance(self.funcs,str))
        if self.weights=={}:
            print('Weights are not initialized, it seems fitting was not done')
            return(0)
        result1=0
        # if isinstance(X1,np.ndarray)!=True:
        #     print('input for forward propagation must be np.ndarray')
        weights1=self.weights['W0']
        if isinstance(self.funcs,str):
            #print(Tr)
            weights1=self.weights['W0']
            dot_product=np.dot(X1,weights1)+self.bias['B0']
            if self.funcs!='linear':
                result1=self.function_call[self.funcs](dot_product,False)
                self.Z_matrix['Z0']=result1
                #print(result1)
            else:
                self.Z_matrix['Z0']=dot_product
                result1=dot_product
                #print(11)
        else:
            for i,func in enumerate(self.funcs):
                #print(func)
                if i == 0:
                    weights1=self.weights['W0']
                    dot_product=np.dot(X1,weights1)+self.bias['B0']
                    #
                    if func!='linear':
                        self.Z_matrix['Z'+str(i)]=self.function_call[func](dot_product,False)
                    else:
                        self.Z_matrix['Z'+str(i)]=dot_product
                else:
                    Z_matrix1=self.Z_matrix['Z'+str(i-1)]
                    weights1=self.weights['W'+str(i)]
                    dot_product=np.dot(Z_matrix1,weights1)+self.bias['B'+str(i)]
                    if func!='linear':
                        result1=self.function_call[func](dot_product,False)
                    else:
                        result1=dot_product
                    self.Z_matrix['Z'+str(i)]=result1
        #print(result1)
        return result1
    
    def fit(self,X1,T1,iter1=10000,rate=1e-5,reg=0,batches=10,decay_cache=0.99,momentum=0.9,cont=False):
        '''
        builds the matrix for weights
        and trains
        '''
        print1=True
        batch_sz=len(X1)/batches
        iter1=self.iter1
        rate=self.rate1
        eps=1e-10
        if isinstance(self.funcs,str):
            if self.funcs in self.function_call.keys():
                pass
        elif isinstance(self.funcs,(list,tuple)):
            for func in self.funcs:
                if func in self.function_call.keys():
                    pass
                else:
                    return(func+" not found, functions available:"+str(list(self.function_call.keys())))
        X1=np.array(X1)
        T1=np.array(T1)
        if len(T1.shape) ==1:
            T1=np.reshape(T1,(-1,1))
        if len(X1.shape) ==1:
            X1=np.reshape(X1,(-1,1)) #unflatten t
        if cont==False: #cont means continue building from past matrix
            self.matrix_size['K0']=T1.shape[1]
            self.matrix_size['D0']=X1.shape[1]
            self.matrix_size['N0']=X1.shape[0]
            if isinstance(self.hidden_nodes,int):
                if self.hidden_nodes!=self.matrix_size['K0']:
                    print('number of output node does not match number of target columns')
                    return
            elif self.hidden_nodes[len(self.funcs)-1]!=self.matrix_size['K0']: #check if last node matches output cols
                print('number of output node does not match number of target columns')
                return
            if isinstance(self.funcs,str):
                num_nodes=1
            else:
                num_nodes=len(self.funcs)
            #create dimensions for matrix
            #print("Building matrix dimensions")
            if isinstance(self.hidden_nodes,int): 
                pass
            else:
                for i,nodes in enumerate(self.hidden_nodes):
                    if i >= (len(self.hidden_nodes)-1): #skip last M, using K for the number of catagory/output columns
                        pass
                    else:
                        self.matrix_size["M"+str(i)]=nodes
            print("Building weights and bias matrix")
            if isinstance(self.funcs,str):
                self.weights['W0']= np.random.randn(self.matrix_size['D0'], self.matrix_size['K0']) /np.sqrt(self.matrix_size['K0'] * self.matrix_size['D0'])
                self.bias['B0']=np.random.randn(self.matrix_size['K0']) / np.sqrt(self.matrix_size['K0'])
            else:
                for i, nodes in enumerate(self.funcs):#initiate matrix for weights and bias
                    #print('weights '+str(i))
                    if i == 0:
                        self.weights['W'+str(i)]= np.random.randn(self.matrix_size['D0'], self.matrix_size['M0']) / np.sqrt(self.matrix_size['M0'] * self.matrix_size['D0'])
                        self.bias['B'+str(i)]=np.random.randn(self.matrix_size['M0'])/ np.sqrt(self.matrix_size['M0'])
                    elif i >= num_nodes-1:
                        # print('last')
                        self.weights['W'+str(num_nodes-1)]= np.random.randn(self.matrix_size['M'+str(num_nodes-2)], self.matrix_size['K0']) / np.sqrt(self.matrix_size['M'+str(num_nodes-2)]* self.matrix_size['K0'])
                        self.bias['B'+str(num_nodes-1)]=np.random.randn(self.matrix_size['K0']) / np.sqrt(self.matrix_size['K0'])
                    else:
                        self.weights['W'+str(i)]= np.random.randn(self.matrix_size['M'+str(i-1)], self.matrix_size['M'+str(i)]) / np.sqrt((self.matrix_size['M'+str(i-1)]* self.matrix_size['M'+str(i)]))
                        self.bias['B'+str(i)]=np.random.randn(self.matrix_size['M'+str(i)]) / np.sqrt(self.matrix_size['M'+str(i)])
            for i in range(num_nodes): #initialize adam cache and momentum
                i2=num_nodes-i-1
                self.adam_cache['mW'+str(i2)]=0
                self.adam_cache['mb'+str(i2)]=0
                self.adam_cache['vW'+str(i2)]=0
                self.adam_cache['vb'+str(i2)]=0
        cost_plot=[]
        accuracy2=0
        t=1
        print2=True
        for epoch in range(iter1):
            if epoch  % 100  == 0 : #and epoch != 0
                Y1=self.predict(X1)
                if self.matrix_size['K0']>1:
                    Y11=DataFrame(Y1)
                    Y11[Y11==0]=1e-10
                    T11=DataFrame(T1)
                    T11[T11==0]=1e-10
                    #print(Y1)
                    # global T1111 , Y1111
                    # T1111=T11
                    # Y1111=Y11
                    cost1=np.sum(np.sum((T11)*np.log(np.array(Y11))))#cross entropy
                    accuracy1=np.sum(np.argmax(np.array(T11),axis=1)==np.argmax(np.array(Y11),axis=1))/self.matrix_size['N0']
                else:
                    cost1=np.sum(Ybatch*np.log(Y1)+(1-Ybatch)*np.log(1-Y1))
                    # global cost123
                    # cost123 = cost1
                    if isinstance(cost1,(list,tuple)):
                        pass
                    elif str(cost1)=='nan':#for when the log receives 0 or less
                        Y11=copy.deepcopy(Y1)
                        T11=copy.deepcopy(T1)
                        Y11[Y11<=0]=1e-10
                        Y11[Y11>=0.9999999]=0.9999999
                        T11[T11==0]=1e-10
                        cost1=np.sum(T11*np.log(Y11)+(1-T11)*np.log(1-Y11))
                    accuracy1=np.sum(Ybatch==np.round(Y1))/self.matrix_size['N0']
                cost_plot.append(cost1)
                print("Epoch: ",epoch, "| Cost: ",'%.2f' % cost1,", | Classification accuracy: ",'%.2f' % (100*accuracy1),"%")
                if accuracy1>accuracy2:
                    accuracy2=accuracy1
                    self.weights_best=self.weights
                    self.bias_best=self.bias
            for bn in range(batches):
                slice_1=int(np.round(bn*batch_sz))
                slice_2=int(np.round(batch_sz))
                # if print2==True:
                #     print (slice_1, slice_2)
                Xbatch = X1[slice_1:( slice_1+ slice_2),:]
                Ybatch = T1[slice_1:( slice_1+ slice_2),:] 
                Y1=self.predict(Xbatch)
                    #return(Y1)
                self.delta['D0']=(Ybatch-Y1)
                for i in range(num_nodes): #-1 as softmax is calculated with logloss at the start
                    #print('deriv'+str(i))
                    #print(num_nodes)
                    i2=num_nodes-i-1 #Delta counts forward, the rest backwards
                    delta1=self.delta['D'+str(i)] #D0 is for 1st calculation, 
                    weights1=self.weights['W'+str(i2)] #while the rest uses the last matrix (W2 if 3 layers counting output layer)
                    bias1=self.bias['B'+str(i2)]
                    if isinstance(self.funcs,str):
                        function1=self.function_call[self.funcs] #a single softmax is recognised as a string not a list of funcs
                    else:
                        function1=self.function_call[self.funcs[i2]]
                    if i2 != 0:
                        Z_matrix1=self.Z_matrix['Z'+str(i2-1)]
                    else:
                        Z_matrix1=np.array([0,0]) #just a random matrix, but it will not be used when i2 == 0
                    #print('D'+str(i),', W'+str(i2),', Z'+str(i2-1),', Func: '+self.funcs[i2])
                    if i2 == 0: #need to skip last function
                        #print((0,i))
                        #print('weights: ',weights1.shape,', Xbatch: ',Xbatch.shape,', delta: ',delta1.shape)
                        deriv_temp_w = rate*(np.dot(Xbatch.T,delta1) - reg * weights1)
                        deriv_temp_b = rate*(delta1.sum(axis=0) - reg * bias1)
                        
                        #momentum
                        mW_temp=momentum * self.adam_cache['mW'+str(i2)] + (1 - momentum) * deriv_temp_w #momentum is the rate of which the old momentum is kept and how much the new deriv is added 
                        mb_temp=momentum * self.adam_cache['mb'+str(i2)] + (1 - momentum) * deriv_temp_b
                        self.adam_cache['mW'+str(i2)]=mW_temp
                        self.adam_cache['mb'+str(i2)]=mb_temp
                        correction_m = 1 - momentum ** t
                        hat_mW_temp = mW_temp / correction_m
                        hat_mb_temp = mb_temp / correction_m
                        
                        #cache or 2nd movement of mean error (mean error squared)
                        #it decays away part of the old cache, and adds 1-decay_rate of the squared error 
                        #cache is to reduce the rate of learning when it becomes big, eg learning is too fast or over a longer period
                        vW_temp = decay_cache * self.adam_cache['vW'+str(i2)] + (1 - decay_cache) * deriv_temp_w * deriv_temp_w
                        vb_temp = decay_cache * self.adam_cache['vb'+str(i2)] + (1 - decay_cache) * deriv_temp_b * deriv_temp_b
                        self.adam_cache['vW'+str(i2)]=vW_temp
                        self.adam_cache['vb'+str(i2)]=vb_temp
                        correction_v = 1 - decay_cache ** t
                        hat_vW_temp = vW_temp / correction_v
                        hat_vb_temp = vb_temp / correction_v
                        
                        #update weights and bias
                        self.weights['W'+str(i2)] += rate * hat_mW_temp / np.sqrt(hat_vW_temp + eps)
                        self.bias['B'+str(i2)] += rate * hat_mb_temp / np.sqrt(hat_vb_temp + eps)
                    else:
                        #print((1,i))
                        #print('weights: ','W'+str(i2),weights1.shape,', delta: ','D'+str(i),delta1.shape,', Z_matrix: ','Z'+str(i2-1),Z_matrix1.shape,", Z': "+self.funcs[i2])
                        self.delta['D'+str(i+1)] = np.dot(delta1,weights1.T)*function1(Z_matrix1,True)
                        deriv_temp_w = rate*(np.dot(Z_matrix1.T,delta1) - reg * weights1)
                        deriv_temp_b = rate*(delta1.sum(axis=0) - reg * bias1)
                        
                        #momentum
                        mW_temp=momentum * self.adam_cache['mW'+str(i2)] + (1 - momentum) * deriv_temp_w
                        mb_temp=momentum * self.adam_cache['mb'+str(i2)] + (1 - momentum) * deriv_temp_b
                        self.adam_cache['mW'+str(i2)]=mW_temp
                        self.adam_cache['mb'+str(i2)]=mb_temp
                        correction_m = 1 - momentum ** t
                        hat_mW_temp = mW_temp / correction_m
                        hat_mb_temp = mb_temp / correction_m
                        
                        #cache 
                        vW_temp = decay_cache * self.adam_cache['vW'+str(i2)] + (1 - decay_cache) * deriv_temp_w * deriv_temp_w
                        vb_temp = decay_cache * self.adam_cache['vb'+str(i2)] + (1 - decay_cache) * deriv_temp_b * deriv_temp_b
                        self.adam_cache['vW'+str(i2)]=vW_temp
                        self.adam_cache['vb'+str(i2)]=vb_temp
                        correction_v = 1 - decay_cache ** t
                        hat_vW_temp = vW_temp / correction_v
                        hat_vb_temp = vb_temp / correction_v
                        
                        #update weights and bias
                        self.weights['W'+str(i2)] += rate * hat_mW_temp / np.sqrt(hat_vW_temp + eps)
                        self.bias['B'+str(i2)] += rate * hat_mb_temp / np.sqrt(hat_vb_temp + eps)
                t+=1
            #     if print1==True:
            #         print(t)
            #         print(bn)
            # print1=False    
            # plt.plot(cost_plot)
            # plt.show()
            print2=False

#test data, based on lazy programmer deep learning class
Nclass=500
D=3

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
