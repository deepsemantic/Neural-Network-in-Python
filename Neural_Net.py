'''
#Author: Cheng Wang
#Email: cheng.wang@hpi.de
#Date: 14/06/2015
#Last Update: 02/07/2016
#Neural Network in Python

'''

import numpy as np
import pdb
import scipy.io as sio 
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros

'''
implement a set of activation functions
derivative=true means in backward pass 
'''
def sgm(x,derivative=False):
  if not derivative:
    return 1/(1+np.exp(-x))
  else:
    out = sgm(x)
    return out*(1.0-out)

def linear(x,derivative=False):
  if not derivative:
    return x
  else:
    return 1.0

def guassian(x,derivative=False):
  if not derivative:
    return np.exp(-x**2)
  else:
    return -2*x*np.exp(-x**2)

def tanh(x,derivative=False):
  if not derivative:
    return np.tanh(x)
  else:
    return (1.0-np.tanh(x)**2)

def softmax(x,derivative=False):
   if not derivative:
       e_x = np.exp(x - np.amax(x, axis=0))
       return e_x / np.sum(e_x,axis=0)
   else:
       return 1.0

class NerualNetwork:
    
  '''class members'''
  layerCount = 0
  shape = None
  weights = []
  learningRate=0.001
  momentum=0.9
  weigt_decay=0.0001
  scaling_learningRate=0.1
  step_size=200 
  numepochs =  1000;   
  batchsize = 200;
  activationFunc=[]
  errors=[]
  loss=[]

  ''' Nerual Network Constructor'''
  def __init__(self,net,activationFunc):
    self.layerCount=len(net)
    self.shape=net
    self.activationFunc=activationFunc
    
    '''preactivation of each layer'''
    self.layerInput = [0]*self.layerCount 
    
    '''activation of each layer'''
    self.layerOutput = [0]*self.layerCount 
    
    '''preactivation of each layer in backprogation'''
    self.diff_layerInput = [0]*(self.layerCount-1) 
 
    self.weightDelta=[0]*(self.layerCount-1)
    
    #self.weightDelta = []
    for(l1,l2) in zip(net[:-1],net[1:]):
      #pdb.set_trace()
      print l1,'',l2
      
      '''randomly generate weights between layers
         weights size is the weight matirx between l1 and l2 the size is l2X(l1+1)'''
      self.weights.append(np.random.normal(scale=0.1,size=(l2,l1+1))) 
      
'''
train and test network
''' 
def nntrain(nn, train_X, train_Y):
  #number of training examples
  num_X=len(train_X)
  batchsize=nn.batchsize
  numepochs=nn.numepochs
  
  num_batches=num_X/batchsize
  
  L=np.zeros(shape=(numepochs*num_batches,1))
  n=1
  #pdb.set_trace()
  for i in range(numepochs):
    
    ''' shuffle training data
    create a random list of training dataset'''
    shuffle=np.random.permutation(num_X)
    #print shuffle
    for l in range(num_batches):
      '''get examples within each batch'''
      batch_X=train_X[shuffle[l*batchsize:(l+1)*batchsize]]
      batch_Y=train_Y[shuffle[l*batchsize:(l+1)*batchsize]]
      
      '''forward pass computation'''
      nn=nn_forward(nn,batch_X,batch_Y)
      
      '''backward pass compuation'''
      nn=nn_backward(nn)
      
      '''update weights'''
      nn=nngrads(nn)
      
    print 'iteraiton: ', i, 'loss:',nn.loss
    if i>0 and i%20==0:
      '''test network'''
      nn_test(nn)
      '''descrease learning rate'''
    if i>0 and i%nn.step_size==0:
      nn.learningRate=nn.learningRate*nn.scaling_learningRate

'''
to compute the gradient
'''
def nngrads(nn):
  for i in range(nn.layerCount-1):
    if nn.weigt_decay>0:
      dW=nn.weightDelta[i]+map(lambda x:x*nn.weigt_decay,nn.weights[i])
    else:
      dW=nn.weightDelta[i]     
    dW = map(lambda x: x * nn.learningRate, dW)
    nn.weights[i]=nn.weights[i]-dW
  return nn

'''
backward pass
(1) compute gradient of weights and bias
(2) update parameter
'''
def nn_backward(nn):
  
  n=nn.layerCount

  ''' output layer calucations'''
  if nn.activationFunc[n-1]=='sgm':
    nn.diff_layerInput[n-2]=-nn.errors*(sgm(nn.layerOutput[n-1],True))
    #print nn.diff_layerInput[n-1]
  elif nn.activationFunc[n-1]=='softmax':
    nn.diff_layerInput[n-2]=-nn.errors
    
  '''hidden layers compuation'''
  for i in range(n)[:1:-1]:
    if nn.activationFunc[i-1]=='sgm':
      d_act=sgm(nn.layerOutput[i-1],True)
    elif nn.activationFunc[i-1]=='tanh':
      d_act=1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn.layerInput[i]*nn.layerInput[i])
    
    ''' Backpropagate'''
    if i+1==n:
      nn.diff_layerInput[i-2]=(nn.weights[i-1].T).dot(nn.diff_layerInput[i-1])*d_act
    else:
      nn.diff_layerInput[i-2]=(nn.weights[i-1].T).dot(nn.diff_layerInput[i-1][:-1])*d_act

  for i in range(n-1):
    if i+2==n:
      nn.weightDelta[i]=(nn.diff_layerInput[i]).dot(nn.layerOutput[i].T)/len(nn.diff_layerInput[i].T)
    else:
      nn.weightDelta[i]=(nn.diff_layerInput[i][:-1]).dot(nn.layerOutput[i].T)/len(nn.diff_layerInput[i].T)
  return nn
  
'''
forward pass:
(1) compute pre-activation of each neuron
(2) compute loss
'''
def nn_forward(nn,X,Y):
    
  #number of layers
  n=nn.layerCount
  m=len(X) 
  # bias term
  bias=np.ones(shape=(1,m))  
  X=np.transpose(X)
  Y=np.transpose(Y)
  
  nn.layerInput[0]=np.vstack([X, bias])
  nn.layerOutput[0]=nn.layerInput[0]	
  
  
  ''' hidden layer calucations'''
  for i in range(1,n-1):  
    #print 'layer: ',i
    if nn.activationFunc[i]=='sgm':
      nn.layerInput[i]=nn.weights[i-1].dot(nn.layerInput[i-1])
      nn.layerOutput[i]=sgm(nn.layerInput[i])
    
    if nn.activationFunc[i]=='tanh':
      nn.layerInput[i]=nn.weights[i-1].dot(nn.layerInput[i-1])
      nn.layerOutput[i]=tanh(nn.layerInput[i])
    
    '''add bias items'''
    bias=np.ones(shape=(1,m)) #column 
    nn.layerOutput[i]=np.vstack([nn.layerOutput[i], bias])
    
  ''' output layer calucations'''
  if nn.activationFunc[n-1]=='sgm':
    nn.layerInput[n-1]=nn.weights[n-2].dot(nn.layerOutput[n-2])
    nn.layerOutput[n-1]=sgm(nn.layerInput[n-1])
    
  if nn.activationFunc[n-1]=='softmax':
    nn.layerInput[n-1]=nn.weights[n-2].dot(nn.layerOutput[n-2])    
    nn.layerOutput[n-1]=softmax(nn.layerInput[n-1])
  
  '''errors'''
  nn.errors=np.subtract(Y,nn.layerOutput[n-1])
  
  '''
  Loss
  for sgm, we consider mean square error
  for softmax, we use cross-entropy loss
  '''
  if nn.activationFunc[n-1]=='sgm':
    nn.loss=(nn.errors**2).sum()/nn.batchsize
    
  if nn.activationFunc[n-1]=='softmax':
    nn.loss=-((Y*(np.log(nn.layerOutput[n-1]))).sum()).sum()/nn.batchsize
  
  return nn
  
'''
test network on 
'''
def nn_test(nn):
  images,labels=load_mnist('testing', digits=np.arange(10))
  X_test=images.reshape((images.shape[0],images.shape[1]*images.shape[2]))
  Y_test=zeros((labels.shape[0], 10), dtype=int8)
  for idx in range(len(labels)):
    y=[0]*10
    y[labels[idx]]=1
    Y_test[idx]=y
  Y_pred=np.zeros_like(Y_test.T)
  correct_count=0
  accuracy=0.0
  Y_pred=[]
  for x_test,y_test,y in zip(X_test,Y_test,labels):
    x_test=x_test.reshape((1,x_test.shape[0]))
    y_test=y_test.reshape((1,y_test.shape[0]))
    nn=nn_forward(nn,x_test,y_test)
    y_pred=np.argmax(nn.layerOutput[nn.layerCount-1])
    Y_pred.append(nn.layerOutput[nn.layerCount-1])
    if y_pred==y[0]:
        correct_count=correct_count+1
    accuracy=correct_count/float(len(labels))
  print 'test loss:', nn.loss, '  acccuracy: ', accuracy, ' learning rate: ', nn.learningRate

'''
Loads MNIST files into 3D numpy arrays
code is adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
'''
def load_mnist(dataset="training", digits=np.arange(10), path="data"):
    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
        print fname_img
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    #print labels
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels
  
if __name__ == "__main__":
  '''configure network'''
  net=[784,1000,10]
  activations=['None','sgm','sgm']
  images,labels=load_mnist('training', digits=np.arange(10))
  X_train=images.reshape((images.shape[0],images.shape[1]*images.shape[2]))
  Y_train=zeros((labels.shape[0], 10), dtype=int8)
  for idx in range(len(labels)):
    y=[0]*10
    y[labels[idx]]=1
    Y_train[idx]=y
  print 'x shape:',X_train.shape
  print 'y shape:',Y_train.shape
  nn=NerualNetwork(net,activations)
  nntrain(nn,X_train,Y_train)
