# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader


class NeuralNet(nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """

        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn


        #convolutional layers
        self.conv1 = nn.Conv2d(3, 31, (5,5))
        #self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(31, 31, (5,5))
        #self.relu2 = nn.ReLU()
        
      
        
        #fully connected layers
        self.fc1 = nn.Linear(31*11*11, 120)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(120, out_size)

        #self.seq = nn.Sequential(self.fc1,self.relu, self.fc2)
        
        self.seq = nn.Sequential(
            self.conv1, self.relu,
            self.conv2, self.relu, self.pool, 
            nn.Flatten(),
            self.fc1, self.relu, self.fc2
        )
        
        #weight decay controls amount of L2 regularization
        self.optimizer = optim.SGD(self.parameters(), lr = lrate, weight_decay=1e-4)
 
    

    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """

        #reshape data here, CNN will expected data with shape (N, 2883)

        x = x.view(-1, 3, 31, 31)
        return self.seq(x)


        #return self.seq(x)

        #raise NotImplementedError("You need to write this part!")
        #return torch.ones(x.shape[0], 1)

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """

        self.optimizer.zero_grad()

        output_from_forward = self.forward(x)
        #get loss from function
        loss = self.loss_fn(output_from_forward, y)
        #now backward
        loss.backward()
        self.optimizer.step()
        

        return loss.item()

        #raise NotImplementedError("You need to write this part!")
        #return 0.0



def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """
    loss_fn = nn.CrossEntropyLoss()
    net = NeuralNet(0.01, loss_fn,2883,4)
    #raise NotImplementedError("You need to write this part!")

    losses = []
    yhats = []

    #standardize training and dev sets using X:=(X−μ)/σ from MP description
    s_train_set = (train_set - train_set.mean()) / (train_set.std())
    s_dev_set = (dev_set - dev_set.mean()) / (dev_set.std())
    data_set = get_dataset_from_arrays(s_train_set,train_labels)
    
    #create data loader object
    data_load = DataLoader(data_set,batch_size,shuffle=F)
    
    for epoch in range(epochs):
        #get inputs and labels, according to source we were given - DIFFERENT NOW
        for d in data_load:
            batch_inputs,batch_labels = d.keys()

            #get loss for this epoch
            loss = net.step(d[batch_inputs], d[batch_labels])

            #append to array
            #print(loss)
            losses.append(loss)
        
    for item in s_dev_set:
        y_val = torch.argmax(net.forward(item))
        #print(y_val.item())
        yhats.append(y_val.item())
    yhats = np.array(yhats)


    return losses, yhats, net


    #raise NotImplementedError("You need to write this part!")
    #return [],[],None
