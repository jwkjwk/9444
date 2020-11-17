#!/usr/bin/env python3

"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as a basic
tokenise function.  You are encouraged to modify these to improve the
performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may only use GloVe 6B word vectors as found in the torchtext package.
"""

import torch
import torch.nn as tnn
import torch.optim as toptim
from torchtext.vocab import GloVe
# import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    #processed = sample.split()

    #return processed
    string = sample.replace('<br />', ' ')
    string = "".join([ c if c.isalnum() else " " for c in string ])
    return string.split()

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    return sample


def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    return batch

stopWords = {}
wordVectors = GloVe(name='6B', dim=300)

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    s = torch.argmax(ratingOutput, dim = 1)
    s1 = s + 1
    output1 = s1.float()
    ss = torch.argmax(categoryOutput, dim = 1)
    s2 = ss + 1
    output2 = s2.float()
    return (output1, output2)

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """
    #instantiate all my modules
    #access them using the names i've given here
    def __init__(self):

        super(network, self).__init__()

        #self.batchSize = 32
        self.hidSize = 200
        self.inputSize = 300
        self.layers = 2
        self.outDim = 5
        self.drop = 0.2
        self.lstm = tnn.LSTM(input_size = self.inputSize, hidden_size = self.hidSize, num_layers = self.layers, batch_first = True, bias = True, bidirectional = True)
        #self.relu = tnn.ReLU()
        #self.softmax = tnn.LogSoftmax()
        self.l1 = tnn.Linear(in_features = self.hidSize*2, out_features = 1)
        #self.l2 = tnn.Linear(in_features = self.outDim, out_features = self.hidSize)
        self.dropout = tnn.Dropout(p=self.drop)

    #forward function that defines the network structure
    def forward(self, input, length):
        batchsize, _, _ = input.size()
        lstmOut, (hn, cn) = self.lstm(input)
        hid = self.dropout(torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1))
        out = self.l1(hid.squeeze(0)).view(batchSize,-1)[:,-1]
        return out
class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """
    def __init__(self):
        super(loss, self).__init__()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        return ratingOutput.BCEWithLogitsLoss() + categoryOutput.BCEWithLogitsLoss()

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 10
optimiser = toptim.SGD(net.parameters(), lr=0.01)

#output = [batch_size, num_classes]



