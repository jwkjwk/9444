'''
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

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.vocab import GloVe
import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """
    string = sample.replace('<br />', ' ')
    string = "".join([ c if c.isalnum() else " " for c in string ])
    return string.split()
    #processed = sample.split()

    return string

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    text = [word.lower() for word in sample]
    return text


def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    global vocab_size
    vocab_size = len(vocab)
    return batch


    vocab_number = vocab.freqs
    vocab_ID = vocab.itos

    def vonu(batchData, vocab_number, vocab_ID):
        count = 0
        for y in batchData:
            if vocab_number[vocab_ID[y]] < 3:
                batchData[count] = 0
                count += 1
        return

    for x in batch:
        vonu(x, vocab_number, vocab_ID)

    return batch

    global vocab_size
    vocab_size = len(vocab)
    return batch
stopWords = {'he', 'I', 'me', 'him', 'her', 'you','your','yours','mine', 'his','hers','himself','she', 'her', 'why', 'a', 'it', 'an','they','them','themselves','and','or','if','have','had','because','am','is','this','these','that','are','was','were','being','been','itself','when','where','how','we','who','why','while','ever','that','looking','for','on','with','the','has','and','to','one','here','couple','said','day','days','our','get','will','of','in','see','weeek','day','month','year','sat','sit','tell','but','my','ll','just','off','o','re','after','before','over','there','about','at','by','doing','do','does','did','because','between','which','whom','those','same','own','other','just','each','should','shouldn','didn','did','hasn','haven','isn','mightn','won','wouldn','ain','further','just','don','hadn','doesn','into', 't','m','whilst','forward','live','sit','find','review','install','company','floor','back','re','once','their','theirs','couldn','wasn','d','o','ve','through','during','against'
}
wordVectors = GloVe(name='6B', dim=50)

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
    #categoryOutput = torch.sum(categoryOutput, dim=1, keepdim=True)
    ratingOutput = torch.sum(ratingOutput, dim=1, keepdim=True)
    ratingOutput = ratingOutput.long()
    categoryOutput = categoryOutput.long()
    return ratingOutput, categoryOutput

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
    def __init__(self, vocab_size=100, input_size=50, hidden_size=128, n_layers=4,
                 dropout=0.2, output_size=128):
        super(network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.embedding = tnn.Embedding(vocab_size, self.input_size)
        self.lstm = tnn.LSTM(input_size, hidden_size, num_layers=n_layers,
                             bidirectional=True, batch_first=True)
        self.dropout = tnn.Dropout(dropout)
        self.fc = tnn.Linear(128 * 2, 128)
        self.fc2 = tnn.Linear(128,64)
        self.fc3 = tnn.Linear(64,1)
        self.softmax = tnn.Softmax(dim=1)
        self.gru = tnn.GRU(
            input_size=50,
            hidden_size=128,
            batch_first=True,
            num_layers=4,
            bidirectional=True,
            dropout=0.5
        )

    def forward(self, input, length):
        """
        Perform a forward pass of our model on some input and hidden state.
        """

        # x = input.long()
        # x = self.embedding(x)
        # length = length.view(self.n_layers*self.n_layers, batch_size, self.hidden_size)
        batch_size = input.size(0)
        x = pack_padded_sequence(input, length, batch_first=True)
        # length = torch.unsqueeze(length, )
        out, _ = self.lstm(x)
        # out = tnn.ReLU(out)
        # out = torch.squeeze(out)

        padded = pad_packed_sequence(out, batch_first=True, total_length=self.hidden_size)
        out, hidden = padded
        out = out.contiguous().view(-1, self.hidden_size)
        #lstm_out, (hn, cn) = self.lstm(input)
        #hidden = self.dropout(torch.cat((hn[-2,:,:], hn[-1,:,:]), dim=1))
        #out = self.fc(hidden.squeeze(0)).view(batchSize, -1)[:, -1]

        out = self.fc(out)
        out = self.softmax(out)
        out = out.view(batch_size, -1, self.output_size)

        # print(_.shape)
        # out = out.view(batch_size, -1)
        out = out[:, -1] # get last batch of labels

        batch_size = input.size(0)
        x = pack_padded_sequence(input, length, batch_first=True)
        # length = torch.unsqueeze(length, )
        out, _ = self.lstm(x)
        # out = tnn.ReLU(out)
        # out = torch.squeeze(out)

        padded = pad_packed_sequence(out, batch_first=True, total_length=self.hidden_size)
        out, hidden = padded
        out = out.contiguous().view(-1, self.hidden_size)
        out, hn = self.gru(input)

        out = out[:, -1, :]

        output = self.dropout((self.fc(out)))

        output = tnn.functional.relu(output)

        output = self.dropout(self.fc2(output))

        output = tnn.functional.relu(output)

        output = self.fc3(output)

        #return output.squeeze()
        return out.squeeze(), hidden

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.BCEWithLogitsLoss = tnn.BCEWithLogitsLoss()
        self.CrossEntropyLoss = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):

        ratingOutput = Variable(ratingOutput.float(), requires_grad=True)
        categoryOutput = Variable(categoryOutput.float(), requires_grad=True)
        categoryTarget = Variable(categoryTarget.float(), requires_grad=True)
        ratingLoss = self.CrossEntropyLoss(ratingOutput, ratingTarget)
        categoryLoss = self.BCEWithLogitsLoss(categoryOutput, categoryTarget)
        final_loss = ratingLoss + categoryLoss

        return final_loss

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 1
optimiser = toptim.SGD(net.parameters(), lr=0.1)
#optimiser = toptim.Adam(net.parameters(), lr=0.01)
'''
######################################
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

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext.vocab import GloVe
import numpy as np
# import sklearn

from config import device

################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    text = [word.lower() for word in sample]
    return text

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """
    #vocab_num - frequencies of tokens in the vocab
    #vocab_id - token strings indexed by their designated number
    vocab_num = vocab.freqs
    vocab_id = vocab.itos
    def vonu(batchData, vocab_num, vocab_id):
        x = 0
        for y in batchData:
            if vocab_num[vocab_id[y]] < 3:
                batchData[x] = 0
                x += 1
        return

    for z in batch:
        vonu(x, vocab_num, vocab_id)
    return batch

stopWords = {'he', 'I', 'me', 'him', 'her', 'you','your','yours','mine', 'his','hers','himself','she', 'her', 'why', 'a', 'it', 'an','they','them','themselves','and','or','if','have','had','because','am','is','this','these','that','are','was','were','being','been','itself','when','where','how','we','who','why','while','ever','that','looking','for','on','with','the','has','and','to','one','here','couple','said','day','days','our','get','will','of','in','see','weeek','day','month','year','sat','sit','tell','but','my','ll','just','off','o','re','after','before','over','there','about','at','by','doing','do','does','did','because','between','which','whom','those','same','own','other','just','each','should','shouldn','didn','did','hasn','haven','isn','mightn','won','wouldn','ain','further','just','don','hadn','doesn','into', 't','m','whilst','forward','live','sit','find','review','install','company','floor','back','re','once','their','theirs','couldn','wasn','d','o','ve','through','during','against'}
wordVectors = GloVe(name='6B', dim=200)

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
    # ratingOutput = torch.sum(ratingOutput, dim=1, keepdim=True)
    categoryOutput = torch.argmax(categoryOutput, dim=1)
    ratingOutput = ratingOutput.long()
    categoryOutput = categoryOutput.long()
    return ratingOutput, categoryOutput

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
    def __init__(self, vocab_size=97028 ,input_size=200, hidden_size=256, n_layers=2,
                 dropout=0.5, output_size1=1, output_size2=5, bidirectional=True):
        super(network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        # self.embedding = tnn.Embedding(vocab_size, self.input_size)
        self.lstm = tnn.LSTM(input_size, hidden_size, num_layers=n_layers,
                             bidirectional=bidirectional, batch_first=True)
        self.dropout = tnn.Dropout(dropout)
        self.fc1 = tnn.Linear(hidden_size*2, output_size1)
        self.fc2 = tnn.Linear(hidden_size*2, output_size2)
        self.sigmoid = tnn.Sigmoid()
        self.softmax = tnn.Softmax(dim=1)

    def forward(self, input, length):
        """
        Perform a forward pass of our model on some i

        nput and hidden state.
        """
        # x = input.long()
        # x = self.embedding(x)
        # length = length.view(self.n_layers*self.n_layers, batch_size, self.hidden_size)
        batch_size = input.size(0)
        # packed = pack_padded_sequence(input, length, batch_first=True)
        # print("length's shape is: " + str(length.shape))
        packed = pack_padded_sequence(input, length, batch_first=True)
        # length = torch.unsqueeze(length, )
        packed_out, (hidden, cell) = self.lstm(packed)
        # out = tnn.ReLU(out)
        # out = torch.squeeze(out)
        # padded = pad_packed_sequence(out, batch_first=True, total_length=self.hidden_size)
        out, out_len = pad_packed_sequence(packed_out)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        # out, hidden = padded
        # out = out.contiguous().view(-1, self.hidden_size)
        # out = self.dropout(out)
        # out = self.fc(out)

        # out = self.softmax(out)
        # out = out.view(batch_size, -1, self.output_size)
        # print(_.shape)
        out = out.view(batch_size, -1)
        # out = out[:, -1] # get last batch of labels
        out1 = self.fc1(hidden)
        out2 = self.fc2(hidden)
        out1 = self.sigmoid(out1)
        out2 = self.softmax(out2)
        # out1 = self.softmax(out1)
        # out1 = torch.argmax(out1, dim=1)
        # out2 = torch.round(torch.sum(out2, dim=1, keepdim=True))
        return out1, out2

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        self.BCELoss = tnn.BCELoss()
        self.NLLLoss = tnn.NLLLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        # categoryOutput = torch.round(torch.sum(torch.sigmoid(categoryOutput - 0.5)))
        ratingOutput = Variable(ratingOutput.float(), requires_grad=True).squeeze(1)
        ratingTarget = Variable(ratingTarget.float(), requires_grad=True)
        ratingTarget = ratingTarget.detach()
        # ratingOutput = torch.round(torch.sum(torch.sigmoid(ratingOutput - 0.5)))
        # categoryOutput = Variable(categoryOutput.float(), requires_grad=True)
        # categoryTarget = Variable(categoryTarget.float(), requires_grad=True)
        # categoryTarget = categoryTarget.detach()
        # print("rating shape is" + str(ratingOutput.shape))
        # print("category output is " + str(categoryOutput.shape))
        ratingLoss = self.BCELoss(ratingOutput, ratingTarget)
        categoryLoss = self.NLLLoss(categoryOutput, categoryTarget.long())
        final_loss = ratingLoss + categoryLoss

        return final_loss

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.8
batchSize = 32
epochs = 1
# optimiser = toptim.SGD(net.parameters(), lr=0.01)
optimiser = toptim.Adam(net.parameters(), lr=0.01)
