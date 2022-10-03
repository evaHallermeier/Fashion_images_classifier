#Eva Hallermeier, Shana Sebban
#337914121, 337912182

import numpy as np
import sys
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
import torch.utils.data as DataLoader
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader


train_average_loss = []
train_average_accuracy = []
validation_average_accuracy = []
validation_average_loss =[]


#*****************************************************************************************
# Class Name: Model_A
# Class Details:   _ 2 hidden layers [100,50,10]
#                  _ activation: RELU
#                  _ optimizer: SGD
#                  _ output with softmax
#*****************************************************************************************
class Model_A(nn.Module):

    def __init__(self):
        super(Model_A, self).__init__()
        self.image_size = 784  # 28*28
        # layer = (nb of input neurons, number of output neurons)
        # 2 hidden layers 100 and 50
        self.fc0 = nn.Linear(self.image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)



#*****************************************************************************************
# Class Name: Model_B
# Class Details:   _ 2 hidden layers [100,50,10]
#                  _ activation: RELU
#                  _ optimizer: ADAM
#                  _ output with softmax
#*****************************************************************************************
class Model_B(nn.Module):

    def __init__(self):
        super(Model_B, self).__init__()
        self.image_size = 784  # 28*28
        # layer = (nb of input neurons, number of output neurons)
        self.fc0 = nn.Linear(self.image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


# *****************************************************************************************
# Class Name: Model_C
# Class Details:   _ 2 hidden layers [100,50,10]
#                  _ activation: RELU
#                  _ optimizer: ADAM
#                  _ dropout on the output of hidden layers
#                  _ output with softmax
# *****************************************************************************************
class Model_C(nn.Module):

    def __init__(self):
        super(Model_C, self).__init__()
        self.image_size = 784  # 28*28
        # layer = (nb of input neurons, number of output neurons)
        self.fc0 = nn.Linear(self.image_size, 100)
        self.drop0 = nn.Dropout(p=0.25)  
        self.fc1 = nn.Linear(100, 50)
        self.drop1 = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = self.drop0(x)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def optimizer(self, params, learning_rate):
        optimizer = optim.Adam(params, learning_rate)
        return optimizer


# *****************************************************************************************
# Class Name: Model_D
# Class Details:   _ 2 hidden layers [100,50,10]
#                  _ activation: RELU
#                  _ optimizer: ADAM
#                  _ batch normalization
#                  _ output with softmax
# *****************************************************************************************
class Model_Da(nn.Module): #with batch normalization before activation

    def __init__(self):
        super(Model_Da, self).__init__()
        self.image_size = 784  # 28*28
        # layer = (nb of input neurons, number of output neurons)
        self.fc0 = nn.Linear(self.image_size, 100)
        self.bn0 = nn.BatchNorm1d(100)  
        self.fc1 = nn.Linear(100, 50)
        self.bn1 = nn.BatchNorm1d(50)   
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.bn0(self.fc0(x)))
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



class Model_Db(nn.Module): #with batch normalization after activation

    def __init__(self):
        super(Model_Db, self).__init__()
        self.image_size = 784  # 28*28
        # layer = (nb of input neurons, number of output neurons)
        self.fc0 = nn.Linear(self.image_size, 100)
        self.bn0 = nn.BatchNorm1d(100)  
        self.fc1 = nn.Linear(100, 50)
        self.bn1 = nn.BatchNorm1d(50)   
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu((self.fc0(x)))
        x = self.bn0(x)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def optimizer(self, params, learning_rate):
        optimizer = optim.Adam(params, learning_rate)
        return optimizer


# *****************************************************************************************
# Class Name: Model_E
# Class Details:   _ 5 hidden layers [128,64,10,10,10]
#                  _ activation: RELU
#               
#                  _ output with softmax
# *****************************************************************************************
class Model_E(nn.Module):

    def __init__(self):
        super(Model_E, self).__init__()
        self.image_size = 784  # 28*28
        # layer = (nb of input neurons, number of output neurons)
        self.fc0 = nn.Linear(self.image_size, 128)  # layer input
        self.fc1 = nn.Linear(128, 64)               # layer 1
        self.fc2 = nn.Linear(64, 10)                # layer 2
        self.fc3 = nn.Linear(10, 10)                # layer 3
        self.fc4 = nn.Linear(10, 10)                # layer 4
        self.fc5 = nn.Linear(10, 10)                # layer 5

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))                    # layer 0
        x = F.relu(self.fc1(x))                    # layer 1
        x = F.relu(self.fc2(x))                    # layer 2
        x = F.relu(self.fc3(x))                    # layer 3
        x = F.relu(self.fc4(x))                    # layer 4
        x = F.relu(self.fc5(x))                    # layer 5
        return F.log_softmax(x, dim=1)

    def optimizer(self, params, learning_rate):
        optimizer = optim.SGD(params, learning_rate)
        return optimizer

# *****************************************************************************************
# Class Name: Model_F
# Class Details:   _ 5 hidden layers [128,64,10,10,10]
#                  _ activation: RELU
#                  _ optimizer: ?
#                  _ output with softmax
# *****************************************************************************************
class Model_F(nn.Module):

    def __init__(self):
        super(Model_F, self).__init__()
        self.image_size = 784  # 28*28
        # layer = (nb of input neurons, number of output neurons)
        self.fc0 = nn.Linear(self.image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 10)  # layer 5

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = torch.sigmoid(self.fc0(x))  # layer 0
        x = torch.sigmoid(self.fc1(x))  # layer 1
        x = torch.sigmoid(self.fc2(x))  # layer 2
        x = torch.sigmoid(self.fc3(x))  # layer 3
        x = torch.sigmoid(self.fc4(x))  # layer 4
        x = torch.sigmoid(self.fc5(x))  # layer 5
        return F.log_softmax(x, dim=1)

    def optimizer(self, params, learning_rate):
        optimizer = optim.SGD(params, learning_rate)
        return optimizer


class MY_Model(nn.Module):

    def __init__(self):
        super(MY_Model, self).__init__()
        self.image_size = 784  # 28*28
        # layer = (nb of input neurons, number of output neurons)
        self.fc0 = nn.Linear(self.image_size, 256)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dp = nn.Dropout(0.25)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))  # layer 0
        x = self.dp(x)
        x = F.relu(self.fc1(x))  # layer 1
        x = self.dp(x)
        x = F.relu(self.fc2(x))  # layer 2
        x = self.dp(x)
        x = F.relu(self.fc3(x))  # layer 3
        x = self.dp(x)

        return F.log_softmax(x, dim=1)


def predict_fortest(model,test_loader):
    test_y=[]
    model.eval() 
    with torch.no_grad():
        for data in test_loader:
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_y.append(pred)
    return test_y

# write labels in a file test_y
def writeAnswers(test_labels, filename):
    f = open(filename, "w+")
    size = len(test_labels)
    for i in range(size-1):
        f.write("%d\n" % test_labels[i])

    f.write("%d" % test_labels[size-1])



#*****************************************************************************************
# Function Name: main
# Function Operation: main function
#*****************************************************************************************
def main():

    learning_rate = 0.0008

    # step 1 : load data from files received as arg
    train_x_path, train_y_path, test_x_path, output_log_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]  # get info from args of execution

    train_images = np.loadtxt(train_x_path)
    train_images = train_images.astype(np.float32)
    train_labels = np.genfromtxt(train_y_path, delimiter='\n')
    train_labels = train_labels.astype(int)
    train_size = len(train_labels)

    #normalize
    normalize_train = train_images / 255

    #shuffle train
    shuffler = np.random.permutation(train_size)
    train_x_shuf = normalize_train[shuffler]
    train_y_shuf = train_labels[shuffler]

    ###############convert train to tensor
    tensor_train_images = torch.tensor(train_x_shuf)
    device = torch.device("cpu")
    tensor_train_labels = torch.tensor(train_y_shuf, dtype=torch.long, device=device)
    tensor_train_labels = tensor_train_labels.type(torch.LongTensor)
    train_dataset = TensorDataset(tensor_train_images, tensor_train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64)

#test
    test_images = np.loadtxt(test_x_path)
    test_images = test_images.astype(np.float32)

    # normalize
    normalize_test = test_images / 255
    
    #convert to tensor
    tensor_test_images = torch.tensor(normalize_test)
    test_loader = DataLoader(tensor_test_images)

    #creation of model
    modelA = MY_Model()

    optimizer = optim.Adam(modelA.parameters(), lr=learning_rate)  # can be other optimizer: SGD, RMSprop, momentum, ...

#training
    for epoch in range(1, 19+1):
        #print("\nepoch ", epoch)
        modelA = train(modelA, train_loader, optimizer)

    predictions = predict_fortest(modelA, test_loader)
    writeAnswers(predictions, output_log_name)

#*****************************************************************************************
# Function Name: train
# Function Operation: train the algo to find the params
#*****************************************************************************************
def train(model, train_loader, optimizer):
    size = len(train_loader.dataset)
    correct = 0
    model.train()
    train_loss = 0
    c=0
    for batch_idx, (data, labels) in enumerate(train_loader):
        c=c+1
        optimizer.zero_grad()
        # forward+backward+optimize
        output = model(data)
        loss = F.nll_loss(output, labels)
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()  # update all w and b of each layers based on backprop
    return model


#*****************************************************************************************
# Function Name: test
# Function Operation: test the algorithm for validation and original test dataset
#*****************************************************************************************
def test(model, test_loader):
    model.eval()  # training
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).cpu().sum()

if __name__ == "__main__":
        main()