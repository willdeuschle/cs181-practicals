from bs4 import BeautifulSoup
import os
import pickle
import pandas as pd
import re

# want to pickle our corpus
def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
tagDict = dict()
for tag in soup.findAll():
    tagDict.setdefault(tag.name, 0)
print(tagDict)

# so do that above for all the words in the corpus
train_arr = os.listdir('../train/')

corpusDict = dict()
for idx, fp in enumerate(train_arr):
    if idx % 50 == 0:
        print(f'iteration {idx + 1}')
    with open(f'../train/{fp}') as f:
        soup = BeautifulSoup(f, 'xml')
        for tag in soup.findAll():
            corpusDict.setdefault(tag.name, 0)
print(corpusDict)

save_obj(corpusDict, 'corpusDict')

malwareCategories = {
    'Agent': 0,
    'AutoRun': 1,
    'FraudLoad': 2,
    'FraudPack': 3,
    'Hupigon': 4,
    'Krap': 5,
    'Lipler': 6,
    'Magania': 7,
    'None': 8,
    'Poison': 9,
    'Swizzor': 10,
    'Tdss': 11,
    'VB': 12,
    'Virut': 13,
    'Zbot': 14,
}

trainYs = {'id': [], 'cat': []}
trainYs = pd.DataFrame(data=trainYs)
trainYs

ids = list()
malware_categories = list()
for idx, fp in enumerate(train_arr):
    if idx % 50 == 0:
        print(f'iteration {idx + 1}')
    id_and_malware = re.match('(?P<id>\w+)\.(?P<malware>\w+)\.xml', fp)
    ids.append(id_and_malware.group('id'))
    malware_categories.append(malwareCategories[id_and_malware.group('malware')])
trainYs = {'id': ids, 'malware_category': malware_categories}
trainYs = pd.DataFrame(data=trainYs)
trainYs

save_obj(trainYs, 'trainYs')

# want an array for each feature (system call)
trainXDict = {k: [] for k in corpusDict.keys()}
# want an id for each document
trainXDict['id'] =  []

for idx, fp in enumerate(train_arr):
    if idx % 50 == 0:
        print(f'iteration {idx + 1}')
    
    id_and_malware = re.match('(?P<id>\w+)\.(?P<malware>\w+)\.xml', fp)
    # add the id to our data
    trainXDict['id'].append(id_and_malware.group('id'))
    
    # make a copy of all the features with a 0 count
    inputCorpusDict = corpusDict.copy()
    with open(f'../train/{fp}') as f:
        soup = BeautifulSoup(f, 'xml')
        for tag in soup.findAll():
            inputCorpusDict[tag.name] += 1
    # add each of these counts to our data
    for (k, v) in inputCorpusDict.items():
        trainXDict[k].append(v)

trainXs = {k: v for (k,v) in trainXDict.items()}
trainXs = pd.DataFrame(data=trainXs)
trainXs

save_obj(trainXs, 'trainXs')

# get our testing inputs
test_arr = os.listdir('../test/')
# want an array for each feature (system call)
testXDict = {k: [] for k in corpusDict.keys()}
# want an id for each document
testXDict['id'] =  []

for idx, fp in enumerate(test_arr):
    if idx % 50 == 0:
        print(f'iteration {idx + 1}')
    
    id_and_malware = re.match('(?P<id>\w+)\.(?P<malware>\w+)\.xml', fp)
    # add the id to our data
    testXDict['id'].append(id_and_malware.group('id'))
    
    # make a copy of all the features with a 0 count
    inputCorpusDict = corpusDict.copy()
    with open(f'../test/{fp}') as f:
        soup = BeautifulSoup(f, 'xml')
        for tag in soup.findAll():
            if tag.name in inputCorpusDict:
                inputCorpusDict[tag.name] += 1
    # add each of these counts to our data
    for (k, v) in inputCorpusDict.items():
        testXDict[k].append(v)

testXs = {k: v for (k,v) in testXDict.items()}
testXs = pd.DataFrame(data=testXs)
testXs

import numpy as np
from sklearn.model_selection import KFold

# accuracy classification
def compute_error(pred, actual):
    num_pts = float(pred.shape[0])
    return np.sum((pred - actual)==0)/num_pts
    #return np.sqrt(np.sum(np.square(np.subtract(pred, actual))) / num_pts)

# estimate model performance using validation sets
def select_best_model(X, Y, num_validation_sets, model_fn):
    # error accumulator
    error = 0
    # set up the kfold operation
    kf = KFold(n_splits=num_validation_sets)
    kf.get_n_splits(X)
    for train_index, validation_index in kf.split(X):
        #print("next validation set")
        # get train and validate data
        X_train, X_validation = X.loc[train_index], X.loc[validation_index]
        Y_train, Y_validation = Y.loc[train_index], Y.loc[validation_index]
        # train, should return a function to make predictions
        #print("training")
        trained_fn = model_fn(X_train, Y_train)
        # get error
        pred = trained_fn(X_validation)
        #print("got a prediction!")
        error += compute_error(pred, Y_validation)
    return error / float(num_validation_sets)

#error for regression
import numpy as np
from sklearn.model_selection import KFold

# to compute classification error
def compute_error(pred, actual):
    num_pts = float(pred.shape[0])
    
    return np.sqrt(np.sum(np.square(np.subtract(pred, actual))) / num_pts)

# estimate model performance using validation sets
def select_best_model(X, Y, num_validation_sets, model_fn):
    # error accumulator
    error = 0
    # set up the kfold operation
    kf = KFold(n_splits=num_validation_sets)
    kf.get_n_splits(X)
    for train_index, validation_index in kf.split(X):
        #print("next validation set")
        # get train and validate data
        X_train, X_validation = X.loc[train_index], X.loc[validation_index]
        Y_train, Y_validation = Y.loc[train_index], Y.loc[validation_index]
        # train, should return a function to make predictions
        #print("training")
        trained_fn = model_fn(X_train, Y_train)
        # get error
        pred = trained_fn(X_validation)
        #print("got a prediction!")
        error += compute_error(pred, Y_validation)
    return error / float(num_validation_sets)

from sklearn.ensemble import RandomForestClassifier

def train_and_test_RF_model(n_estimators):
    def helper(X_train, Y_train):
        RF = RandomForestClassifier(n_estimators=n_estimators)
        RF.fit(X_train, Y_train)
        # return the predictor we've trained
        return RF.predict
    return helper

# random forest, ten-fold validation, initially 5 estimators
# simple tag counts + run time, 
print("with 5 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(5)))
print("with 10 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(10)))
print("with 15 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(15)))
print("with 20 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(20)))
print("with 25 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(25)))
print("with 30 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(30)))
print("with 35 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(35)))
print("with 100 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(100)))
print("with 200 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(200)))

# random forest, ten-fold validation, initially 5 estimators
# counts per tag,  run time
print("with 5 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(5)))
print("with 10 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(10)))
print("with 15 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(15)))
print("with 20 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(20)))
print("with 25 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(25)))
print("with 30 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(30)))
print("with 35 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(35)))
print("with 100 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(100)))
print("with 200 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(200)))

# random forest, ten-fold validation, initially 5 estimators
# counts per tag, tags per section,  run time (trainXs_new)
print("with 5 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(5)))
print("with 10 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(10)))
print("with 15 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(15)))
print("with 20 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(20)))
print("with 25 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(25)))
print("with 30 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(30)))
print("with 35 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(35)))
print("with 100 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(100)))
print("with 200 estimators", select_best_model(XsNoId, YsNoId['malware_category'], 20, train_and_test_RF_model(200)))

# doing some old train stuff with estimators
trainXsJustCount = load_obj('trainXs')
XsNoIdCounts = trainXsJustCount.drop(['id'], axis=1)
# random forest, ten-fold validation, initially 5 estimators
# counts per tag only
print("with 5 estimators", select_best_model(XsNoIdCounts, YsNoId['malware_category'], 20, train_and_test_RF_model(5)))
print("with 10 estimators", select_best_model(XsNoIdCounts, YsNoId['malware_category'], 20, train_and_test_RF_model(10)))
print("with 15 estimators", select_best_model(XsNoIdCounts, YsNoId['malware_category'], 20, train_and_test_RF_model(15)))
print("with 20 estimators", select_best_model(XsNoIdCounts, YsNoId['malware_category'], 20, train_and_test_RF_model(20)))
print("with 25 estimators", select_best_model(XsNoIdCounts, YsNoId['malware_category'], 20, train_and_test_RF_model(25)))
print("with 30 estimators", select_best_model(XsNoIdCounts, YsNoId['malware_category'], 20, train_and_test_RF_model(30)))
print("with 35 estimators", select_best_model(XsNoIdCounts, YsNoId['malware_category'], 20, train_and_test_RF_model(35)))
print("with 100 estimators", select_best_model(XsNoIdCounts, YsNoId['malware_category'], 20, train_and_test_RF_model(100)))
print("with 200 estimators", select_best_model(XsNoIdCounts, YsNoId['malware_category'], 20, train_and_test_RF_model(200)))

import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

HIDDEN_LAYER_DIM = 200
NUM_OUTPUT_CATS = 15

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(XsNoId.shape[1], HIDDEN_LAYER_DIM)
        self.fc2 = nn.Linear(HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM)
        #self.fc3 = nn.Linear(HIDDEN_LAYER_DIM, HIDDEN_LAYER_DIM)
        self.fc4 = nn.Linear(HIDDEN_LAYER_DIM, NUM_OUTPUT_CATS)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x)
    
net = Net()
print(net)

import numpy as np
import torch.utils.data as data

class MalwareDataset(data.Dataset):
    """Malware dataset."""

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_val = torch.from_numpy(np.array(self.X.loc[[idx]])).float()
        y_val = torch.from_numpy(np.array(self.Y.loc[[idx]])).long()
        return (x_val, y_val)

my_dataset = MalwareDataset(XsNoId, YsNoId['malware_category'])

from torch.utils.data import DataLoader

epochs = 50
log_interval = 10
batch_size = 200

train_loader = torch.utils.data.DataLoader(
    my_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

# run the main training loop
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # note: we have to squeeze the target because we just want a 
        # tensor of length 200 instead of a 200x1 tensor, and using a numpy
        # array for some reason causes it to be a 200x1 tensor
        data, target = Variable(data), torch.squeeze(Variable(target))
        # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
        data = data.view(-1, XsNoId.shape[1])
        optimizer.zero_grad()
        net_out = net(data)
        loss = criterion(net_out, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.data[0]))
            

preds = list()
for index, row in testInputs.iterrows():
    data = Variable(torch.from_numpy(np.array(row)).float())
    net_out = net(data)
    pred = net_out.data.max(0)[1][0]  # get the index of the max log-probability
    if pred is not 8:
        print("not 8", pred)
    preds.append(pred)