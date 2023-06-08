#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from torchvision.utils import make_grid
import os, json
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from PIL import Image

# %%
torch.__version__

# %%
data_path = Path("../../data/synthetic/high_noise")

# %%
#Overwrite getitem and len 
#See https://pytorch.org/docs/stable/data.html#map-style-datasets 

class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.FloatTensor(targets) #Change when running on GPU
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        
        if self.transform:
            x = Image.fromarray(self.data[index], mode='RGB') 
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)


# %%
# we simple load in the data file into dictionary
def get_board_dicts(imgdir):
    json_file = imgdir / "dataset.json"
    
    with open(json_file) as f:
        dataset_dicts = json.load(f)

    for i in dataset_dicts:
        filename = i["file_name"] 
        i["file_name"] = imgdir / filename 
        i["width"] = int(i["width"])
        i["height"] = int(i["height"])

        for j in i["annotations"]:
            j["bbox"] = [float(num) for num in j["bbox"]]
            j["bbox_mode"] = int(j["bbox_mode"]) # BoxMode.XYWH_ABS
            j["category_id"] = int(j["category_id"])

    return dataset_dicts

# %%
dicts_train = get_board_dicts(data_path / 'train')
dicts_test = get_board_dicts(data_path / 'test')

# %%
torch.cuda.empty_cache()

# %%
#Make datastructures
Xtrain = []
ytrain = []

Xtest = []
ytest = []

features = None
labels = None
label = []
     

# %%
for dictionary in dicts_train: 
  Xtrain.append(np.load(dictionary["file_name"]))
  #empty list equals false in python
  label = 1.0 if dictionary["annotations"] else 0.0 
  ytrain.append(label)

print(len(Xtrain))
print(len(ytrain))



for dictionary in dicts_test:
  Xtest.append(np.load(dictionary["file_name"]))
  #empty list equals false in python
  label = 1.0 if dictionary["annotations"] else 0.0
  ytest.append(label)

print(len(Xtest))
print(len(ytest))
# %%


# %%
#composite transformations
transform = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])

#datasets
train_dataset = MyDataset(Xtrain, ytrain, transform=transform)
validation_dataset = MyDataset(Xtest, ytest, transform=transform)
#test_dataset = MyDataset(X_test, y_test, transform=transform)


# %%
train_gen = DataLoader(train_dataset, num_workers=2, batch_size=8)
valid_gen = DataLoader(validation_dataset, num_workers=2, batch_size=8)
#test_gen = DataLoader(test_dataset, num_workers=2)

# %%
device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print(device)

# %%
# instantiate transfer learning model
resnet_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# set all paramters as trainable
for param in resnet_model.parameters():
    param.requires_grad = True

# get input of fc layer
n_inputs = resnet_model.fc.in_features

# redefine fc layer / top layer/ head for our classification problem
resnet_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())

# set all paramters of the model as trainable
for name, child in resnet_model.named_children():
  for name2, params in child.named_parameters():
    params.requires_grad = True

# set model to run on GPU or CPU absed on availibility
resnet_model.to(device)

# print the trasnfer learning NN model's architecture
resnet_model
     

# %%
# loss function
# if GPU is available set loss function to use GPU
criterion = nn.BCELoss().to(device)

# optimizer
optimizer = torch.optim.SGD(resnet_model.parameters(), momentum=0.9, lr=3e-4)

# number of training iterations
epochs = 30

# empty lists to store losses and accuracies
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# %%
#Used to meassure accuracy for each epoch
def correct_number_of_classifications(y_true, y_pred, return_y_pred = False):
    y_pred = y_pred >= 0.5
    if return_y_pred:
        return (y_pred==y_true).sum().item(), y_pred    
    else:
        return (y_pred==y_true).sum().item()

# %%
# set training start time
start_time = time.time()

# empty batch variables
b = None
train_b = None
test_b = None
batch_size = 8


# start training
for i in range(epochs):
    # empty training correct and test correct counter as 0 during every iteration
    trn_corr = 0
    tst_corr = 0
    
    #set loss vals
    total_loss_train = 0.0
    total_loss_val = 0.0

    
    # set epoch's starting time
    e_start = time.time()
    
    # train in batches
    for b, (X, y) in enumerate(train_gen):
        # set label as cuda if device is cuda
        X, y = X.to(device), y.to(device)

        # forward pass image sample         
        y_pred = resnet_model(X)
        
        #calculate loss and update total loss
        loss = criterion(y_pred, y.reshape(-1,1))
        total_loss_train += float(loss.item())

        # back propagate with loss
        loss.backward()

        #Find my label which is 1 if y_pred > 0.5
        predicted = y_pred.data.clone()
        
        # if predicted label is correct as true label, calculate the sum for samples
        batch_correct = correct_number_of_classifications(torch.flatten(y), torch.flatten(predicted))
       
        # increment train correct with correcly predicted labels per batch
        trn_corr += batch_correct    
        
        # perform optimizer step for this batch
        optimizer.step()

        # set optimizer gradients to zero
        optimizer.zero_grad()
        
    # set epoch's end time
    e_end = time.time()
    
    # print training metrics
    print(f'Epoch {(i+1)} Batch {(b+1)}\nAccuracy: {trn_corr*100/((b+1)*batch_size):2.2f} %  total  train loss: {total_loss_train/(b+1):2.4f}  Duration: {((e_end-e_start)/60):2.2f} minutes')

    # some metrics storage for visualization
    train_losses.append(total_loss_train/(b+1))
    train_correct.append(trn_corr*100/((b+1)*batch_size))

    #clear X and y
    X, y, b = None, None, None

    # validate using validation generator
    # do not perform any gradient updates while validation
    with torch.no_grad():
        for b, (X, y) in enumerate(valid_gen):
            # set label as cuda if device is cuda
            X, y = X.to(device), y.to(device)
            
            # forward pass image
            y_val = resnet_model(X)

            # get loss of validation set
            loss = criterion(y_val, y.reshape(-1,1))

            #update total loss
            total_loss_val  += loss.item()
            
            # if predicted label is correct as true label, calculate the sum for samples
            batch_correct = correct_number_of_classifications(torch.flatten(y), torch.flatten(y_val))
            
            # increment train correct with correcly predicted labels per batch
            tst_corr += batch_correct

            
    print(f'Validation Accuracy {tst_corr*100/((b+1)*batch_size):2.2f} Validation Loss: {total_loss_val/(b+1):2.4f}\n')
    
    # some metrics storage for visualization
    test_losses.append(total_loss_val/(b+1))
    test_correct.append(tst_corr*100/((b+1)*batch_size))

# set models end time
end_time = time.time() - start_time    

# print models summary
print("\nTraining Duration {:.2f} minutes".format(end_time/60))
#print("GPU memory used : {} kb".format(torch.cuda.memory_allocated()))
#print("GPU memory cached : {}  kb".format(torch.cuda.memory_cached()))
     

# %%
torch.cuda.empty_cache()

# %%
plt.figure('loss')
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.title('Loss on high noise')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.savefig(f'loss_synthetic_data_high_noise.png')



# In[339]:

plt.figure('accuracy')
plt.plot(train_correct, label='Training accuracy')
plt.plot(test_correct, label='Validation accuracy')
plt.title('Accuracy on high noise')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.savefig(f'accuracy_synthetic_high_low_noise.png')

     

# %%



