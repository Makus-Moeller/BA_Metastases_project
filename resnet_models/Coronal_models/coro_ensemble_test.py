#!/usr/bin/env python
# coding: utf-8

# In[273]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path


# In[274]:


torch.__version__


# In[282]:


data_path = Path("../../data/")
batch_size = 8

#composite transformations
transform = transforms.Compose([transforms.Resize(512)]) #resample to same resolution used in github repo

torch.cuda.empty_cache()


# ## Data preperation

# In[283]:

torch.multiprocessing.set_start_method('spawn')

#Overwrite getitem and len 
#See https://pytorch.org/docs/stable/data.html#map-style-datasets 

class BrainTumorDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.cuda.FloatTensor(targets) #OBS special datatype when runnig on GPU
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        x = torch.cuda.FloatTensor(x).permute(2,0,1)
        y = self.targets[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)


# In[284]:


#Make grayscale image to three channels
def expand_np_image_to_3_channels(grey_image_np):
    grey_image_arr = np.expand_dims(grey_image_np, -1)
    grey_image_arr_3_channel = grey_image_arr.repeat(3, axis=-1)
    return grey_image_arr_3_channel

#Used to meassure accuracy for each epoch
def correct_number_of_classifications(y_true, y_pred, return_y_pred = False):
    y_pred = y_pred >= 0.5
    if return_y_pred:
        return (y_pred==y_true).sum().item(), y_pred    
    else:
        return (y_pred==y_true).sum().item()


# In[325]:

#AXI_ALL

#Make datastructures for pet
Xt_pet = []
yt_pet = []     

for coro_scan_anony in data_path.glob('nii/anonymized/test/*/coro/PET*'): 
  scan = np.load(coro_scan_anony)
  Xt_pet.append(expand_np_image_to_3_channels(scan))
  yt_pet.append(1.0)

for coro_scan_control in data_path.glob('nii/controls/test/*/coro/PET*'): 
  scan = np.load(coro_scan_control)
  Xt_pet.append(expand_np_image_to_3_channels(scan))
  yt_pet.append(0.0)


#datasets
test_dataset = BrainTumorDataset(Xt_pet, yt_pet, transform=transform)
print(test_dataset.__len__())
test_gen_pet = DataLoader(test_dataset, num_workers=0)



#Make datastructures for ct
Xt_ct = []
yt_ct = []     

for coro_scan_anony in data_path.glob('nii/anonymized/test/*/coro/CT_*'): 
  scan = np.load(coro_scan_anony)
  Xt_ct.append(expand_np_image_to_3_channels(scan))
  yt_ct.append(1.0)

for coro_scan_control in data_path.glob('nii/controls/test/*/coro/CT_*'): 
  scan = np.load(coro_scan_control)
  Xt_ct.append(expand_np_image_to_3_channels(scan))
  yt_ct.append(0.0)


#datasets
test_dataset = BrainTumorDataset(Xt_ct, yt_ct, transform=transform)
print(test_dataset.__len__())
test_gen_ct = DataLoader(test_dataset, num_workers=0)



#Make datastructures for norm
Xt_norm = []
yt_norm = []     

for coro_scan_anony in data_path.glob('nii/anonymized/test/*/coro/norm_*'): 
  scan = np.load(coro_scan_anony)
  Xt_norm.append(expand_np_image_to_3_channels(scan))
  yt_norm.append(1.0)

for coro_scan_control in data_path.glob('nii/controls/test/*/coro/norm_*'): 
  scan = np.load(coro_scan_control)
  Xt_norm.append(expand_np_image_to_3_channels(scan))
  yt_norm.append(0.0)


#datasets
test_dataset = BrainTumorDataset(Xt_norm, yt_norm, transform=transform)
print(test_dataset.__len__())
test_gen_norm = DataLoader(test_dataset, num_workers=0)


#Make datastructures for three
Xt_three = []
yt_three = []     

for coro_scan_anony in data_path.glob('nii/anonymized/test/*/coro/three_*'): 
  scan = np.load(coro_scan_anony)
  Xt_three.append(scan)
  yt_three.append(1.0)

for coro_scan_control in data_path.glob('nii/controls/test/*/coro/three_*'): 
  scan = np.load(coro_scan_control)
  Xt_three.append(scan)
  yt_three.append(0.0)

#datasets
test_dataset = BrainTumorDataset(Xt_three, yt_three, transform=transform)
print(test_dataset.__len__())
test_gen_three = DataLoader(test_dataset, num_workers=0)



# ## Define model and fit fully connected layer to our problem

device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print('running on: ', device)

# In[344]:

# instantiate transfer learning model

#PET model
resnet_model_pet = models.resnet50()
n_inputs = resnet_model_pet.fc.in_features

# redefine fc layer / top layer/ head for our classification problem
resnet_model_pet.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())

resnet_model_pet.load_state_dict(torch.load('coro_pet_model.pt'))
resnet_model_pet.to(device)

#CT model
resnet_model_ct = models.resnet50()
resnet_model_ct.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())


resnet_model_ct.load_state_dict(torch.load('coro_ct_model.pt'))
resnet_model_ct.to(device)

#norm model
resnet_model_norm = models.resnet50()
resnet_model_norm.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())

resnet_model_norm.load_state_dict(torch.load('coro_norm_pet_model.pt'))
resnet_model_norm.to(device)

#three channel model
resnet_model_three = models.resnet50()
resnet_model_three.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())

resnet_model_three.load_state_dict(torch.load('coro_three_channel_model.pt'))
resnet_model_three.to(device)

# set model to evaluation mode
resnet_model_pet.eval()
resnet_model_ct.eval()
resnet_model_norm.eval()
resnet_model_three.eval()

# if GPU is available set loss function to use GPU
criterion = nn.BCELoss().to(device)

# perform no gradient updates
with torch.no_grad():
    # some metrics storage for visualization and analysis
    total_test_loss = 0.0
    test_correct = 0
    labels = []
    pred = []
    # perform test set evaluation batch wise
    for b, ((X_pet, y_pet), (X_ct, y_ct), (X_norm, y_norm), (X_three, y_three)) in enumerate(zip(test_gen_pet, test_gen_ct, test_gen_norm, test_gen_three)):
        # set label to use CUDA if available
        X_pet, y_pet = X_pet.to(device), y_pet.to(device)
        X_ct, y_ct = X_ct.to(device), y_ct.to(device)
        X_norm, y_norm = X_norm.to(device), y_norm.to(device)
        X_three, y_three = X_three.to(device), y_three.to(device)

        # append original labels
        labels.append(y_pet.data)

        # perform forward pass each model
        y_pet_pred = resnet_model_pet(X_pet)
        y_ct_pred = resnet_model_ct(X_ct)
        y_norm_pred = resnet_model_norm(X_norm)
        y_three_pred = resnet_model_three(X_three)

        #find average
        tensor_list = [y_pet_pred, y_ct_pred, y_norm_pred, y_three_pred] 
        y_mean = torch.mean(torch.stack(tensor_list, dim=0))
        
        # calculate loss
        loss = criterion(y_mean.reshape(-1,1), y_pet.reshape(-1,1))

        #update total loss
        total_test_loss  += loss.item()

        # if predicted label is correct as true label, calculate the sum for samples
        batch_correct, predictions = correct_number_of_classifications(torch.flatten(y_pet), torch.flatten(y_mean.reshape(-1,1)), return_y_pred = True) 
        
        # increment train correct with correcly predicted labels per batch
        test_correct += batch_correct

        # append predicted label
        pred.append(predictions.data)

print(f'Test Loss: {total_test_loss/(b+1):.4f}')

print(f'Test accuracy: {test_correct*100/(b+1):.2f}%')


# In[346]:


labels = torch.stack(labels)
pred = torch.stack(pred)


# In[347]:


print(f"Clasification Report\n\n{classification_report(pred.view(-1).cpu(), labels.view(-1).cpu())}")

