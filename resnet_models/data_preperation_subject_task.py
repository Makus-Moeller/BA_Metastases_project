#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path
#from rhscripts import dcm
from typing import Optional, Tuple, Callable, Dict, Union
import torchio as tio
import nibabel as nib
import numpy as np
import os
from matplotlib import pyplot as plt
from itertools import chain
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import numpy as np



# In[2]:


import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(formatter={'float_kind':"{:.2f}".format})


# In[ ]:


data_path = Path('../../data/') 
anonymized_data = Path('../../data/nii/anonymized')
control_data = Path('../../data/nii/controls')
image_dimensions = 128
batch_size = 8

#composite transformations
transform = transforms.Compose([transforms.Resize(512)]) #resample to same resolution used in github repo

torch.cuda.empty_cache()
torch.multiprocessing.set_start_method('spawn')


# In[ ]:


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

#Make the three grayscale images to three channels
def expand_np_images_to_3_channels(scans,index,axis):
    arrays = []
    for scan in scans:
        if axis == 'axi':
            arrays.append(scan[:,:,index])
        elif axis == 'sagi':
            arrays.append(scan[:,index,:])
        else:
            arrays.append(scan[index,:,:])
    return np.stack(arrays, axis=2)

#Used to meassure accuracy for each epoch
def correct_number_of_classifications(y_true, y_pred, return_y_pred = False):
    y_pred = y_pred >= 0.5
    if return_y_pred:
        return (y_pred==y_true).sum().item(), y_pred
    else:
        return (y_pred==y_true).sum().item()



# In[ ]:


def data_gen(scan, axis, target):
    #Make datastructures for pet
    X = []
    y = []

    for index in range(image_dimensions):
        if axis == 'axi':
            X.append(expand_np_image_to_3_channels(scan[:,:,index]))
        elif axis == 'sagi':
            X.append(expand_np_image_to_3_channels(scan[:,index,:]))
        else:
            X.append(expand_np_image_to_3_channels(scan[index,:,:]))

        y.append(target)
    #datasets
    test_dataset = BrainTumorDataset(X, y, transform=transform)
    
    return DataLoader(test_dataset, num_workers=0)


def data_gen_three_channels(scans, axis, target):
    #Make datastructures for pet
    X = []
    y = []

    for index in range(image_dimensions):
        X.append(expand_np_images_to_3_channels(scans, index, axis))
        y.append(target)
    
    #datasets
    test_dataset = BrainTumorDataset(X, y, transform=transform)
    
    return DataLoader(test_dataset, num_workers=0)


# In[ ]:


device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print('running on: ', device)

# In[344]:

# instantiate transfer learning models

#CORO

#PET model
coro_pet_model = models.resnet50()
n_inputs = coro_pet_model.fc.in_features
coro_pet_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())

coro_pet_model.load_state_dict(torch.load('coro_pet_model.pt'))
coro_pet_model.to(device) 

#CT model
coro_ct_model = models.resnet50()
coro_ct_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())


coro_ct_model.load_state_dict(torch.load('coro_ct_model.pt'))
coro_ct_model.to(device)

#norm model
coro_norm_model = models.resnet50()
coro_norm_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())
coro_norm_model.load_state_dict(torch.load('coro_norm_pet_model.pt'))
coro_norm_model.to(device)

#three channel model
coro_three_model = models.resnet50()
coro_three_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())

coro_three_model.load_state_dict(torch.load('coro_three_channel_model.pt'))
coro_three_model.to(device)


# set model to evaluation mode
coro_pet_model.eval()
coro_ct_model.eval()
coro_norm_model.eval()
coro_three_model.eval()

#AXI

#PET model
axi_pet_model = models.resnet50()
n_inputs = axi_pet_model.fc.in_features
axi_pet_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())

axi_pet_model.load_state_dict(torch.load('axi_pet_model.pt'))
axi_pet_model.to(device) 

#CT model
axi_ct_model = models.resnet50()
axi_ct_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())


axi_ct_model.load_state_dict(torch.load('axi_ct_model.pt'))
axi_ct_model.to(device)

#norm model
axi_norm_model = models.resnet50()
axi_norm_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())
axi_norm_model.load_state_dict(torch.load('axi_norm_pet_model.pt'))
axi_norm_model.to(device)

#three channel model
axi_three_model = models.resnet50()
axi_three_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())

axi_three_model.load_state_dict(torch.load('axi_three_channel_model.pt'))
axi_three_model.to(device)


# set model to evaluation mode
axi_pet_model.eval()
axi_ct_model.eval()
axi_norm_model.eval()
axi_three_model.eval()

#SAGI

#PET model
sagi_pet_model = models.resnet50()
n_inputs = sagi_pet_model.fc.in_features
sagi_pet_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())

sagi_pet_model.load_state_dict(torch.load('sagi_pet_model.pt'))
sagi_pet_model.to(device) 

#CT model
sagi_ct_model = models.resnet50()
sagi_ct_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())


sagi_ct_model.load_state_dict(torch.load('sagi_ct_model.pt'))
sagi_ct_model.to(device)

#norm model
sagi_norm_model = models.resnet50()
sagi_norm_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())
sagi_norm_model.load_state_dict(torch.load('sagi_norm_pet_model.pt'))
sagi_norm_model.to(device)

#three channel model
sagi_three_model = models.resnet50()
sagi_three_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())

sagi_three_model.load_state_dict(torch.load('sagi_three_channel_model.pt'))
sagi_three_model.to(device)


# set model to evaluation mode
sagi_pet_model.eval()
sagi_ct_model.eval()
sagi_norm_model.eval()
sagi_three_model.eval()


#loss function
criterion = nn.BCELoss().to(device)


# In[ ]:

#make df for subject classification
df_pt = pd.DataFrame(columns=['pt', 'axi_avg_prob', 'axi_amount_sick', 'axi_max_prob', 'axi_top10_prob', 
                              'coro_avg_prob', 'coro_amount_sick', 'coro_max_prob', 'coro_top10_prob',
                              'sagi_avg_prob', 'sagi_amount_sick', 'sagi_max_prob', 'sagi_top10_prob', 'target'])

for pt_path in anonymized_data.glob('*/*'):
    target = 1.0 #we are in the anonymized folder and hence class 1
    
    #Get pt id
    #start_index = len(str(pt_path))-8 # 37 length of pet name + pt name
    #end_index = start_index + 8  #8 = number of chars in pt name   
    #pt_id = str(pt_path)[start_index:end_index]

    scans = []

    #load scans
    #pet
    img_pet = nib.load(pt_path / 'final_cropped_128_pet.nii.gz')
    scan_pet = img_pet.get_fdata()
    scans.append(scan_pet)
    coro_pet_gen = data_gen(scan_pet, 'coro', target)
    axi_pet_gen = data_gen(scan_pet, 'axi', target)
    sagi_pet_gen = data_gen(scan_pet, 'sagi', target)

    #ct
    img_ct = nib.load(pt_path / 'final_cropped_128_CT.nii.gz')
    scan_ct = img_ct.get_fdata()
    scans.append(scan_ct)
    coro_ct_gen = data_gen(scan_ct, 'coro', target)
    axi_ct_gen = data_gen(scan_ct, 'axi', target)
    sagi_ct_gen = data_gen(scan_ct, 'sagi', target)

    #pet norm
    img_norm = nib.load(pt_path / 'final_cropped_128_pet_norm.nii.gz')
    scan_norm = img_norm.get_fdata()
    scans.append(scan_norm)
    coro_norm_gen = data_gen(scan_norm, 'coro', target)
    axi_norm_gen = data_gen(scan_norm, 'axi', target)
    sagi_norm_gen = data_gen(scan_norm, 'sagi', target)
    
    #three channel
    coro_three_gen = data_gen_three_channels(scans,'coro', target)
    axi_three_gen = data_gen_three_channels(scans,'axi', target)
    sagi_three_gen = data_gen_three_channels(scans,'sagi', target)

    # perform no gradient updates
    with torch.no_grad():
        # some metrics storage for visualization and analysis
        axi_accumulated_pred = 0.0
        axi_counter = 0
        axi_max_prob = 0.0
        axi_top10_prob = np.zeros(13)
        
        coro_accumulated_pred = 0.0
        coro_counter = 0
        coro_max_prob = 0.0
        coro_top10_prob = np.zeros(13)
        
        sagi_accumulated_pred = 0.0
        sagi_counter = 0
        sagi_max_prob = 0.0
        sagi_top10_prob = np.zeros(13)
        
        # perform test set evaluation batch wise for axi 
        for b, ((X_pet, y_pet), (X_ct, y_ct), (X_norm, y_norm), (X_three, y_three)) in enumerate(zip(axi_pet_gen, axi_ct_gen, axi_norm_gen, axi_three_gen)):
            # set label to use CUDA if available
            X_pet, y_pet = X_pet.to(device), y_pet.to(device)
            X_ct, y_ct = X_ct.to(device), y_ct.to(device)
            X_norm, y_norm = X_norm.to(device), y_norm.to(device)
            X_three, y_three = X_three.to(device), y_three.to(device)


            # perform forward pass each model
            y_pet_pred = axi_pet_model(X_pet)
            y_ct_pred = axi_ct_model(X_ct)
            y_norm_pred = axi_norm_model(X_norm)
            y_three_pred = axi_three_model(X_three)

            #find average
            tensor_list = [y_pet_pred, y_ct_pred, y_norm_pred, y_three_pred]
            y_mean = torch.mean(torch.stack(tensor_list, dim=0))
            
            if y_mean.item() >= 0.5:
               axi_counter += 1
            
            if y_mean.item() > axi_max_prob:
                axi_max_prob = y_mean.item()

            if axi_top10_prob[0] < y_mean.item():
                axi_top10_prob[0]= y_mean.item() 
                axi_top10_prob.sort()

            axi_accumulated_pred += y_mean.item()
        

        # perform test set evaluation batch wise for coro 
        for b, ((X_pet, y_pet), (X_ct, y_ct), (X_norm, y_norm), (X_three, y_three)) in enumerate(zip(coro_pet_gen, coro_ct_gen, coro_norm_gen, coro_three_gen)):
            # set label to use CUDA if available
            X_pet, y_pet = X_pet.to(device), y_pet.to(device)
            X_ct, y_ct = X_ct.to(device), y_ct.to(device)
            X_norm, y_norm = X_norm.to(device), y_norm.to(device)
            X_three, y_three = X_three.to(device), y_three.to(device)


            # perform forward pass each model
            y_pet_pred = coro_pet_model(X_pet)
            y_ct_pred = coro_ct_model(X_ct)
            y_norm_pred = coro_norm_model(X_norm)
            y_three_pred = coro_three_model(X_three)

            #find average
            tensor_list = [y_pet_pred, y_ct_pred, y_norm_pred, y_three_pred]
            y_mean = torch.mean(torch.stack(tensor_list, dim=0))
            
            if y_mean.item() >= 0.5:
               coro_counter += 1

            if y_mean.item() > coro_max_prob:
                coro_max_prob = y_mean.item()

            if coro_top10_prob[0] < y_mean.item():
                coro_top10_prob[0]= y_mean.item() 
                coro_top10_prob.sort()

            coro_accumulated_pred += y_mean.item()
        

        # perform test set evaluation batch wise for sagi 
        for b, ((X_pet, y_pet), (X_ct, y_ct), (X_norm, y_norm), (X_three, y_three)) in enumerate(zip(sagi_pet_gen, sagi_ct_gen, sagi_norm_gen, sagi_three_gen)):
            # set label to use CUDA if available
            X_pet, y_pet = X_pet.to(device), y_pet.to(device)
            X_ct, y_ct = X_ct.to(device), y_ct.to(device)
            X_norm, y_norm = X_norm.to(device), y_norm.to(device)
            X_three, y_three = X_three.to(device), y_three.to(device)


            # perform forward pass each model
            y_pet_pred = sagi_pet_model(X_pet)
            y_ct_pred = sagi_ct_model(X_ct)
            y_norm_pred = sagi_norm_model(X_norm)
            y_three_pred = sagi_three_model(X_three)

            #find average
            tensor_list = [y_pet_pred, y_ct_pred, y_norm_pred, y_three_pred]
            y_mean = torch.mean(torch.stack(tensor_list, dim=0))
            
            if y_mean.item() >= 0.5:
               sagi_counter += 1

            if y_mean.item() > sagi_max_prob:
                sagi_max_prob = y_mean.item()  
            
            if sagi_top10_prob[0] < y_mean.item():
                sagi_top10_prob[0]= y_mean.item() 
                sagi_top10_prob.sort()

            sagi_accumulated_pred += y_mean.item()
        

        df_pt = df_pt.append({'pt': pt_path, 
                                'axi_avg_prob': axi_accumulated_pred/image_dimensions, 
                                'axi_amount_sick': axi_counter, 'axi_max_prob': axi_max_prob,
                                'axi_top10_prob': np.mean(axi_top10_prob),
                                'coro_avg_prob': coro_accumulated_pred/image_dimensions, 
                                'coro_amount_sick': coro_counter, 'coro_max_prob': coro_max_prob,
                                'coro_top10_prob': np.mean(coro_top10_prob),
                                'sagi_avg_prob': sagi_accumulated_pred/image_dimensions, 
                                'sagi_amount_sick': sagi_counter, 'sagi_max_prob': sagi_max_prob,
                                'sagi_top10_prob': np.mean(sagi_top10_prob), 
                                'target': target}, ignore_index=True) 
        
        
    print(pt_path) 


print('Now going to controls')

# In[ ]:


for pt_path in control_data.glob('*/*'):
    target = 0.0 #we are in the control folder and hence class 0
    
    #Get pt id
    #start_index = len(str(pt_path))-8 # 37 length of pet name + pt name
    #end_index = start_index + 8  #8 = number of chars in pt name   
    #pt_id = str(pt_path)[start_index:end_index]
    
    scans = []

    #load scans
    #pet
    img_pet = nib.load(pt_path / 'final_cropped_128_pet.nii.gz')
    scan_pet = img_pet.get_fdata()
    scans.append(scan_pet)
    coro_pet_gen = data_gen(scan_pet, 'coro', target)
    axi_pet_gen = data_gen(scan_pet, 'axi', target)
    sagi_pet_gen = data_gen(scan_pet, 'sagi', target)

    #ct
    img_ct = nib.load(pt_path / 'final_cropped_128_CT.nii.gz')
    scan_ct = img_ct.get_fdata()
    scans.append(scan_ct)
    coro_ct_gen = data_gen(scan_ct, 'coro', target)
    axi_ct_gen = data_gen(scan_ct, 'axi', target)
    sagi_ct_gen = data_gen(scan_ct, 'sagi', target)

    #pet norm
    img_norm = nib.load(pt_path / 'final_cropped_128_pet_norm.nii.gz')
    scan_norm = img_norm.get_fdata()
    scans.append(scan_norm)
    coro_norm_gen = data_gen(scan_norm, 'coro', target)
    axi_norm_gen = data_gen(scan_norm, 'axi', target)
    sagi_norm_gen = data_gen(scan_norm, 'sagi', target)
    
    #three channel
    coro_three_gen = data_gen_three_channels(scans,'coro', target)
    axi_three_gen = data_gen_three_channels(scans,'axi', target)
    sagi_three_gen = data_gen_three_channels(scans,'sagi', target)

    # perform no gradient updates
    with torch.no_grad():
        # some metrics storage for visualization and analysis
        axi_accumulated_pred = 0.0
        axi_counter = 0
        axi_max_prob = 0.0
        axi_top10_prob = np.zeros(13)
        
        coro_accumulated_pred = 0.0
        coro_counter = 0
        coro_max_prob = 0.0
        coro_top10_prob = np.zeros(13)
        
        sagi_accumulated_pred = 0.0
        sagi_counter = 0
        sagi_max_prob = 0.0
        sagi_top10_prob = np.zeros(13)
        
        # perform test set evaluation batch wise for axi 
        for b, ((X_pet, y_pet), (X_ct, y_ct), (X_norm, y_norm), (X_three, y_three)) in enumerate(zip(axi_pet_gen, axi_ct_gen, axi_norm_gen, axi_three_gen)):
            # set label to use CUDA if available
            X_pet, y_pet = X_pet.to(device), y_pet.to(device)
            X_ct, y_ct = X_ct.to(device), y_ct.to(device)
            X_norm, y_norm = X_norm.to(device), y_norm.to(device)
            X_three, y_three = X_three.to(device), y_three.to(device)


            # perform forward pass each model
            y_pet_pred = axi_pet_model(X_pet)
            y_ct_pred = axi_ct_model(X_ct)
            y_norm_pred = axi_norm_model(X_norm)
            y_three_pred = axi_three_model(X_three)

            #find average
            tensor_list = [y_pet_pred, y_ct_pred, y_norm_pred, y_three_pred]
            y_mean = torch.mean(torch.stack(tensor_list, dim=0))
            
            if y_mean.item() >= 0.5:
               axi_counter += 1
            
            if y_mean.item() > axi_max_prob:
                axi_max_prob = y_mean.item()

            if axi_top10_prob[0] < y_mean.item():
                axi_top10_prob[0]= y_mean.item() 
                axi_top10_prob.sort()

            axi_accumulated_pred += y_mean.item()
        

        # perform test set evaluation batch wise for coro 
        for b, ((X_pet, y_pet), (X_ct, y_ct), (X_norm, y_norm), (X_three, y_three)) in enumerate(zip(coro_pet_gen, coro_ct_gen, coro_norm_gen, coro_three_gen)):
            # set label to use CUDA if available
            X_pet, y_pet = X_pet.to(device), y_pet.to(device)
            X_ct, y_ct = X_ct.to(device), y_ct.to(device)
            X_norm, y_norm = X_norm.to(device), y_norm.to(device)
            X_three, y_three = X_three.to(device), y_three.to(device)


            # perform forward pass each model
            y_pet_pred = coro_pet_model(X_pet)
            y_ct_pred = coro_ct_model(X_ct)
            y_norm_pred = coro_norm_model(X_norm)
            y_three_pred = coro_three_model(X_three)

            #find average
            tensor_list = [y_pet_pred, y_ct_pred, y_norm_pred, y_three_pred]
            y_mean = torch.mean(torch.stack(tensor_list, dim=0))
            
            if y_mean.item() >= 0.5:
               coro_counter += 1

            if y_mean.item() > coro_max_prob:
                coro_max_prob = y_mean.item()

            if coro_top10_prob[0] < y_mean.item():
                coro_top10_prob[0]= y_mean.item() 
                coro_top10_prob.sort()

            coro_accumulated_pred += y_mean.item()
        

        # perform test set evaluation batch wise for sagi 
        for b, ((X_pet, y_pet), (X_ct, y_ct), (X_norm, y_norm), (X_three, y_three)) in enumerate(zip(sagi_pet_gen, sagi_ct_gen, sagi_norm_gen, sagi_three_gen)):
            # set label to use CUDA if available
            X_pet, y_pet = X_pet.to(device), y_pet.to(device)
            X_ct, y_ct = X_ct.to(device), y_ct.to(device)
            X_norm, y_norm = X_norm.to(device), y_norm.to(device)
            X_three, y_three = X_three.to(device), y_three.to(device)


            # perform forward pass each model
            y_pet_pred = sagi_pet_model(X_pet)
            y_ct_pred = sagi_ct_model(X_ct)
            y_norm_pred = sagi_norm_model(X_norm)
            y_three_pred = sagi_three_model(X_three)

            #find average
            tensor_list = [y_pet_pred, y_ct_pred, y_norm_pred, y_three_pred]
            y_mean = torch.mean(torch.stack(tensor_list, dim=0))
            
            if y_mean.item() >= 0.5:
               sagi_counter += 1

            if y_mean.item() > sagi_max_prob:
                sagi_max_prob = y_mean.item()  
            
            if sagi_top10_prob[0] < y_mean.item():
                sagi_top10_prob[0]= y_mean.item() 
                sagi_top10_prob.sort()

            sagi_accumulated_pred += y_mean.item()
        

        df_pt = df_pt.append({'pt': pt_path, 
                                'axi_avg_prob': axi_accumulated_pred/image_dimensions, 
                                'axi_amount_sick': axi_counter, 'axi_max_prob': axi_max_prob,
                                'axi_top10_prob': np.mean(axi_top10_prob),
                                'coro_avg_prob': coro_accumulated_pred/image_dimensions, 
                                'coro_amount_sick': coro_counter, 'coro_max_prob': coro_max_prob,
                                'coro_top10_prob': np.mean(coro_top10_prob),
                                'sagi_avg_prob': sagi_accumulated_pred/image_dimensions, 
                                'sagi_amount_sick': sagi_counter, 'sagi_max_prob': sagi_max_prob,
                                'sagi_top10_prob': np.mean(sagi_top10_prob), 
                                'target': target}, ignore_index=True)

    print(pt_path)

#Save df as pickle
df_pt.to_pickle(data_path / 'logistic_regression_revised_data.pkl')

