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
import numpy.linalg as npl
import copy
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
from PIL import ImageOps
from matplotlib import pyplot as plt
from itertools import chain
import random


# In[2]:


import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(formatter={'float_kind':"{:.2f}".format})


# In[3]:


data_path = Path('../data/') 
anonymized_data = Path('../data/nii/anonymized')
control_data = Path('../data/nii/controls')


# In[4]:


#make sure values are in range [0,1]
def rescale_pet(scan):
    amax = np.amax(scan)
    rescaled_scan = scan / amax if amax > 0.0 else scan
    return rescaled_scan

def rescale_ct_and_norm(scan):
    amin = np.amin(scan)
    add_min = scan + abs(amin)
    amax = np.amax(add_min)
    rescaled_scan = add_min / amax if amax > 0.0 else add_min
    return rescaled_scan



#Make the three grayscale images to three channels
def expand_np_images_to_3_channels_rescaled(scans,index,axis):
    arrays = []
    modalities = ['pet', 'ct', 'norm_pet']
    for scan, modality in zip(scans, modalities):
        if modality == 'pet':
            if axis == 'axi':
                arrays.append(rescale_pet(scan[:,:,index]))
            elif axis == 'sagi':
                arrays.append(rescale_pet(scan[:,index,:]))
            else:
                arrays.append(rescale_pet(scan[index,:,:]))
        else:
            if axis == 'axi':
                arrays.append(rescale_ct_and_norm(scan[:,:,index]))
            elif axis == 'sagi':
                arrays.append(rescale_ct_and_norm(scan[:,index,:]))
            else:
                arrays.append(rescale_ct_and_norm(scan[index,:,:]))

    return np.stack(arrays, axis=2)

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



# Now make control slices

# In[10]:


for pt_path in control_data.glob('*/*'):
    if not (pt_path/'axi').exists():
        os.mkdir(pt_path/'axi')
    
    if not (pt_path/'sagi').exists():
        os.mkdir(pt_path/'sagi')
    
    if not (pt_path/'coro').exists():
        os.mkdir(pt_path/'coro')
    
       
    #remove existing files
    for file in chain(pt_path.glob("axi/*")):
        os.remove(file)

    for file in chain(pt_path.glob("sagi/*")):
        os.remove(file)

    for file in chain(pt_path.glob("coro/*")):
        os.remove(file)
    
    #Make scan list and modality list 
    scans = []
    modalities = ['PET', 'CT', 'norm_PET']
    
    #load scans and append to list
    img_pet = nib.load(pt_path / 'final_cropped_128_pet.nii.gz')
    scan_pet = img_pet.get_fdata()
    scans.append(scan_pet)

    img_ct = nib.load(pt_path / 'final_cropped_128_CT.nii.gz')
    scan_ct = img_ct.get_fdata()
    scans.append(scan_ct)

    img_norm_pet = nib.load(pt_path / 'final_cropped_128_pet_norm.nii.gz')
    scan_norm_pet = img_norm_pet.get_fdata()
    scans.append(scan_norm_pet)
    
    rand_lst = random.sample(range(15,115), 2)

    for i in rand_lst:
        #Make threechannel image
        axi = expand_np_images_to_3_channels(scans, i, 'axi')
        np.save(pt_path / f'axi/three_channels_slice_{i}', axi)

        sagi = expand_np_images_to_3_channels(scans, i, 'sagi')
        np.save(pt_path / f'sagi/three_channels_slice_{i}', sagi)
        
        coro = expand_np_images_to_3_channels(scans, i, 'coro')
        np.save(pt_path / f'coro/three_channels_slice_{i}', coro)

        #Make single channel images
        for scan, modality in zip(scans, modalities): 
        
            #axi
            np.save(pt_path / f'axi/{modality}_slice_{i}', scan[:,:,i])
            
            #sagi
            np.save(pt_path / f'sagi/{modality}_slice_{i}', scan[:,i,:])
            
            #coro
            np.save(pt_path / f'coro/{modality}_slice_{i}', scan[i,:,:])
    
    print('worked for: ', pt_path)
        


# In[ ]:




