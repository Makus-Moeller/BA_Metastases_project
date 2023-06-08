#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path
import nibabel as nib
import numpy as np
import os
from itertools import chain


# In[2]:


import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(formatter={'float_kind':"{:.2f}".format})


# In[3]:


data_path = Path('../data/') 
anonymized_data = Path('../data/nii/anonymized')
control_data = Path('../data/nii/controls')

#rescale 2d scans to range [0,1]
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


# Make functions for each plane

# Each function takes a list of scans 

# In[4]:


def axi(scans, pt_path, annotations, voxelsize=0.2, cutoff=0.6):
    positives = []
    negatives = []
    modalities = ['PET', 'CT', 'norm_PET']
    
    for metastasis in annotations.iloc():
        diameter = float(metastasis.diameter_in_cm.replace(',','.'))
        radius = diameter/2
        revised_radius = (diameter/2)*cutoff
        voxel_z = float(metastasis.voxel_z)

        #positive slices 
        lower_positive = int(voxel_z - revised_radius/voxelsize)
        upper_positive = int(voxel_z + revised_radius/voxelsize)

 
        #negative slices
        buffer = 3
        lower_lower_negative = int(voxel_z - (radius/voxelsize) - buffer)
        upper_lower_negative = lower_positive - 1
        lower_upper_negative = upper_positive + 1
        upper_upper_negative = int(voxel_z + (radius/voxelsize) + buffer)

        for i in range(lower_positive, upper_positive + 1):
            positives.append(i)
        
        for i in range(lower_lower_negative, upper_lower_negative + 1):
            negatives.append(i)
        
        for i in range(lower_upper_negative, upper_upper_negative + 1):
            negatives.append(i)

    #remove duplicates
    positives = list(dict.fromkeys(positives))
    negatives = list(dict.fromkeys(negatives))

    #remove slices that are postives
    negatives = [x for x in negatives if x not in positives]
    
    #iterate through slices and save them as png and classify them
    df_pt = pd.DataFrame(columns=['slice','class'])
    
    for i in range(scans[0].shape[2]):
        label = 0 
        if i in positives:
            
            #Make single channel images
            for scan, modality in zip(scans, modalities):  
                np.save(pt_path / f'axi/{modality}_slice_{i}', scan[:,:,i])  
            
            #Make three channel images
            three_channel_array = expand_np_images_to_3_channels(scans, i, 'axi')  
            np.save(pt_path / f'axi/three_channels_slice_{i}', three_channel_array)          
            label = 1 
        df_pt = df_pt.append({'slice': i, 'class': label}, ignore_index=True) 

    return df_pt


# In[5]:


def sagi(scans, pt_path, annotations, voxelsize=0.2, cutoff=0.6):
    positives = []
    negatives = []
    modalities = ['PET', 'CT', 'norm_PET']
    
    for metastasis in annotations.iloc():
        diameter = float(metastasis.diameter_in_cm.replace(',','.'))
        radius = diameter/2
        revised_radius = (diameter/2)*cutoff
        voxel_y = float(metastasis.voxel_y)

        #positive slices 
        lower_positive = int(voxel_y - revised_radius/voxelsize)
        upper_positive = int(voxel_y + revised_radius/voxelsize)

 
        #negative slices
        buffer = 3
        lower_lower_negative = int(voxel_y - (radius/voxelsize) - buffer)
        upper_lower_negative = lower_positive - 1
        lower_upper_negative = upper_positive + 1
        upper_upper_negative = int(voxel_y + (radius/voxelsize) + buffer)

        for i in range(lower_positive, upper_positive + 1):
            positives.append(i)
        
        for i in range(lower_lower_negative, upper_lower_negative + 1):
            negatives.append(i)
        
        for i in range(lower_upper_negative, upper_upper_negative + 1):
            negatives.append(i)

    #remove duplicates
    positives = list(dict.fromkeys(positives))
    negatives = list(dict.fromkeys(negatives))

    #remove slices that are postives
    negatives = [x for x in negatives if x not in positives]
    
    #iterate through slices and save them as png and classify them
    df_pt = pd.DataFrame(columns=['slice','class'])
    
    for i in range(scans[0].shape[1]):
        label = 0
        if i in positives:  
            
            #make single channel images
            for scan, modality in zip(scans, modalities):
                np.save(pt_path / f'sagi/{modality}_slice_{i}', scan[:,i,:]) 
            
            #make three channel image
            three_channel_array = expand_np_images_to_3_channels(scans, i, 'sagi')  
            np.save(pt_path / f'sagi/three_channels_slice_{i}', three_channel_array)      
            
            label = 1  
        df_pt = df_pt.append({'slice': i, 'class': label}, ignore_index=True)  
        

    return df_pt


# In[6]:


def coro(scans, pt_path, annotations, voxelsize=0.2, cutoff=0.6):
    positives = []
    negatives = []
    modalities = ['PET', 'CT', 'norm_PET']

    for metastasis in annotations.iloc():
        diameter = float(metastasis.diameter_in_cm.replace(',','.'))
        radius = diameter/2
        revised_radius = (diameter/2)*cutoff
        voxel_x = float(metastasis.voxel_x)

        #positive slices 
        lower_positive = int(voxel_x - revised_radius/voxelsize)
        upper_positive = int(voxel_x + revised_radius/voxelsize)

 
        #negative slices
        buffer = 3
        lower_lower_negative = int(voxel_x - (radius/voxelsize) - buffer)
        upper_lower_negative = lower_positive - 1
        lower_upper_negative = upper_positive + 1
        upper_upper_negative = int(voxel_x + (radius/voxelsize) + buffer)

        for i in range(lower_positive, upper_positive + 1):
            positives.append(i)
        
        for i in range(lower_lower_negative, upper_lower_negative + 1):
            negatives.append(i)
        
        for i in range(lower_upper_negative, upper_upper_negative + 1):
            negatives.append(i)

    #remove duplicates
    positives = list(dict.fromkeys(positives))
    negatives = list(dict.fromkeys(negatives))

    #remove slices that are postives
    negatives = [x for x in negatives if x not in positives]
    
    #iterate through slices and save them as png and classify them
    df_pt = pd.DataFrame(columns=['slice','class'])
    for i in range(scans[0].shape[0]):
        label = 0 
        if i in positives:
            
            #make single channel images
            for scan, modality in zip(scans, modalities):  
                np.save(pt_path / f'coro/{modality}_slice_{i}', scan[i,:,:])   

            #make three channel image
            three_channel_array = expand_np_images_to_3_channels(scans, i, 'coro')  
            np.save(pt_path / f'coro/three_channels_slice_{i}', three_channel_array)      
                  
            label = 1  
        df_pt = df_pt.append({'slice': i, 'class': label}, ignore_index=True)  

    return df_pt


# Apply functions on data

# In[7]:


rtss_df = pd.read_pickle(data_path/'cropped_reference_markings_revised.pkl')


# ## Make 2d slices

# In[8]:


for pt_path in anonymized_data.glob('*/*'):
    print(pt_path)

    if not (pt_path/'axi').exists():
        os.mkdir(pt_path/'axi')
    
    if not (pt_path/'sagi').exists():
        os.mkdir(pt_path/'sagi')
    
    if not (pt_path/'coro').exists():
        os.mkdir(pt_path/'coro')
    
       
    #remove existing files
    for file in chain(pt_path.glob("axi/*")):
        os.remove(file)

    #remove existing files
    for file in chain(pt_path.glob("sagi/*")):
        os.remove(file)

    #remove existing files
    for file in chain(pt_path.glob("coro/*")):
        os.remove(file)


    #Get pt id
    start_index = len(str(pt_path))-8 # 37 length of pet name + pt name
    end_index = start_index + 8  #8 = number of chars in pt name   
    pt_id = str(pt_path)[start_index:end_index]

    scans = []
    
    #load scans
    img_pet = nib.load(pt_path / 'final_cropped_128_pet.nii.gz')
    scan_pet = img_pet.get_fdata()
    scans.append(scan_pet)

    img_ct = nib.load(pt_path / 'final_cropped_128_CT.nii.gz')
    scan_ct = img_ct.get_fdata()
    scans.append(scan_ct)

    img_norm_pet = nib.load(pt_path / 'final_cropped_128_pet_norm.nii.gz')
    scan_norm_pet = img_norm_pet.get_fdata()
    scans.append(scan_norm_pet)
    
    
    #make 2d slices, only choose category 2 metastasis for now 
    axi(scans, pt_path, rtss_df[(rtss_df.ID.isin([pt_id]))  & (rtss_df.classificaton.isin(['0','1','2']))])
    sagi(scans, pt_path, rtss_df[(rtss_df.ID.isin([pt_id]))  & (rtss_df.classificaton.isin(['0','1','2']))])
    coro(scans, pt_path, rtss_df[(rtss_df.ID.isin([pt_id]))  & (rtss_df.classificaton.isin(['0','1','2']))])
    
    print('worked for: ', pt_path)

