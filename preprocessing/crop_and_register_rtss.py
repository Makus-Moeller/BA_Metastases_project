#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from pathlib import Path
#from rhscripts import dcm
from rhscripts.plotting import _PETRainbowCMAP
from typing import Optional, Tuple, Callable, Dict, Union
import pydicom
from pydicom import dcmread
import torchio as tio
import numpy as np
from matplotlib import pyplot as plt
import os
from nibabel.affines import apply_affine
import numpy.linalg as npl
import copy
from matplotlib.backends.backend_pdf import PdfPages
from torchvision.ops import masks_to_boxes
import nibabel as nib
import math
from PIL import Image
import torch


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


path_data = Path('../data')
anonymized_data = path_data / 'nii/anonymized'

np.set_printoptions(formatter={'float_kind':"{:.2f}".format})


# In[4]:


#Set our brain dimensions
largest_brain_dimensions = [128, 128, 128]


# In[5]:


image_path = path_data / "nii/anonymized/"
control_path =  path_data / "nii/controls/"
output_folders = []
control_folders = []
for pt in image_path.glob("*/"):
    output_folders.append(pt)
for pt in control_path.glob("*/"):
    output_folders.append(pt)
    control_folders.append(pt)


# In[8]:


#This path is for for final script
df_final = pd.read_pickle(path_data /'registered_reference_markings_revised.pkl')


# In[9]:


def control_crop(pt, do_plot=False, pdf=None, plot_counter=0, max_plot=0):
    
    #Cropping the control group
    pt_pet = tio.Subject(
        scan = tio.ScalarImage(pt / '2mmPET_crop_BET.nii.gz'),
        brain_mask = tio.LabelMap(pt / '2mmCT_brain_mask_to_PET.nii.gz'),
    )
    
    pt_ct = tio.Subject(
        scan = tio.ScalarImage(pt/ '2mmCT_BET_rslPET_crop.nii.gz') ,
        brain_mask = tio.LabelMap(pt / '2mmCT_brain_mask_to_PET.nii.gz'),
    )
    
    transform = tio.CropOrPad(
        tuple(largest_brain_dimensions),
        mask_name = 'brain_mask',
    )

    transformed_pet = transform(pt_pet)
    transformed_ct = transform(pt_ct)
    
    transformed_pet.scan.save(pt/'final_cropped_128_pet.nii.gz')
    transformed_ct.scan.save(pt/'final_cropped_128_CT.nii.gz')
    
    if do_plot & (pdf is not None) & (plot_counter < max_plot):
        

        container = transformed_pet.scan
        other = transformed_ct.scan
        
            
        try:            
            # PLOT
            fig, axes = plt.subplots(1, 2, dpi=50, sharex=True, sharey=True, figsize=(10, 5))

            # Load other image as well

            cmap_container = _PETRainbowCMAP
            cmap_other = 'gray'

            titles = [
                'PT: {}, Modality = PET'.format(pt), 
                'Modality = CT'
            ] 

            for ax_ind, (img, cmap, title) in enumerate(zip([container, other], [cmap_container, cmap_other], titles)):
                ax = axes[ax_ind]
                ax.imshow(np.fliplr(np.rot90(img.numpy()[0,:,:,int(largest_brain_dimensions[2]/2)], 3)), cmap=cmap)
                ax.set_title(title)
                ax.axis('off')

            plt.tight_layout()
            pdf.savefig(fig)
            print(f"\n\nworked for pt {pt}")
        except:
            print(f"\n\nerror for pt {pt}") 
    


# In[15]:


def rtss_reg_cropped(row, pet_name, ct_name, do_plot=False, pdf=None):
    
    #make subjects 
    pt_pet = tio.Subject(
        scan = tio.ScalarImage(anonymized_data /'{}/{}'.format(row.ID, pet_name)),
        brain_mask = tio.LabelMap(anonymized_data /'{}/2mmCT_brain_mask_to_PET.nii.gz'.format(row.ID)),
    )
    
    pt_ct = tio.Subject(
        scan = tio.ScalarImage(anonymized_data /'{}/{}'.format(row.ID, ct_name)),
        brain_mask = tio.LabelMap(anonymized_data /'{}/2mmCT_brain_mask_to_PET.nii.gz'.format(row.ID)),
    )
    
    #make cropping
    transform = tio.CropOrPad(
        tuple(largest_brain_dimensions),
        mask_name = 'brain_mask',
    )
    
    transformed_pet = transform(pt_pet)
    transformed_ct = transform(pt_ct)
    
    
    #Save images
    transformed_pet.scan.save(anonymized_data/"{}/final_cropped_128_pet.nii.gz".format(row.ID))
    transformed_ct.scan.save(anonymized_data/"{}/final_cropped_128_CT.nii.gz".format(row.ID))
    
    
    #Calculate new voxel cordinate 
    world = [row.world_x, row.world_y, row.world_z]
    voxel = [row.voxel_x, row.voxel_y, row.voxel_z]
    
  
    
    #Corresponds to transforming from voxel to world to voxel in new space
    vox2cropped_vox = npl.inv(transformed_pet.scan.affine).dot(pt_pet.scan.affine)
    transformed_voxel = apply_affine(vox2cropped_vox, voxel)
    voxel_int = list(map(round,transformed_voxel))
    
    
    #Do the plotting
    if do_plot & (pdf is not None):
        
        if row.linked_to == 'PT':
            container = transformed_pet.scan
            other = transformed_ct.scan
        else:
            container = transformed_ct.scan
            other = transformed_pet.scan
            
        try:            
            # PLOT
            fig, axes = plt.subplots(1, 2, dpi=50, sharex=True, sharey=True, figsize=(10, 5))

            # Load other image as well
            if row.linked_to == "CT":
                cmap_container = 'gray'
                cmap_other = _PETRainbowCMAP
            else:
                cmap_other = 'gray'
                cmap_container = _PETRainbowCMAP

            titles = [
                'PT: {}, Location: {}. linked to: {}'.format(row.ID, voxel_int, row.linked_to), 
                'Classification: {}. Diameter in cm: {}.'.format(row.classificaton, row.diameter_in_cm)
            ] 
        
            for ax_ind, (img, cmap, title) in enumerate(zip([container, other], [cmap_container, cmap_other], titles)):
                ax = axes[ax_ind]
                ax.imshow(np.fliplr(np.rot90(img.numpy()[0,:,:,voxel_int[2]], 3)), cmap=cmap)
                ax.plot(voxel_int[0], voxel_int[1], 'bo', alpha=.3)
                ax.set_title(title)
                ax.axis('off')
                
            plt.tight_layout()
            pdf.savefig(fig)
            print(f"\n\nworked for pt {row.ID}")
        except:
            print(f"\n\nerror for pt {row.ID}") 
    
    return transformed_voxel


# In[16]:


#For final script
pdf = PdfPages(f"{path_data}/rtss_registrations_cropped.pdf")
df_final_crop = pd.DataFrame(columns=['ID','world_x', 'world_y', 'world_z','voxel_x', 'voxel_y', 'voxel_z','classificaton','diameter_in_cm', 'linked_to'])


for pt in df_final.iloc:
    if not (anonymized_data / f"{pt.ID}/2mmCT_BET_rslPET_crop.nii.gz").exists() and (anonymized_data / f"{pt.ID}/2mmPET_crop_BET.nii.gz").exists():
        print(f"File: {anonymized_data}/{pt.ID}  Not Found")
        continue
    voxel = rtss_reg_cropped(pt, '2mmPET_crop_BET.nii.gz', '2mmCT_BET_rslPET_crop.nii.gz', do_plot=False, pdf=pdf)
    df_final_crop = df_final_crop.append(pd.DataFrame({'ID': pt.ID, 'world_x': pt.world_x, 'world_y': pt.world_y, 'world_z': pt.world_z, 'voxel_x': voxel[0], 'voxel_y': voxel[1], 'voxel_z': voxel[2], 'classificaton': pt.classificaton, 'diameter_in_cm': pt.diameter_in_cm, 'linked_to': pt.linked_to}, index=[0]), ignore_index=True)

pdf.close()
print("Done with Anonimized\n")


# In[11]:


df_final_crop.to_pickle(path_data / 'cropped_reference_markings_revised.pkl')


# In[ ]:


#Now Crop control group
#pdf = PdfPages(f"{path_data}/control_cropped.pdf")
#counter = 0
#for pt in control_folders:
#    if not (pt / '2mmCT_BET_rslPET_crop.nii.gz').exists() and (pt / '2mmPET_crop_BET.nii.gz').exists():
#        print(f"Files for {pt} Not found")
#        continue
#    control_crop(pt, do_plot=True, pdf=pdf, plot_counter=counter, max_plot=20)
#    counter = counter+1
#    if (counter % 100) == 0:
#        print(f"Progress: {counter}/{len(control_folders)}\n" )
#pdf.close()   

