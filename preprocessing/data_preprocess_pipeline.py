#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torchio as tio
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from itertools import chain
from nipype.interfaces.ants import Registration, ApplyTransforms
from matplotlib.backends.backend_pdf import PdfPages
from nipype.interfaces.fsl.maths import Threshold, ApplyMask
import re
import os
import pandas as pd
from math import sqrt
import json
from math import floor, ceil


# In[2]:


limit_g = -1
save_pdf = True

path_data = Path('../data/')


# In[3]:


#Delete files we want to generate
for pt in path_data.glob('nii/*/*'):
    for file in ['CT_brain_mask_to_PET', 'PET_crop_BET', '2mmCT_BET_rslPET_crop', '2mmPET_crop_BET', '2mmCT_brain_mask_to_PET']:
        if pt.joinpath(f'{file}.nii.gz').exists():
            os.remove(pt.joinpath(f'{file}.nii.gz'))


# In[4]:


# Block: ...
print("Warping CT mask to PET...")

limit = limit_g

at = ApplyTransforms()
at.inputs.dimension = 3
at.inputs.interpolation = 'Linear'
at.inputs.default_value = 0
at.inputs.invert_transform_flags = False

def warp_CT_mask_2_PET(pt):
    if pt.joinpath('PET_crop.nii.gz').exists() and pt.joinpath("ANAT_smoothed_0-100_BET_mask.nii.gz").exists() and pt.joinpath("Composite.h5").exists():
        at.inputs.input_image = str(pt.joinpath('ANAT_smoothed_0-100_BET_mask.nii.gz'))
        at.inputs.reference_image = str(pt.joinpath('PET_crop.nii.gz'))
        at.inputs.output_image = str(pt.joinpath('CT_brain_mask_to_PET.nii.gz'))
        at.inputs.transforms = str(pt.joinpath('Composite.h5'))
        at.run()
        print('\twarped CT brain mask.')

for pt in path_data.glob('nii/*/*'):
    if limit == 0: break
    limit -= 1
    print(pt)
    warp_CT_mask_2_PET(pt)

#Something is wrong when we are warping the brain mask. Not even the warped CT matches the warped mask. 
#The registration however from CT_BET to PET_crop looks fine.
#E.g look at this example.Insert this command in terminal 
#register bach_files/test_scripts/data/nii/anonymized/Cmet0013/CT_brain_mask_to_PET.nii.gz bach_files/test_scripts/data/nii/anonymized/Cmet0013/CT_BET_rslPET_crop.nii.gz
#register bach_files/test_scripts/data/nii/anonymized/Cmet0014/PET_crop.nii.gz bach_files/test_scripts/data/nii/anonymized/Cmet0014/CT_BET_rslPET_crop.nii.gz


# In[5]:


# Block:...
print("Apply brainmask on PET scan")

limit = limit_g
mask = ApplyMask()

def bet_PET(pt):
    if pt.joinpath('CT_brain_mask_to_PET.nii.gz').exists() and pt.joinpath("PET_crop.nii.gz").exists() and not pt.joinpath("PET_crop_BET.nii.gz").exists():
        
        mask.inputs.in_file = f"{pt}/PET_crop.nii.gz"
        mask.inputs.mask_file = f"{pt}/CT_brain_mask_to_PET.nii.gz"
        mask.inputs.out_file = f"{pt}/PET_crop_BET.nii.gz" #Final marius pet scan
        mask.run()
        print('\tskull stripped PET.')

for pt in path_data.glob('nii/*/*'):
    if limit == 0: break
    limit -= 1
    print(pt)
    bet_PET(pt)



# In[6]:


# Block: ..
print("Resampling to 2mm isotropic...")

limit = limit_g
rsl = tio.Resample(2)

def resample_to_isotropic(pt):
    for file in ['CT_BET_rslPET_crop','PET_crop_BET','CT_brain_mask_to_PET']:
        if pt.joinpath(f'{file}.nii.gz').exists() and not pt.joinpath(f"2mm{file}.nii.gz").exists():
            if file.endswith('mask'):
                img = tio.LabelMap(pt.joinpath(f'{file}.nii.gz'))
            else:
                img = tio.ScalarImage(pt.joinpath(f'{file}.nii.gz'))
            rslImg = rsl(img)
            rslImg.save(pt.joinpath(f"2mm{file}.nii.gz"))
            print('\tsaved 2mm for',file)

for pt in path_data.glob('nii/*/*'):
    if limit == 0: break
    limit -= 1
    print(pt)
    resample_to_isotropic(pt)




# In[7]:


# Block: saves registrations as pdf
def plot_registration(pt, pdf, ctmin=0, ctmax=300):
    if pt.joinpath('2mmPET_crop_BET.nii.gz').exists() and pt.joinpath('2mmCT_BET_rslPET_crop.nii.gz').exists():
        PET = tio.ScalarImage(f"{pt}/2mmPET_crop_BET.nii.gz")
        CT = tio.ScalarImage(f"{pt}/2mmCT_BET_rslPET_crop.nii.gz")
        
        fig, ax = plt.subplots(1,2,dpi=300,figsize=(20,10))
        fig.suptitle(str(pt))

        PET_slice = np.argmax(np.sum(PET.numpy(),(0,1)))
        CT_slice = np.argmax(np.sum(CT.numpy(),(0,1)))

        ax[0].imshow(PET.numpy()[0,:,:,50], alpha=0.5)
        ax[0].set_title('PET 2mm BET')

        ax[1].imshow(CT.numpy()[0,:,:,50], alpha=0.5)
        ax[1].set_title('CT 2mm BET resampled to PET 2mm BET')

        pdf.savefig(fig)

if save_pdf:
    print("saving registrations to pdf...")

    pdf = PdfPages(f"{path_data}/registrations_CT_to_PET.pdf")

    for pt in path_data.glob('nii/anonymized/*'):
        plot_registration(pt, pdf)

    pdf.close()

    pdf = PdfPages(f"{path_data}/registrations_controls_CT_to_PET.pdf")

    for pt in path_data.glob('nii/controls/*'):
        plot_registration(pt, pdf)

    pdf.close()

