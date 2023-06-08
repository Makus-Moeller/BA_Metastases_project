#!/usr/bin/env python
# coding: utf-8

# In[54]:


import numpy as np
import pandas as pd
from pathlib import Path
import json
from PIL import Image
from nibabel.affines import apply_affine
import numpy.linalg as npl
import os
import torch
from PIL import Image
from matplotlib import pyplot as plt
import cv2
import torchio as tio


# In[55]:


anonymized_path = Path('../data/nii/anonymized')
control_path = Path('../data/nii/controls')
data_path = Path('../data')
all_scans = []


# In[56]:


for pt in anonymized_path.glob("*/"):
        all_scans.append(pt)

for pt in control_path.glob("*/"):
        all_scans.append(pt)


# In[57]:


for pt in all_scans:
    try:
        mask  = tio.Subject(
            brain = tio.LabelMap(pt / '2mmCT_brain_mask_to_PET.nii.gz'),
            brain_mask = tio.LabelMap(pt / '2mmCT_brain_mask_to_PET.nii.gz')
        )
    
        transform = tio.CropOrPad(
            (128,128,128),
            mask_name = 'brain_mask',
        )
        mask_128 = transform(mask)
        mask_128['brain'].save(pt / '2mmCT_brain_mask_to_PET_128.nii.gz')
    
        subject_pt = tio.Subject(
            img = tio.ScalarImage(pt / 'final_cropped_128_pet.nii.gz'),
            brain = tio.LabelMap(pt / '2mmCT_brain_mask_to_PET_128.nii.gz')
            )
    
        normalizer = tio.ZNormalization(masking_method='brain')
        norm_img = normalizer(subject_pt)
        norm_img['img'].save(pt / 'final_cropped_128_pet_norm.nii.gz')
    except:
        print('didnt work for:  ', pt)


# In[64]:





# In[59]:





# In[ ]:





# In[ ]
