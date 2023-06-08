#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from pathlib import Path
import torchio as tio
import numpy as np
import os


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


path_data = Path('../data')
anonymized_data = path_data / 'nii/anonymized'

np.set_printoptions(formatter={'float_kind':"{:.2f}".format})


# In[ ]:


largest_brain_dimensions = [0, 0, 0]


# In[ ]:


image_path = path_data / "nii/anonymized/"
control_path =  path_data / "nii/controls/"
output_folders = []
control_folders = []
for pt in image_path.glob("*/"):
    output_folders.append(pt)
for pt in control_path.glob("*/"):
    output_folders.append(pt)
    control_folders.append(pt)


# In[ ]:


limit = -1
for pt_folder in output_folders:
    if limit == 0: break
    limit -= 1
   
    #make subject
    subject_pt = tio.Subject(
        t1 = tio.ScalarImage(pt_folder/'2mmPET_crop_BET.nii.gz'),
        seg = tio.LabelMap(pt_folder/'2mmCT_brain_mask_to_PET.nii.gz'),
        )
   
    #make cropping
    transform = tio.CropOrPad(
        None,
        mask_name = 'seg',
    )

    transformed = transform(subject_pt)
    largest_brain_dimensions[0] = max(largest_brain_dimensions[0], transformed.shape[1])
    largest_brain_dimensions[1] = max(largest_brain_dimensions[1], transformed.shape[2])
    largest_brain_dimensions[2] = max(largest_brain_dimensions[2], transformed.shape[3])
print(largest_brain_dimensions)

