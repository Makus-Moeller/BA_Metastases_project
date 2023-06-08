#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
from nipype.interfaces.ants import ApplyTransformsToPoints
import h5py


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


path_data = Path('../data/')
anonymized_data = path_data / 'nii/anonymized/'

np.set_printoptions(formatter={'float_kind':"{:.2f}".format})


# In[5]:


def rtss_reg(row, pet_name, ct_name, resample=True, do_plot=False, extra_resampling=None, pdf=None):
    container = tio.ScalarImage(anonymized_data / '{}/{}.nii.gz'.format(row.ID, pet_name if row.linked_to == 'PT' else ct_name))
    
    if extra_resampling is not None:
        rsl = tio.Resample(extra_resampling)
        container = rsl(container)

    world = [row.world_x, row.world_y, row.world_z]
    voxel = apply_affine(npl.inv(container.affine), world)
    voxel_i = list(map(round, voxel))

    if do_plot & (pdf is not None):
        try:
            other = tio.ScalarImage(anonymized_data / '{}/{}.nii.gz'.format(row.ID, pet_name if not row.linked_to == 'PT' else ct_name))
            if extra_resampling is not None:
                other = rsl(other)

            # PLOT
            fig, axes = plt.subplots(1, 2, dpi=50, sharex=True, sharey=True, figsize=(10, 5))

            # Load other image as well
            if resample:
                rsl = tio.Resample(container)
                other = rsl(other)

            if row.linked_to == "CT":
                cmap_container = 'gray'
                cmap_other = _PETRainbowCMAP
            else:
                cmap_other = 'gray'
                cmap_container = _PETRainbowCMAP

            titles = [
                'PT: {}, Location: {}'.format(row.ID, voxel_i), 
                'Classification: {}. Diameter in cm: {}.'.format(row.classificaton, row.diameter_in_cm)
            ] 
            for ax_ind, (img, cmap, title) in enumerate(zip([container, other], [cmap_container, cmap_other], titles)):
                ax = axes[ax_ind]
                ax.imshow(np.fliplr(np.rot90(img.numpy()[0,:,:,voxel_i[2]], 3)), cmap=cmap)
                ax.plot(voxel_i[0], voxel_i[1], 'bo', alpha=.3)
                ax.set_title(title)
                ax.axis('off')
                
            plt.tight_layout()
            pdf.savefig(fig)
            print(f"\n\nworked for pt {row.ID}")
        except:
            print(f"\n\nerror for pt {row.ID}")

    return voxel


# In[11]:


df = pd.read_pickle(path_data / 'reference_markings.pkl')


# In[16]:


df_final = pd.DataFrame(columns=['ID','world_x', 'world_y', 'world_z','voxel_x', 'voxel_y', 'voxel_z','classificaton','diameter_in_cm', 'linked_to'])

pdf = PdfPages(f"{path_data}/rtss_registrations_test.pdf")

for pt in df.iloc:
    if not (anonymized_data / f"{pt.ID}/2mmCT_BET_rslPET_crop.nii.gz").exists():
        continue

    if pt.linked_to == 'CT':
        
        df_reg = pd.DataFrame({'ID': [pt.ID], 'world_x': [pt.world_x], 'world_y': [pt.world_y], 'world_z': [pt.world_z], 't': [0]})
              
        csv = df_reg[['world_x', 'world_y', 'world_z', 't']].rename(columns={'world_x':'x', 'world_y':'y', 'world_z':'z'})
        
        csv.to_csv(f'temp.csv', index=False)
        
        #Why the hell should we use the Inverse???
        os.system(f'antsApplyTransformsToPoints -d 3 -i temp.csv -o temp_reg.csv -t [{anonymized_data}/{pt.ID}/Composite.h5, 1]')
                
        csv = pd.read_csv('temp_reg.csv')
        
        #Update world cordinates
        pt['world_x'] = csv.x[0]
        pt['world_y'] = csv.y[0]
        pt['world_z'] = csv.z[0]
        
        voxel = rtss_reg(pt, pet_name='2mmPET_crop_BET', ct_name='2mmCT_BET_rslPET_crop', resample=False, do_plot=False, pdf=pdf)
    else:
        
        #intermediaryb df used cause it does not work without
        df_reg = pd.DataFrame({'ID': [pt.ID], 'world_x': [pt.world_x], 'world_y': [pt.world_y], 'world_z': [pt.world_z], 't': [0]})
        csv = df_reg[['world_x', 'world_y', 'world_z']].rename(columns={'world_x':'x', 'world_y':'y', 'world_z':'z'})
        voxel = rtss_reg(pt, pet_name='2mmPET_crop_BET', ct_name='2mmCT_BET_rslPET_crop', resample=False, do_plot=False, pdf=pdf)

    df_final = df_final.append(pd.DataFrame({'ID': pt.ID, 'world_x': csv.x, 'world_y': csv.y, 'world_z': csv.z, 'voxel_x': voxel[0], 'voxel_y': voxel[1], 'voxel_z': voxel[2], 'classificaton': pt.classificaton, 'diameter_in_cm': pt.diameter_in_cm, 'linked_to': pt.linked_to}, index=[0]), ignore_index=True)
    
pdf.close()
df_final.to_pickle(path_data / 'registered_reference_markings_revised.pkl')

