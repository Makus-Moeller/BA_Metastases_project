import pandas as pd
from pathlib import Path
import torchio as tio
import numpy as np
from pathlib import Path
from itertools import chain
import warnings
warnings.filterwarnings('ignore')

data_path = Path('../data/nii')

df = pd.DataFrame(columns=['modality', 'pixel_spacing', 'thickness'])

for patient in chain(data_path.glob('anonymized/*/*'), data_path.glob('controls/*/*')):
    ct_image = tio.ScalarImage(patient/'CT.nii.gz')
    pet_image = tio.ScalarImage(patient/'PET.nii.gz')
    df = df.append(pd.DataFrame({'modality': 'CT', 'pixel_spacing': round(ct_image.spacing[0], 2), 'thickness': round(ct_image.spacing[2], 2)}, index=[0]), ignore_index=True)
    df = df.append(pd.DataFrame({'modality': 'PET', 'pixel_spacing': round(pet_image.spacing[0], 2), 'thickness': round(pet_image.spacing[2], 2)}, index=[0]), ignore_index=True)

print(f"{len(df.index.unique())} patients were loaded\n") 

print("CT pixelspacing:")
print(df[df['modality'] == 'CT'].pixel_spacing.value_counts())
print("thicknes")
print(df[df['modality'] == 'CT'].thickness.value_counts())

print("PET Pixel Spacing:")
print(df[df['modality'] == 'PET'].pixel_spacing.value_counts())
print('thickness')
print(df[df['modality'] == 'PET'].thickness.value_counts())

