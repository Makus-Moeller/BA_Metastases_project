#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import os


# In[36]:


#Generated with random number generator
seed = 61260


# In[37]:


#Get paths
path_data = Path('../data')
anonymized_path = path_data / "nii/anonymized/"
control_path =  path_data / "nii/controls/"
controls = []
anonymized = []
for pt in anonymized_path.glob("train/Cmet*/"):
    anonymized.append(pt)
for pt in control_path.glob("train/Cmet*/"):
    controls.append(pt)
    
controls = np.array(controls)
anonymized = np.array(anonymized)


# In[38]:


X_train_control, X_val_control, _, _ = train_test_split(controls, np.zeros(len(controls)), test_size=0.20, random_state=seed)
X_train_anonymized, X_val_anonymized, _, _ = train_test_split(anonymized, np.zeros(len(anonymized)), test_size=0.20, random_state=seed)

for i in X_val_control:
    print(i)
    os.system(f'mv {i} {control_path}/val')


for i in X_val_anonymized:
    os.system(f'mv {i} {anonymized_path}/val')

# In[39]:





#df_train_control = pd.DataFrame(data=np.reshape(X_train_control, (len(X_train_control), -1)),columns=['pt'])
#df_train_anonymized = pd.DataFrame(data=np.reshape(X_train_anonymized, (len(X_train_anonymized), -1)),columns=['pt'])

#df_test_control = pd.DataFrame(data=np.reshape(X_test_control, (len(X_test_control), -1)),columns=['pt'])
#df_test_anonymized = pd.DataFrame(data=np.reshape(X_test_anonymized, (len(X_test_anonymized), -1)),columns=['pt'])





# In[41]:


#df_train_control.to_csv(path_data/'train_control.csv')
#df_train_anonymized.to_csv(path_data/'train_anonymized.csv')
#df_test_control.to_csv(path_data/'test_control.csv')
#df_test_anonymized.to_csv(path_data/'test_anonymized.csv')


# In[ ]:





# In[ ]:




