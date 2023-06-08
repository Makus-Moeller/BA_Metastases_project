#!/usr/bin/env python
# coding: utf-8

# In[251]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image
from nibabel.affines import apply_affine
import numpy.linalg as npl
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import cv2
import torchvision.transforms as T
from nibabel.affines import apply_affine
import numpy.linalg as npl

import cv 
import random
from detectron2.structures import BoxMode
import torchio as tio


# In[252]:


path = Path('../Model_Loss')
files = []
for i in path.glob('*/'):
    files.append(i)
    print(i)


# In[253]:


names = ['Loss on 3 PET Axial',
         'Loss on 3 PET Axial Unfiltered',
         'Loss on 3 PET Coronal',
         'Loss on 3 PET Coronal Unfiltered',
         'Loss on 3 PET Sagital',
         'Loss on 3 PET Sagital Unfiltered',
         'Loss on PET/CT/NormPET Axial',
         'Loss on PET/CT/NormPET Axial Unfiltered',
         'Loss on PET/CT/NormPET Coronal',
         'Loss on PET/CT/NormPET Coronal Unfiltered',
         'Loss on PET/CT/NormPET Sagital',
         'Loss on PET/CT/NormPET Sagital Unfiltered'
    
]


# In[254]:


x_val = [i for i in range(1001) if i %50 == 0 and not i==0]
print(x_val, len(x_val))
fig, ax = plt.subplots(3,4, figsize=(20,20))
y_count = 0
for idx, (i, name) in enumerate(zip(files,names)):
    loss = pd.read_csv(i)
    #name = str(i)[14:]
    
    ax[y_count, idx%4].set_title(name)
    ax[y_count, idx%4].plot(x_val, loss['Train'], label = 'Train')
    ax[y_count, idx%4].plot(x_val, loss['VAL: '], label = 'validation')
    ax[y_count, idx%4].set_xlabel('Iterations')
    ax[y_count, idx%4].set_ylabel('Loss')
    ax[y_count, idx%4].legend()
    if (idx+1)%4==0:
        y_count+=1
plt.show()    


# In[259]:


AP = [0.0,0.0, 0., 0., 0.,0.,0.001,0.001,0.001,0.0,0.002,0.0,0.001,0.003,0.0,0.0,0.0,0.001,0.0,0.009]
AP50 = [0.,0.0, 0.002, 0.003,0.001,0.001,0.009,0.006,0.007,0.00,0.009,0.002,0.006,0.018,0.0,0.0,0.001,0.008,0.003,0.09]
AP75 = [0 for i in range(20)] #They were all zero so didn't want to write them by hand
loss = pd.read_csv('../Model_Loss/loss_each_channel_empty_ax')['VAL: ']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_title('Loss on PET/CT/NormPET Axial Unfiltered')
ax1.plot(x_val, loss, label='validation', color='blue')
ax2.plot(x_val, AP, label='AP', color='red')
ax2.plot(x_val, AP50, label='AP50', color='green')
ax2.plot(x_val, AP75, label='AP75', color='orange')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss', color='blue')
ax2.set_ylabel('AP', color='black')
ax2.legend(loc=('upper right'))


# In[261]:


count=0
x_val = [i for i in range(10001) if i %50 == 0 and not i==0]
ap=[]
ap50=[]
ap75 = []
file =json.load(open('../ax_aug_empty.json'))
for i in file:
    #print((i['iteration']+1)%50)
    if (i['iteration']+1) % 50 ==0 and i['iteration']!=9999:
        ap.append(i['bbox/AP'])
        ap50.append(i['bbox/AP50'])
        ap75.append(i['bbox/AP75'])
ap.append(file[-1]['bbox/AP'])
ap50.append(file[-1]['bbox/AP50'])
ap75.append(file[-1]['bbox/AP75'])        


loss = pd.read_csv('../loss_each_channel_empty_aug')['VAL: ']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_title('Loss on PET/CT/NormPET Axial Unfiltered')
ax1.plot(x_val, loss, label='validation', color='blue')
ax2.plot(x_val, ap, label='AP', color='red')
ax2.plot(x_val, ap50, label='AP50', color='green')
ax2.plot(x_val, ap75, label='AP75', color='orange')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('Loss', color='blue')
ax2.set_ylabel('AP', color='black')
ax2.legend()


# In[245]:


file =json.load(open('../cor_aug_empty.json'))
rpn_loss_cls = 0
rpn_loss_rpn = 0
loss_box_reg = 0
loss_cls = 0
count=0
for i in file:
    try:
        
        rpn_loss_cls+=i['loss_rpn_cls']
        rpn_loss_rpn+=i['loss_rpn_loc']
        loss_cls+=i['loss_cls']
        loss_box_reg+=i['loss_box_reg']
        count+=1
    except:
        continue
    
print(rpn_loss_cls/count)
print(rpn_loss_rpn/count)
print('new')
print(loss_cls/count)
print(loss_box_reg/count)


# In[ ]:


def get_board_dicts(imgdir):
    json_file = Path(imgdir/"metastasis_annotations_coco_cor_val.json")
    with open(json_file) as f:
        dataset_dicts = json.load(f)
    

    #filtered_filenames = [d for d in dataset_dicts if (d["file_name"]) in data_list]
    #dataset_dicts = filtered_filenames
    #print(len(dataset_dicts) == len(data_list))

    for i in dataset_dicts:
        
        filename = i["file_name"] 
        i["file_name"] = imgdir / filename 
        i["width"] = int(i["width"])
        i["height"] = int(i["height"])

        for j in i["annotations"]:
            j["bbox"] = [float(num) for num in j["bbox"]]
            j["bbox_mode"] = BoxMode.XYXY_ABS # BoxMode.XYWH_ABS
            j["category_id"] = int(j["category_id"])


    return dataset_dicts


# In[262]:


from detectron2.utils.visualizer import ColorMode
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from rhscripts.plotting import _PETRainbowCMAP
ground_trouth = json.load(open("../metastasis_annotations_coco_train_old_anno.json"))
print(len(ground_trouth))
outputs = json.load(open('../coco_instances_results_ax.json')) 
for i in ground_trouth:
    if i['image_id'] in [20973, 23697, 24159, 24225]:
        img_id = i['image_id']
      
        fig, ax = plt.subplots()
        im = np.load(f'../3channel_anonymized_axial_id_{img_id}.npy')[:,:,1]
        count = 0
        for k in outputs:
            if k['image_id'] == img_id:
                #if count > 2:
                #    break
                box = k['bbox']
                prediction = patches.Rectangle((new_bbx[0],new_bbx[1]), (new_bbx[2]-new_bbx[0]),(new_bbx[3]-new_bbx[1]), linewidth=2, 
                                                edgecolor="blue", facecolor="none")
                prediction.set_label(f"Tumor; score: {k['score']:.4f}")
                ax.add_patch(prediction)
                count+=1



        box_real = i['annotations']
        count = 0
        for k in box_real:
            new_bbx =k['bbox']        

            ground_truth = patches.Rectangle((new_bbx[0],new_bbx[1]), (new_bbx[2]-new_bbx[0]),(new_bbx[3]-new_bbx[1]), linewidth=2, 
                                           edgecolor="red", facecolor="none")
            if count == 0:
                ground_truth.set_label(f"Annotation")
            ax.add_patch(ground_truth)
            count+=1
        


        ax.legend()
        
        ax.imshow(im, cmap =_PETRainbowCMAP)


# In[139]:


print(len(ground_trouth))
amount = 0
count=0
ground_trouth = json.load(open("../lets_try.json"))
for i in ground_trouth:
    amount +=len(i['annotations'])
    if len(i['annotations']) >0:
        count+=1
print(amount/count)
print(count)


# In[ ]:


anno = 0
empty = 0
for i in ground_trouth:
    if len(i['annotations']) > 0:
        anno +=1
    else:
        empty +=1

print(anno)
print(empty)
print(anno/(empty+anno))
        


# In[ ]:


print(anno+empty)


# In[ ]:


for i in ground_trouth:
    
    if i['image_id'] in [19879, 20065, 20110, 20170, 22573]:
        print("New Image\n")
        img_id = i['image_id']
        fig, (ax1,ax2,ax3) = plt.subplots(3, figsize=(10,7))
        im = np.load(f'../3channel_anonymized_coronal_id_19879.npy')
        print(im.shape)
        break

        box_real = i['annotations']
        count = 0
        
        #Ground Trouth No flip
        for k in box_real:
            new_bbx =k['bbox']        

            ground_truth = patches.Rectangle((new_bbx[0],new_bbx[1]), (new_bbx[2]-new_bbx[0]),(new_bbx[3]-new_bbx[1]), linewidth=2, 
                                           edgecolor="red", facecolor="none")
           
            ax1.add_patch(ground_truth)

        
        #Ground Truth Vertical flip
        for k in box_real:
            new_bbx = k['bbox'].copy()
            new_bbx[1] = 512 - new_bbx[1]
            new_bbx[3] = 512 - new_bbx[3]

            ground_truth = patches.Rectangle((new_bbx[0],new_bbx[1]), (new_bbx[2]-new_bbx[0]),(new_bbx[3]-new_bbx[1]), linewidth=2, 
                                           edgecolor="red", facecolor="none")
            ax2.add_patch(ground_truth)

        
        #Ground Truth Horizontal flip
        for k in box_real:
            new_bbx =k['bbox'].copy()
            new_bbx[0] = 512 - new_bbx[0]
            new_bbx[2] = 512 - new_bbx[2]

            ground_truth = patches.Rectangle((new_bbx[0],new_bbx[1]), (new_bbx[2]-new_bbx[0]),(new_bbx[3]-new_bbx[1]), linewidth=2, 
                                           edgecolor="red", facecolor="none")
            
            ax3.add_patch(ground_truth)
 
        
        
        
        ax1.imshow(im, cmap =_PETRainbowCMAP)
        ax2.imshow(np.flip(im, axis=0), cmap =_PETRainbowCMAP)
        ax3.imshow(np.flip(im, axis=1), cmap =_PETRainbowCMAP)
        


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




