#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
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
import warnings


import cv 
import random
from detectron2.structures import BoxMode
import torchio as tio
warnings.filterwarnings("ignore")


# In[2]:


def make_image(image_id, name, width, height, bbox):
    annotation = []
    if bbox != None:
        annotation = make_annotation(bbox)
    if len(bbox)>0:
        image = {
        'file_name':name,
        'image_id': image_id,
        'width': width,
        'height': height,
        'annotations': [make_annotation(bbox)]

        }
    else:
        image = {
        'file_name':name,
        'image_id': image_id,
        'width': width,
        'height': height,
        'annotations': []
        }
        
    return image

def make_annotation(bbox):
    annotation = {
    'bbox': bbox,
    "bbox_mode": BoxMode.XYXY_ABS,
    "segmentation" : [],
    "category_id": 0
    }
    return annotation


# In[3]:


anonymized_path = Path('../data/nii/anonymized')
control_path = Path('../data/nii/controls')
save_path_all = Path('../data/frcnn_3channel')
data_path = Path('../data')
im_save_path = Path('')
annotation_save_path = Path('')
len_pt_name = 8


# In[ ]:





# In[4]:


#save also image
img_id = 0
annotation_id = 0
cut_off = 0.6 #only use 60 percent of diameter
image_size = 0.05 #cm
df_annotations = pd.read_pickle(data_path/'cropped_reference_markings_revised.pkl')
#df_cerrebellum_avg = pd.read_pickle(data_path/'cerrebellum_avg.pkl')


def process_annotated_nifty(pt_path, split, pet='final_cropped_128_pet.nii.gz', pet_norm='final_cropped_128_pet_norm.nii.gz', CT='final_cropped_128_CT.nii.gz', width=512, height=512):
    images_ax = []
    images_cor = []
    images_sag = []

    global img_id


    pet_img = tio.ScalarImage(pt_path / pet)
    transform = tio.Resize((512,512,512))
    image_pet_trans = transform(pet_img)
    pet_img = image_pet_trans.data
    pet_img = np.array(pet_img[0]).astype(np.float32)
    
    
    pet_img_norm = tio.ScalarImage(pt_path / pet_norm)
    transform = tio.Resize((512,512,512))
    image_norm_trans = transform(pet_img_norm)
    pet_img_norm = image_norm_trans.data
    pet_img_norm = np.array(pet_img_norm[0]).astype(np.float32)
    
    CT_img = tio.ScalarImage(pt_path / CT)
    transform = tio.Resize((512,512,512))
    CT_trans = transform(CT_img)
    CT_img = CT_trans.data
    CT_img = np.array(CT_img[0]).astype(np.float32)
    
    nifty_id = str(pt_path)[len(str(pt_path)) - len_pt_name:]
    df_ann = df_annotations[df_annotations['ID'] == nifty_id]
    
    
    df_ann['key'] = df_ann.index
    df_ann['anno_axial'] = [[] for i in range(len(df_ann))]
    df_ann['anno_coronal'] = [[] for i in range(len(df_ann))]
    df_ann['anno_sagital'] = [[] for i in range(len(df_ann))]
    
    
    
    inv = npl.inv(image_pet_trans.affine)
    df_ann['new_voxel'] = df_ann.apply(lambda x : apply_affine(inv, (x.world_x, x.world_y, x.world_z)), axis=1)
    
    
    
    #Find the overlapping annotations for each of the images
    for idx, i in enumerate(df_ann.iloc()):
        for kdx, k in enumerate(df_ann.iloc()):
            if i.key == k.key:
                continue
            #divided by 0.1 because we are interested in the radius
            if abs(i.new_voxel[2] - k.new_voxel[2]) < abs((float(i.diameter_in_cm.replace(',','.'))/0.1)*cut_off + (float(k.diameter_in_cm.replace(',','.'))/0.1)*cut_off):
                 i.anno_axial.append(k.key) 
    
    for idx, i in enumerate(df_ann.iloc()):
        for kdx, k in enumerate(df_ann.iloc()):
            if i.key == k.key:
                continue
            #divided by 0.1 because we are interested in the radius
            if abs(i.new_voxel[1] - k.new_voxel[1]) < abs((float(i.diameter_in_cm.replace(',','.'))/0.1)*cut_off + (float(k.diameter_in_cm.replace(',','.'))/0.1)*cut_off):
                 i.anno_coronal.append(k.key) 
    
    for idx, i in enumerate(df_ann.iloc()):
        for kdx, k in enumerate(df_ann.iloc()):
            if i.key == k.key:
                continue
            #divided by 0.1 because we are interested in the radius
            if abs(i.new_voxel[0] - k.new_voxel[0]) < abs((float(i.diameter_in_cm.replace(',','.'))/0.1)*cut_off + (float(k.diameter_in_cm.replace(',','.'))/0.1)*cut_off):
                 i.anno_sagital.append(k.key) 
    
    
    
    for i in df_ann.iloc():
        
        #Diameter of sphere bounded by cutoff
        
        voxel_diameter = float(i.diameter_in_cm.replace(',','.'))/0.05 #Convert to mm
        voxel_offset_diameter = int(voxel_diameter*cut_off)
        
        center = i.new_voxel
        voxel_center = [int(center[0]), int(center[1]), int(center[2])]
        
        
        
        
        
        for k in range(voxel_offset_diameter):
            
            #use pythogorim theorem: int(np.sqrt((max_rad)**2 - ((k+1)**2)))
            #discrete distance form k to center= k has to be zero when center = slice-idx or something
            slice_idx = list(map(lambda x : x -(int(voxel_offset_diameter/2))+k, voxel_center))
            new_radius = (np.sqrt((voxel_diameter/2)**2 - ((abs((voxel_offset_diameter/2)-k))**2)))
            

            #Axial View

            axial_img = np.stack((pet_img[:,:,slice_idx[2]],pet_img_norm[:,:,slice_idx[2]],CT_img[:,:,slice_idx[2]]), axis=-1)



            #For png nomalization
            max_val = np.amax(axial_img)



            #save_image_as_png
            #m = Image.fromarray(((axial_img*(1/max_val))*255).astype(np.uint8), mode="RGB")

            save_path = Path(save_path_all/ 'axial' /f'{split}'/ f"3channel_anonymized_axial_id_{img_id}")
            np.save(save_path, axial_img)

            #annotation for axial image and image

            radius = voxel_diameter/2
            images_ax.append(make_image(
                                     image_id = img_id,
                                     name=f"3channel_anonymized_axial_id_{img_id}.npy",
                                     width=width,
                                     height=height,
                                     bbox=[voxel_center[0]-radius, voxel_center[1]-radius, voxel_center[0]+radius, voxel_center[1]+radius],
                                    ))


            #Change to new id's
            img_id=img_id+1
            for j in i['anno_axial']:
                annos = df_ann[df_ann['key'] == j]
                diameter1 = float(annos.diameter_in_cm.iloc[0].replace(',','.'))/0.05 
                
                center1 = annos.new_voxel.iloc[0]
                radius1 = diameter1/2
                voxel_center1 = [int(center1[0]), int(center1[1]), int(center1[2])]
                
                
                images_ax[-1]['annotations'].append(make_annotation([voxel_center1[0]-radius1, voxel_center1[1]-radius1, voxel_center1[0]+radius1, voxel_center1[1]+radius1]))
                

            #Coronal View
            coronal_img = np.stack((pet_img[:,slice_idx[1],:],pet_img_norm[:,slice_idx[1],:],CT_img[:,slice_idx[1],:]), axis=-1)
            save_path = str(save_path_all / 'coronal' /f'{split}'/ f"3channel_anonymized_coronal_id_{img_id}")
            np.save(save_path, coronal_img)

            #annotation for coronal image

            images_cor.append(make_image(
                                 image_id = img_id,
                                 name=f"3channel_anonymized_coronal_id_{img_id}.npy",
                                 width=width,
                                 height=height,
                                 bbox=[voxel_center[0]-radius, voxel_center[2]-radius, voxel_center[0]+radius, voxel_center[2]+radius],
                                ))


            #Change to new id's
            img_id=img_id+1
            
            for j in i['anno_coronal']:
                annos = df_ann[df_ann['key'] == j]
                diameter1 = float(annos.diameter_in_cm.iloc[0].replace(',','.'))/0.05 
                
                center1 = annos.new_voxel.iloc[0]
                radius1 = diameter1/2
                voxel_center1 = [int(center1[0]), int(center1[1]), int(center1[2])]
                images_cor[-1]['annotations'].append(make_annotation([voxel_center1[0]-radius1, voxel_center1[2]-radius1, voxel_center1[0]+radius1, voxel_center1[2]+radius1]))


            


            #Sagital View
            sagtial_img = np.stack((pet_img[slice_idx[0],:,:],pet_img_norm[slice_idx[0],:,:],CT_img[slice_idx[0],:,:]), axis=-1)                

            #im = Image.fromarray(((sagtial_img*(1/max_val))*255).astype(np.uint8), mode="RGB")
            save_path = str(save_path_all / 'sagital'/f'{split}'/ f"3channel_anonymized_sagital_id_{img_id}")
            #im.save(save_path)
            np.save(save_path, sagtial_img)


            #annotation for sagital image
            images_sag.append(make_image(
                                 image_id = img_id,
                                 name=f"3channel_anonymized_coronal_id_{img_id}.npy",
                                 width=width,
                                 height=height,
                                 bbox=[voxel_center[1]-radius, voxel_center[2]-radius, voxel_center[1]+radius, voxel_center[2]+radius],
                                ))
            #Change to new id's
            img_id=img_id+1
            
            for j in i['anno_sagital']:
                annos = df_ann[df_ann['key'] == j]
                diameter1 = float(annos.diameter_in_cm.iloc[0].replace(',','.'))/0.05 
                
                center1 = annos.new_voxel.iloc[0]
                radius1 = diameter1/2
                voxel_center1 = [int(center1[0]), int(center1[1]), int(center1[2])]
                images_sag[-1]['annotations'].append(make_annotation([voxel_center1[1]-radius1, voxel_center1[2]-radius1, voxel_center1[1]+radius1, voxel_center1[2]+radius1]))
                
        
    return images_ax, images_sag, images_cor


# In[5]:


def process_control_nifty(pt_path, split, slice_per_axis, pet='final_cropped_128_pet.nii.gz', pet_norm='final_cropped_128_pet_norm.nii.gz', CT='final_cropped_128_CT.nii.gz', width=128, height=128):
    images_ax = []
    images_cor = []
    images_sag = []
    global img_id
    
    
    pet_img = tio.ScalarImage(pt_path / pet)
    transform = tio.Resize((512,512,512))
    image_pet_trans = transform(pet_img)
    pet_img = image_pet_trans.data
    pet_img = np.array(pet_img[0]).astype(np.float32)
    
    
    pet_img_norm = tio.ScalarImage(pt_path / pet_norm)
    transform = tio.Resize((512,512,512))
    image_norm_trans = transform(pet_img_norm)
    pet_img_norm = image_norm_trans.data
    pet_img_norm = np.array(pet_img_norm[0]).astype(np.float32)
    
    CT_img = tio.ScalarImage(pt_path / CT)
    transform = tio.Resize((512,512,512))
    CT_trans = transform(CT_img)
    CT_img = CT_trans.data
    CT_img = np.array(CT_img[0]).astype(np.float32)
    
    #Uniformly slicing over every_brain possibly not desirable
    indexes = np.random.randint(100, 400, slice_per_axis)
    
    for i in indexes:
        
        #Axial View
        axial_img = np.stack((pet_img[:,:,i],pet_img_norm[:,:,i],CT_img[:,:,i]),axis=-1)

        save_path = str(save_path_all/ 'axial'/f'{split}' / f"3channel_anonymized_axial_id_{img_id}")
        np.save(save_path, axial_img)

        
        #annotation for axial image and image
        images_ax.append(make_image(
                                 image_id = img_id,
                                 name=f"3channel_anonymized_axial_id_{img_id}.npy",
                                 width=width,
                                 height=height,
                                 bbox=[],
                                ))
        img_id=img_id+1

        
        
        
        coronal_img = np.stack((pet_img[:,i,:],pet_img_norm[:,i,:],CT_img[:,i,:]), axis=-1)
        save_path = str(save_path_all / 'coronal'/f'{split}' / f"3channel_anonymized_coronal_id_{img_id}")
        np.save(save_path, coronal_img)

        #annotation for coronal image

        images_cor.append(make_image(
                             image_id = img_id,
                             name=f"3channel_anonymized_coronal_id_{img_id}.npy",
                             width=width,
                             height=height,
                             bbox=[],
                            ))
        #Change to new id's
        img_id=img_id+1

        
        sagtial_img = np.stack((pet_img[i,:,:],pet_img_norm[i,:,:],CT_img[i,:,:]),axis=-1)
        save_path = str(save_path_all / 'sagital'/f'{split}'/ f"3channel_anonymized_sagital_id_{img_id}")
        np.save(save_path, sagtial_img)


        #annotation for sagital image
        images_sag.append(make_image(
                             image_id = img_id,
                             name=f"3channel_anonymized_coronal_id_{img_id}.npy",
                             width=width,
                             height=height,
                             bbox=[],
                            ))
        img_id=img_id+1
    return images_ax, images_sag, images_cor


# In[6]:


import random
frac_ctrl = 0.2
slc_per_ctrl = 2
def all_annotated_images(anon_scans1, control_scans, split):
    all_scans = anon_scans1
    sag = []
    cor = []
    ax = []
    all_annotations = []
    
    for im in all_scans:
        images_ax, images_sag, images_cor = process_annotated_nifty(im, split)
        ax += images_ax
        cor += images_cor
        sag += images_sag
        print('One_Done')
    
    
    ctr_l = len(control_scans)
    ctrl_slc = len(ax)*frac_ctrl
    pt_crl = random.choices(control_scans, k=int(ctrl_slc/slc_per_ctrl))
    
    
    for im in pt_crl:
        images_ax, images_sag, images_cor = process_control_nifty(im, split, slice_per_axis=slc_per_ctrl)
        ax += images_ax
        cor += images_cor
        sag += images_sag
        print('One_Done Control')

    with open(save_path_all / "axial"/f'{split}'/"metastasis_annotations_coco_train.json", "w") as json_file:
        json.dump(ax, json_file)
        
    with open(save_path_all / "coronal"/f'{split}'/"metastasis_annotations_coco_train.json", "w") as json_file:
        json.dump(cor, json_file)
        
    with open(save_path_all / "sagital"/f'{split}'/"metastasis_annotations_coco_train.json", "w") as json_file:
        json.dump(sag, json_file)  
    return ax


# In[7]:


from sklearn.model_selection import train_test_split
def split_anon():
    train = []
    test = []
    val =  []
    for pt in (anonymized_path/"train").glob("*/"):
        train.append(pt)


    for pt in (anonymized_path/"test").glob("*/"):
        test.append(pt)


    for pt in (anonymized_path/"val").glob("*/"):
        val.append(pt)


    return train, test, val

def split_control():


    train = []
    test = []
    val =  []

    for pt in (control_path/"train").glob("*/"):
        train.append(pt)


    for pt in (control_path/"test").glob("*/"):
        test.append(pt)


    for pt in (control_path/"val").glob("*/"):
        val.append(pt)


    return train, test, val

# In[8]:


train, test, val = split_anon()
train_ctr, test_ctr, val_ctr = split_control()

print(train)


# In[91]:


all_annotated_images(train, train_ctr, 'train')
all_annotated_images(val, val_ctr, 'val')
all_annotated_images(test, test_ctr, 'test')

