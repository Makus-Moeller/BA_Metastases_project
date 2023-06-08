#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!python -m pip install pyyaml==5.1
# Detectron2 has not released pre-built binaries for the latest pytorch (https://github.com/facebookresearch/detectron2/issues/4053)
# so we install from source instead. This takes a few minutes.
#!python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install pre-built detectron2 that matches pytorch version, if released:
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
#!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/{CUDA_VERSION}/{TORCH_VERSION}/index.html

#exit(0)  # After installation, you may need to "restart runtime" in Colab. This line can also restart runtime


# In[1]:


#!pip install --quiet torchio==0.18.83


# In[2]:





# In[1]:


import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os
import random
import cv2
import json
import torch
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torchio as tio
import logging
from termcolor import colored
from detectron2.utils.visualizer import ColorMode
# common detectron2 utilities
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg
from detectron2.structures import BoxMode

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime



# In[2]:


data_path = Path("../data/frcnn_data/axial")


# In[3]:


# we simple load in the data file into dictionary
def get_board_dicts(imgdir, data_list):
    json_file = Path(imgdir / "metastasis_annotations_coco_train.json")
    with open(json_file) as f:
        dataset_dicts = json.load(f)
    

    filtered_filenames = [d for d in dataset_dicts if (d["file_name"]) in data_list]
    dataset_dicts = filtered_filenames
    print(len(dataset_dicts) == len(data_list))

    for i in dataset_dicts:
        
        filename = i["file_name"] 
        i["file_name"] = imgdir / filename 
        i["width"] = int(i["width"])
        i["height"] = int(i["height"])

        for j in i["annotations"]:
            j["bbox"] = [float(num) for num in j["bbox"]]
            j["bbox_mode"] = int(j["bbox_mode"]) # BoxMode.XYXY_ABS
            j["category_id"] = int(j["category_id"])


    return dataset_dicts

# We make a mapping function, which loads in the given dataset dictionary
# It maps it using torch and we return another dictionary we can use when training.
# Each dictionary is an image
def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)

    image = np.load(dataset_dict["file_name"])

    auginput = T.AugInput(image)
    image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
    annos = [
        utils.transform_instance_annotations(annotation, [], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]

    return {
       "image": image,
       "image_id": dataset_dict["image_id"],
       "width": dataset_dict["width"],
       "height": dataset_dict["height"],
       "instances": utils.annotations_to_instances(annos, image.shape[1:])
    }

def mapper2(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    image = np.load(dataset_dict["file_name"])
    transform_list = [T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                      T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                      ]

    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = instances#utils.filter_empty_instances(instances)
    return dataset_dict

# In[4]:


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        losses1.append(mean_loss)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        print('Mean Validation loss is: ', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        print(colored("We are at iteration: ", 'blue'), self.trainer.iter)
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0) or self._period == 0:
            print(colored("Time for validation", 'red'))
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

losses1 = []
losses2 = []

class LossTrainEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        losses2.append(mean_loss)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        print('Mean Traom loss is: ', mean_loss)
        comm.synchronize()

        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        print(colored("We are at iteration: ", 'blue'), self.trainer.iter)
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0) or self._period == 0:
            print(colored("Time for validation", 'red'))
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)
# In[5]:


class SyntheticTrainer(DefaultTrainer):
    """
    Custom Trainer deriving from the "DefaultTrainer"

    Overloads build_hooks to add a hook to calculate loss on the test set during training.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        print("using evaluator")
        return COCOEvaluator(dataset_name, tasks=['bbox'], output_dir=cfg.OUTPUT_DIR)
        
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        print("using test loader")
        return detectron2.data.build_detection_test_loader(cfg, dataset_name, mapper=mapper)
    
    @classmethod
    def build_train_loader(cls, cfg):
        print("using train loader")
        return detectron2.data.build_detection_train_loader(cfg, mapper=mapper2)

    def build_hooks(self):
        hooks = super().build_hooks()
        
        hooks.insert(-1, LossEvalHook(
            cfg.TEST.EVAL_PERIOD, # Frequency of calculation - every eval period
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST,
                mapper = mapper
            )
        ))

        return hooks


# In[7]:


import glob

train_data = []
data_path_train = Path(data_path /"train")
for filename in os.listdir(data_path_train):
  if filename.endswith("npy"):
    train_data.append(filename)


test_data = []
data_path_test = Path(data_path /"val")
for filename in os.listdir(data_path_test):
  if filename.endswith("npy"):
    test_data.append(filename)





# In[9]:


# Number of folds
num_folds = 5
num_img = int(len(train_data)/num_folds)
# Shuffle the data
np.random.shuffle(train_data)

DatasetCatalog.clear()

DatasetCatalog.register("brain_metastasis_" + 'train', lambda d=data_path_train: get_board_dicts(d, train_data))
MetadataCatalog.get("brain_metastasis_" + 'train').set(thing_classes=["TUMOR"])


DatasetCatalog.register("brain_metastasis_" + 'val', lambda d=data_path_test: get_board_dicts(d, test_data))
MetadataCatalog.get("brain_metastasis_" + 'val').set(thing_classes=["TUMOR"])
print("datasets: ",DatasetCatalog.list())


#https://detectron2.readthedocs.io/en/latest/modules/config.html For more training options
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = [f"brain_metastasis_train"]
cfg.DATASETS.TEST = [f"brain_metastasis_val"]

# Number of data loading threads
cfg.DATALOADER.NUM_WORKERS = 2
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # weights from detectron2 model zoo

# Number of images per batch across all machines.
# SOLVER.IMS_PER_BATCH is what's commonly known as batch size in deep learning and refers to the number of training examples utilized in one iteration.
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.000125 
cfg.SOLVER.MAX_ITER = 10000
#cfg.SOLVER.STEPS = []
#cfg.SOLVER.AMP.ENABLED = False
# MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE is a parameter that is used to sample a subset of proposals coming out of RPN to calculate cls and reg loss during training.
# - rpn: A Region Proposal Network, or RPN, is a fully convolutional network that simultaneously predicts object bounds and objectness scores at each position.
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.TEST.EVAL_PERIOD = 50


cfg.OUTPUT_DIR = "../data/frcnn_data/axial/3pet_empty"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = SyntheticTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


# In[15]:

loss = {'VAL: ': losses1}
loss_df = pd.DataFrame(loss)
loss_df.to_csv(data_path/"loss_3pet_empty")



# In[ ]:






def get_board_dicts2(imgdir):
    json_file = Path(imgdir / "metastasis_annotations_coco_train.json")
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
            j["bbox_mode"] = int(j["bbox_mode"]) # BoxMode.XYWH_ABS
            j["category_id"] = int(j["category_id"])


    return dataset_dicts









#cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   # set a custom testing threshold
#predictor = DefaultPredictor(cfg)

#data = get_board_dicts2(Path("../frcnn_data/axial/test"))
#for idx,i in enumerate(data):
#    img = np.load(i['file_name'])
#    outputs = predictor(img)
#    print(outputs)
#    v = Visualizer(img[:, :, ::-1],
#                   metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]), 
#                   scale=0.5, 
#                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#    )
#    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#    cv2.imwrite(f"../frcnn_data/test{idx}.png", out.get_image()[:, :, ::-1])


# %%
