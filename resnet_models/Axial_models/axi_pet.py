#!/usr/bin/env python
# coding: utf-8

# In[273]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pathlib import Path


# In[274]:


torch.__version__


# In[282]:

model_name = 'axi_pet'
data_path = Path("../../data/")
batch_size = 8

torch.cuda.empty_cache()


# ## Data preperation

# In[283]:

torch.multiprocessing.set_start_method('spawn')

#Overwrite getitem and len 
#See https://pytorch.org/docs/stable/data.html#map-style-datasets 

class BrainTumorDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.cuda.FloatTensor(targets) #OBS special datatype when runnig on GPU
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.data[index]
        x = torch.cuda.FloatTensor(x).permute(2,0,1)
        y = self.targets[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y
    
    def __len__(self):
        return len(self.data)


# In[284]:


#Make grayscale image to three channels
def expand_np_image_to_3_channels(grey_image_np):
    grey_image_arr = np.expand_dims(grey_image_np, -1)
    grey_image_arr_3_channel = grey_image_arr.repeat(3, axis=-1)
    return grey_image_arr_3_channel

#Used to meassure accuracy for each epoch
def correct_number_of_classifications(y_true, y_pred, return_y_pred = False):
    y_pred = y_pred >= 0.5
    if return_y_pred:
        return (y_pred==y_true).sum().item(), y_pred    
    else:
        return (y_pred==y_true).sum().item()


# In[325]:


#Make datastructures
X_train = []
y_train = []
X_valid = []
y_valid = []
X_test = []
y_test = []
features = None
labels = None
label = []
     


# ### For axi only

# In[326]:


#For PET only to begin with

#train

for axi_scan_anony in data_path.glob('nii/anonymized/train/*/axi/PET*'): 
  scan = np.load(axi_scan_anony)
  flipped_scan_horizontal = np.flip(scan,axis=1)
  flipped_scan_vertical = np.flip(scan,axis=0)
  rescaled = scan * 1.1
  #rescaled_scan = ((scan / np.amax(scan))*255).astype(np.uint8) #If we want to squezze values between 0-1 
  #amax = np.amax(scan)
  #rescaled_scan = scan / amax if amax > 0.0 else scan
  X_train.append(expand_np_image_to_3_channels(flipped_scan_horizontal))
  X_train.append(expand_np_image_to_3_channels(flipped_scan_vertical))
  X_train.append(expand_np_image_to_3_channels(rescaled))
  X_train.append(expand_np_image_to_3_channels(scan))
  y_train.append(1.0)
  y_train.append(1.0)
  y_train.append(1.0)
  y_train.append(1.0)

for axi_scan_control in data_path.glob('nii/controls/train/*/axi/PET*'): 
  scan = np.load(axi_scan_control)
  flipped_scan_horizontal = np.flip(scan,axis=1)
  flipped_scan_vertical = np.flip(scan,axis=0)
  rescaled = scan * 1.1
  #rescaled_scan = ((scan / np.amax(scan))*255).astype(np.uint8) #If we want to squezze values between 0-1 
  #amax = np.amax(scan)
  #rescaled_scan = scan / amax if amax > 0.0 else scan
  X_train.append(expand_np_image_to_3_channels(flipped_scan_horizontal))
  X_train.append(expand_np_image_to_3_channels(flipped_scan_vertical))
  X_train.append(expand_np_image_to_3_channels(rescaled))
  X_train.append(expand_np_image_to_3_channels(scan))
  y_train.append(0.0)
  y_train.append(0.0)
  y_train.append(0.0)
  y_train.append(0.0)



print('number of X observations: ', len(X_train))
print('number of y observations: ', len(y_train))


# In[327]:


print('fraction of sick patients: ', np.count_nonzero(y_train)/len(y_train))



#VALIDATION

for axi_scan_anony in data_path.glob('nii/anonymized/val/*/axi/PET*'): 
  scan = np.load(axi_scan_anony)
  #rescaled_scan = ((scan / np.amax(scan))*255).astype(np.uint8) #If we want to squezze values between 0-1 
  #amax = np.amax(scan)
  #rescaled_scan = scan / amax if amax > 0.0 else scan
  X_valid.append(expand_np_image_to_3_channels(scan))
  y_valid.append(1.0)

for axi_scan_control in data_path.glob('nii/controls/val/*/axi/PET*'):
  scan = np.load(axi_scan_control)
  #rescaled_scan = ((scan / np.amax(scan))*255).astype(np.uint8)
  #amax = np.amax(scan)
  #rescaled_scan = scan / amax if amax > 0.0 else scan
  X_valid.append(expand_np_image_to_3_channels(scan))
  y_valid.append(0.0)


print('number of X observations: ', len(X_valid))
print('number of y observations: ', len(y_valid))


# In[327]:


print('fraction of sick patients: ', np.count_nonzero(y_valid)/len(y_valid))




#test

for axi_scan_anony in data_path.glob('nii/anonymized/test/*/axi/PET*'):
  scan = np.load(axi_scan_anony)
  #rescaled_scan = ((scan / np.amax(scan))*255).astype(np.uint8) #If we want to squezze values between 0-1 
  #amax = np.amax(scan)
  #rescaled_scan = scan / amax if amax > 0.0 else scan
  X_test.append(expand_np_image_to_3_channels(scan))
  y_test.append(1.0)

for axi_scan_control in data_path.glob('nii/controls/test/*/axi/PET*'):
  scan = np.load(axi_scan_control)
  #rescaled_scan = ((scan / np.amax(scan))*255).astype(np.uint8)
  #amax = np.amax(scan)
  #rescaled_scan = scan / amax if amax > 0.0 else scan
  X_test.append(expand_np_image_to_3_channels(scan))
  y_test.append(0.0)


print('number of X observations: ', len(X_test))
print('number of y observations: ', len(y_test))


# In[327]:


print('fraction of sick patients: ', np.count_nonzero(y_test)/len(y_test))

#Empty cache
features = None
labels = None
label = None
training_data = None 


# In[331]:


print(f"Number of training samples: {len(X_train)}")
print(f"Number of validation samples: {len(X_valid)}")
print(f"Number of testing samples: {len(X_test)}")


# In[332]:


#composite transformations
transform = transforms.Compose([transforms.Resize(512)]) # is an option because image net are 256x256

#datasets
train_dataset = BrainTumorDataset(X_train, y_train, transform = transform)
validation_dataset = BrainTumorDataset(X_valid, y_valid, transform = transform)
test_dataset = BrainTumorDataset(X_test, y_test, transform = transform)

#empty cache
X_train = None
y_train = None
X_valid = None
y_valid = None
X_test = None
y_test = None

# In[334]:


train_gen = DataLoader(train_dataset, num_workers=0, shuffle=True, batch_size=batch_size)
valid_gen = DataLoader(validation_dataset, num_workers=0, batch_size=batch_size)
test_gen = DataLoader(test_dataset, num_workers=0)


# ## Define model and fit fully connected layer to our problem

# In[335]:


print('is gpu available? :', torch.cuda.is_available())
print('device name: ', torch.cuda.get_device_name())
print('device count: ', torch.cuda.device_count())


# In[ ]:


device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)
print(device)


# In[303]:


# instantiate transfer learning model
resnet_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# set all paramters as trainable
for param in resnet_model.parameters():
    param.requires_grad = True

# get input of fc layer
n_inputs = resnet_model.fc.in_features

# redefine fc layer / top layer/ head for our classification problem
resnet_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 1), #Only have two classes
                                nn.Sigmoid())

# set all paramters of the model as trainable
for name, child in resnet_model.named_children():
  for name2, params in child.named_parameters():
    params.requires_grad = True

# set model to run on GPU or CPU absed on availibility
resnet_model.to(device)

# print the trasnfer learning NN model's architecture
resnet_model
     


# ## Train model

# In[336]:


# loss function
# if GPU is available set loss function to use GPU
criterion = nn.BCELoss().to(device)

# optimizer
optimizer = torch.optim.SGD(resnet_model.parameters(), momentum=0.9, lr=3e-4) #learning rate from github repo 

# number of training iterations
epochs = 30

# empty lists to store losses and accuracies
train_losses = []
test_losses = []
train_correct = []
test_correct = []


# In[95]:


def save_checkpoint(state, is_best, filename=f'{model_name}_ckpt.pth.tar'):
    if is_best:
        torch.save(state, filename)


# In[337]:


# set training start time
start_time = time.time()

# set best_prec loss value as 2 for checkpoint threshold
best_accuracy = 0.0

# empty batch variables
b = None
train_b = None
test_b = None

# start training
for i in range(epochs):
    # empty training correct and test correct counter as 0 during every iteration
    trn_corr = 0
    tst_corr = 0
    
    #set loss vals
    total_loss_train = 0.0
    total_loss_val = 0.0

    
    # set epoch's starting time
    e_start = time.time()
    
    # train in batches
    for b, (X, y) in enumerate(train_gen):
        # set label as cuda if device is cuda
        X, y = X.to(device), y.to(device)

        # forward pass image sample         
        y_pred = resnet_model(X)
       
        #calculate loss and update total loss
        loss = criterion(y_pred, y.reshape(-1,1))
        total_loss_train += float(loss.item())

        # back propagate with loss
        loss.backward()
                
        #copy y_pred without gradients
        predicted = y_pred.data.clone()
        
        # if predicted label is correct as true label, calculate the sum for samples
        batch_correct = correct_number_of_classifications(torch.flatten(y), torch.flatten(predicted))
        
        # increment train correct with correcly predicted labels per batch
        trn_corr += batch_correct
        
        # perform optimizer step for this BATCH
        optimizer.step()

        # set optimizer gradients to zero
        optimizer.zero_grad()
        
    # set epoch's end time
    e_end = time.time()
    
    # print training metrics
    print(f'Epoch {(i+1)} Batch {(b+1)}\nAccuracy: {trn_corr*100/((b+1)*batch_size):2.2f} %  total  train loss: {total_loss_train/(b+1):2.4f}  Duration: {((e_end-e_start)/60):2.2f} minutes')

    # some metrics storage for visualization
    train_losses.append(total_loss_train/(b+1))
    train_correct.append(trn_corr*100/((b+1)*batch_size))

    #clear X and y
    X, y, b = None, None, None

    # validate using validation generator
    # do not perform any gradient updates while validation
    with torch.no_grad():
        for b, (X, y) in enumerate(valid_gen):
            # set label as cuda if device is cuda
            X, y = X.to(device), y.to(device)
            
            # forward pass image
            y_val = resnet_model(X)

            # get loss of validation set
            loss = criterion(y_val, y.reshape(-1,1))

            #update total loss
            total_loss_val  += loss.item()
            
            # if predicted label is correct as true label, calculate the sum for samples
            batch_correct = correct_number_of_classifications(torch.flatten(y), torch.flatten(y_val))
        
            # increment train correct with correcly predicted labels per batch
            tst_corr += batch_correct
            
    print(f'Validation Accuracy {tst_corr*100/((b+1)*batch_size):2.2f} Validation Loss: {total_loss_val/(b+1):2.4f}\n')

    # if current validation loss is less than previous iteration's validatin loss create and save a checkpoint
    is_best = tst_corr*100/((b+1)*batch_size) > best_accuracy
    best_accuracy = max(tst_corr*100/((b+1)*batch_size), best_accuracy)
    save_checkpoint({
            'epoch': i + 1,
            'state_dict': resnet_model.state_dict(),
            'best_prec1': best_accuracy,
        }, is_best)

    # some metrics storage for visualization
    test_losses.append(total_loss_val/(b+1))
    test_correct.append(tst_corr*100/((b+1)*batch_size))

# set models end time
end_time = time.time() - start_time    

# print models summary
print("\nTraining Duration {:.2f} minutes".format(end_time/60))
print("GPU memory used : {} kb".format(torch.cuda.memory_allocated()))
print("GPU memory cached : {}  kb".format(torch.cuda.memory_cached()))
     


# In[307]:


torch.cuda.empty_cache()


# In[ ]:


torch.save(resnet_model.state_dict(), f'{model_name}_model.pt')


# In[338]:

plt.figure('loss')
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.title('Loss Metrics')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.savefig(f'loss_{model_name}.png')
     


# In[339]:

plt.figure('accuracy')
plt.plot(train_correct, label='Training accuracy')
plt.plot(test_correct, label='Validation accuracy')
plt.title('Accuracy Metrics')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.savefig(f'accuracy_{model_name}.png')
     


# ## Time for test

# In[344]:


resnet_model.load_state_dict(torch.load(f'{model_name}_model.pt'))
train_gen = None
valid_gen = None
train_set = None
valid_set = None


# In[345]:


# set model to evaluation mode
resnet_model.eval()

# perform no gradient updates
with torch.no_grad():
    # some metrics storage for visualization and analysis
    total_test_loss = 0.0
    test_correct = 0
    labels = []
    pred = []
    # perform test set evaluation batch wise
    for b, (X, y) in enumerate(test_gen):
        # set label to use CUDA if available
        X, y = X.to(device), y.to(device)

        # append original labels
        labels.append(y.data)

        # perform forward pass
        y_val = resnet_model(X)

        # calculate loss
        loss = criterion(y_val, y.reshape(-1,1))

        #update total loss
        total_test_loss  += loss.item()

        # if predicted label is correct as true label, calculate the sum for samples
        batch_correct, predictions = correct_number_of_classifications(torch.flatten(y), torch.flatten(y_val), return_y_pred = True) 
        
        # increment train correct with correcly predicted labels per batch
        test_correct += batch_correct

        # append predicted label
        pred.append(predictions.data)

print(f"Test Loss: {total_test_loss/(b+1):.4f}")

print(f'Test accuracy: {test_correct*100/(b+1):.2f}%')


# In[346]:


labels = torch.stack(labels)
pred = torch.stack(pred)


# In[347]:


print(f"Clasification Report\n\n{classification_report(pred.view(-1).cpu(), labels.view(-1).cpu())}")

