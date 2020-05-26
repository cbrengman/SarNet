#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:59:34 2019

@author: cbrengman
"""

import time
import torch
from tkinter import Tk
from torch import nn,optim
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from dataloaders.loaders import MyImageFolder
from tkinter.filedialog import askopenfilename
from models.CNN.DD.sarnet import sarnet1 as net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)   

batch_size = 25
n_epochs = 500

image_transforms = {
        'train':
        transforms.Compose([
                transforms.Grayscale(),
                #transforms.RandomRotation(degrees=15,fill=(0,)),
                #transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.5,],
                                     [0.5,])
                ]),
        'val':
        transforms.Compose([
                transforms.Grayscale(),
                #transforms.RandomRotation(degrees=15,fill=(0,)),
                #transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.5,],
                                     [0.5,])
        ]),
}
    
directory = "Transfer_Data/augmented/"
train_dataloader = DataLoader(MyImageFolder(directory,mode='train',transform=image_transforms['train']),batch_size=batch_size,shuffle=True)
val_dataloader = DataLoader(MyImageFolder(directory,mode='val',transform=image_transforms['val']),batch_size=batch_size*2,shuffle=True)
dataloaders = {'train': train_dataloader, 'val': val_dataloader}
dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

model = net()
optimizer = optim.SGD(model.parameters(),lr=0.3,momentum=0.9,nesterov=True)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
#Asks for filename and loads checkpoint model
root = Tk()
root.withdraw()
file = askopenfilename()
checkpoint = torch.load(file)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['opt_dict'])
#finalconv_name = 'layer4'

#freeze model weights

for i,param in enumerate(model.parameters()): 
    if i < 60:
        param.requires_grad = False
        # i < 60 --> trains only the last FC layer
        # i < 45 --> trains last FC layer and layer 4
        # i < 30 --> trains last FC layer, layer 4, and layer 3
        # i < 15 --> trains last FC layer, layer 4, layer 3, and layer 2
        # i < 03 --> trains last FC layer, layer 4, layer 3, layer 2, and layer 1
        
#Add on a classifier
#model.fc = nn.Linear(in_features=512,out_features=2)
#model.fc = nn.Sequential(nn.Linear(in_features=512,out_features=256),
#                         nn.Linear(in_features=256,out_features=2))

total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')

model.to(device)
criterion = nn.CrossEntropyLoss() 

n_epochs_stop = 50
epochs_no_improve = 0

since = time.time()
best_acc = 0
while epochs_no_improve < n_epochs_stop:
    for epoch in range(n_epochs):
        print('Epoch {}/{}'.format(epoch+1, n_epochs))
        print('-' * 10)
        for phase in ['train','val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for data, targets in dataloaders[phase]:
                data = data.to(device)
                targets = targets.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(data)
                    _,preds = torch.max(outputs,1)
                    loss = criterion(outputs,targets)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * targets.size(0)
                running_corrects += torch.sum(preds == targets.data)
                
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase] * 100
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'acc': running_loss,
                        'opt_dict': optimizer.state_dict(),
                        }, "test_e{}.model".format(epoch+1))        
                epochs_no_improve=0
            elif phase == 'val':
                epochs_no_improve+=1
                if epochs_no_improve >= n_epochs_stop:
                    print()
                    print("Process Terminated Early")
                    break
        if epochs_no_improve >= n_epochs_stop:
            break
        print()
    if epochs_no_improve >= n_epochs_stop:
        break
print()
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
    
        
#    for data, targets in dataloaders['train']:
#        out = model(data.to(device))
#        loss = criterion(out,targets.to(device))
#        loss.backward()
#        optimizer.step()
#    for data,targets in dataloaders['val']:
#        out=model(data.to(device))
#        loss=criterion(out,targets.to(device))
#        val_loss+=loss
#    val_loss = val_loss / len(dataloaders['train'])
#    if val_loss < min_val_loss:
#        torch.save({
#                    'epoch': epoch + 1,
#                    'state_dict': model.state_dict(),
#                    'acc': val_loss,
#                    'opt_dict': optimizer.state_dict(),
#                    }, "test_e{}.model".format(epoch+1))
#        epochs_no_improve = 0
##        minval_lss = val_loss
#    else:
#        epochs_no_improve+=1
#        if epochs_no_improve == n_epochs_stop:
#            print('Early stopping')
#            #model = torch.load('test.model')

