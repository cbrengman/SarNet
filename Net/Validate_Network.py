#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:38:48 2020

@author: cbrengman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 11:59:34 2019

@author: cbrengman
"""

import time
import torch
import numpy as np
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

batch_size = 2
n_epochs = 25

image_transforms = {
        'val':
        transforms.Compose([
                transforms.Grayscale(),
                #transforms.RandomRotation(degrees=15),
                #transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.5,],
                                     [0.5,])
        ]),
}
    
directory = "Transfer_Data/augmented/"
val_dataloader = DataLoader(MyImageFolder(directory,mode='val',transform=image_transforms['val']),batch_size=batch_size*2,shuffle=True)
dataloaders = {'val': val_dataloader}
dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['val']}

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

model.to(device)
criterion = nn.CrossEntropyLoss() 

model.eval()
running_loss = 0.0
running_corrects = 0
for data, targets in dataloaders['val']:
    data = data.to(device)
    targets = targets.to(device)
                
    outputs = model(data)
    _,preds = torch.max(outputs,1)
    loss = criterion(outputs,targets)
        
    running_loss += loss.item() * targets.size(0)
    running_corrects += torch.sum(preds == targets.data)
            
print('{} Loss: {:.4f} Acc: {:.4f}'.format('val', running_loss, running_corrects.double()/dataset_sizes['val']*100))


