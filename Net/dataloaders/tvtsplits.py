# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 10:33:18 2019

@author: cbrengman
"""

import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler

#Algorithm to load data and split it into training and test sets 
def Split75_20_05(image_data, bs):
    train_data = image_data
    test_data = image_data
    val_data = image_data
    num_train = len(train_data) #total number of data read in 
    indices = list(range(num_train)) #list of intergers for full dataset
    split1 = int(np.floor(0.75 * num_train)) #what image number to split the dataset at
    split2 = int(np.floor(0.95 * num_train))
    np.random.shuffle(indices) #randomizes image order
    
    #get indices for test and train data split
    train_idx, val_idx, test_idx = indices[:split1], indices[split1:split2], indices[split2:] 
    train_sampler = SubsetRandomSampler(train_idx) #make sampler for train indices
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx) #make sampler for test indices
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=bs) #Put train data in loader
    valloader = torch.utils.data.DataLoader(val_data, sampler=val_sampler, batch_size=bs) #Put test data in loader
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=bs) #Put train data in loader
    return trainloader, valloader,testloader

#Algorithm to load data and split it into training and test sets 
def Split80_15_05(image_data, bs):
    train_data = image_data
    test_data = image_data
    val_data = image_data
    num_train = len(train_data) #total number of data read in 
    indices = list(range(num_train)) #list of intergers for full dataset
    split1 = int(np.floor(0.80 * num_train)) #what image number to split the dataset at
    split2 = int(np.floor(0.95 * num_train))
    np.random.shuffle(indices) #randomizes image order
    
    #get indices for test and train data split
    train_idx, val_idx, test_idx = indices[:split1], indices[split1:split2], indices[split2:] 
    train_sampler = SubsetRandomSampler(train_idx) #make sampler for train indices
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx) #make sampler for test indices
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=bs) #Put train data in loader
    valloader = torch.utils.data.DataLoader(val_data, sampler=val_sampler, batch_size=bs) #Put test data in loader
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=bs) #Put train data in loader
    return trainloader, valloader,testloader

#Algorithm to load data and split it into training and test sets 
def Split80_10_10(image_data, bs):
    train_data = image_data
    test_data = image_data
    val_data = image_data
    num_train = len(train_data) #total number of data read in 
    indices = list(range(num_train)) #list of intergers for full dataset
    split1 = int(np.floor(0.80 * num_train)) #what image number to split the dataset at
    split2 = int(np.floor(0.90 * num_train))
    np.random.shuffle(indices) #randomizes image order
    
    #get indices for test and train data split
    train_idx, val_idx, test_idx = indices[:split1], indices[split1:split2], indices[split2:] 
    train_sampler = SubsetRandomSampler(train_idx) #make sampler for train indices
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx) #make sampler for test indices
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=bs) #Put train data in loader
    valloader = torch.utils.data.DataLoader(val_data, sampler=val_sampler, batch_size=bs) #Put test data in loader
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=bs) #Put train data in loader
    return trainloader, valloader,testloader

#Algorithm to load data and split it into training and test sets 
def Split70_20_10(image_data, bs):
    train_data = image_data
    test_data = image_data
    val_data = image_data
    num_train = len(train_data) #total number of data read in 
    indices = list(range(num_train)) #list of intergers for full dataset
    split1 = int(np.floor(0.70 * num_train)) #what image number to split the dataset at
    split2 = int(np.floor(0.90 * num_train))
    np.random.shuffle(indices) #randomizes image order
    
    #get indices for test and train data split
    train_idx, val_idx, test_idx = indices[:split1], indices[split1:split2], indices[split2:] 
    train_sampler = SubsetRandomSampler(train_idx) #make sampler for train indices
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx) #make sampler for test indices
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=bs) #Put train data in loader
    valloader = torch.utils.data.DataLoader(val_data, sampler=val_sampler, batch_size=bs) #Put test data in loader
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=bs) #Put train data in loader
    return trainloader, valloader,testloader

#Algorithm to load data and split it into training and test sets 
def Split75_15_10(image_data, bs):
    train_data = image_data
    test_data = image_data
    val_data = image_data
    num_train = len(train_data) #total number of data read in 
    indices = list(range(num_train)) #list of intergers for full dataset
    split1 = int(np.floor(0.75 * num_train)) #what image number to split the dataset at
    split2 = int(np.floor(0.90 * num_train))
    np.random.shuffle(indices) #randomizes image order
    
    #get indices for test and train data split
    train_idx, val_idx, test_idx = indices[:split1], indices[split1:split2], indices[split2:] 
    train_sampler = SubsetRandomSampler(train_idx) #make sampler for train indices
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx) #make sampler for test indices
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=bs) #Put train data in loader
    valloader = torch.utils.data.DataLoader(val_data, sampler=val_sampler, batch_size=bs) #Put test data in loader
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=bs) #Put train data in loader
    return trainloader, valloader,testloader

#Algorithm to load data and split it into training and test sets 
def Split80_20(image_data, bs):
    train_data = image_data
    test_data = image_data
    num_train = len(train_data) #total number of data read in 
    indices = list(range(num_train)) #list of intergers for full dataset
    split1 = int(np.floor(0.80 * num_train)) #what image number to split the dataset at
    np.random.shuffle(indices) #randomizes image order
    
    #get indices for test and train data split
    train_idx, test_idx = indices[:split1], indices[split1:] 
    train_sampler = SubsetRandomSampler(train_idx) #make sampler for train indices
    test_sampler = SubsetRandomSampler(test_idx) #make sampler for test indices
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=bs) #Put train data in loader
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=bs) #Put train data in loader
    return trainloader, testloader

#Algorithm to load data and split it into training and test sets 
def Split90_10(image_data, bs):
    train_data = image_data
    test_data = image_data
    num_train = len(train_data) #total number of data read in 
    indices = list(range(num_train)) #list of intergers for full dataset
    split1 = int(np.floor(0.90 * num_train)) #what image number to split the dataset at
    np.random.shuffle(indices) #randomizes image order
    
    #get indices for test and train data split
    train_idx, test_idx = indices[:split1], indices[split1:] 
    train_sampler = SubsetRandomSampler(train_idx) #make sampler for train indices
    test_sampler = SubsetRandomSampler(test_idx) #make sampler for test indices
    trainloader = torch.utils.data.DataLoader(train_data, sampler=train_sampler, batch_size=bs) #Put train data in loader
    testloader = torch.utils.data.DataLoader(test_data, sampler=test_sampler, batch_size=bs) #Put train data in loader
    return trainloader, testloader