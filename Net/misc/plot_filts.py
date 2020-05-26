# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:11:46 2019

@author: cbrengman
"""

#load saved model
#a will store each batch layer
#b = a[n] will extract specific layer n
#c = b.weight.data.numpy() will return that layers filters (nfilt,1,x,y)

import matplotlib.pyplot as plt
import torch.nn as nn
a = []
for m in model.modules():
    if isinstance(m,nn.Conv2d):
        a.append(m)
        
img2 = []
for filt in c3:
    #tmp = filt[0]
    #fig,ax = plt.subplots()
    #im = ax.pcolor(tmp,cmap='jet')
    #fig.colorbar(im,ax=ax)
    img2.append(filt[0])

figsize = [13.25,7.5]


import numpy as np
def show_images(images, rows = 1):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    """
    n_images = len(images)
    fig = plt.figure()
    for n, image in enumerate(images):
        a = fig.add_subplot(rows, np.ceil(n_images/float(rows)), n + 1)
        a.set_xticks([])
        a.set_yticks([])
        plt.imshow(image,cmap='jet')
    fig.set_size_inches(figsize)
    plt.tight_layout(True)
    plt.show()


