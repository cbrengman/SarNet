#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:07:32 2020

@author: cbrengman
"""

import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp, resize

fold = 'Transfer_Data/'
filel = ['val/data/','val/noise/','train/data/','train/noise/']

for folder in filel:
    fileloc = fold+folder
    filenames = [f for f in os.listdir(fileloc) if os.path.isfile(os.path.join(fileloc,f))]

    for image in filenames:
        img = io.imread(fileloc + image)
        #if img.shape != (224,224,3):
        #    img = resize(img,(224,224,3))
        rot_angles = []
        for k in range(0,360):
            if k % 30 == 0:
                rot_angles.append(k)
        flipped = []
        flipped.append(img)
        flipped.append(np.fliplr(img))
        flipped.append(np.flipud(img))
        flipped.append(np.flipud(np.fliplr(img)))
        
        rotated = []
        for tmpimage in flipped:
            for angle in rot_angles:
                rotated.append(rotate(tmpimage,angle=angle,mode='constant'))
                
        transform1 = AffineTransform(translation=(25,25))
        transform2 = AffineTransform(translation=(-25,-25))
        #transform3 = AffineTransform(translation=(-50,-50))
        #transform4 = AffineTransform(translation=(50,50))
        shifted = []
        for tmpimage in rotated:
            shifted.append(warp(tmpimage,transform1,mode='constant'))
            shifted.append(warp(tmpimage,transform2,mode='constant'))
            #shifted.append(warp(tmpimage,transform3,mode='constant'))
            #shifted.append(warp(tmpimage,transform4,mode='constant'))
        
        for tmpimage in shifted:
            rotated.append(tmpimage)
                
        for i,outimg in enumerate(rotated):
            io.imsave('Transfer_Data/augmented/' + folder + image[:-4] + '_' + str(i) + '.png', outimg)
        
    
    