#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:51:34 2019

@author: Glarus
"""
from skimage.transform import resize
import numpy as np
import h5py
import os


def convert_dem(fname,size=3, outshape=(224,224)):
    if size==3:
        size=1201
    elif size==1:
        size=3601
    else:
        print("size arguement 1 for SRTM1 data and 3 for SRTM3 data")

    dem = np.fromfile('dem_hgts/' + fname,np.dtype('>i2'),size*size).reshape((size,size))

    dem = resize(dem, outshape) #resize to noise scale
    
    file = h5py.File('dem_h5/' + fname[:-4] + '.h5','w')
    file.create_dataset('dem',data=dem)
    file.close()

def file_check(fname):
    return os.path.isfile('dem_h5/' + fname[:-4] + '.h5')

if __name__ == "__main__":
    files = [f for f in os.listdir('dem_hgts/') if not f.startswith('.')] #get list of possible dem's 
    
    for file in files:
        if file_check(file):
            print("H5 file already exists, skipping")
        else:
            convert_dem(file)
