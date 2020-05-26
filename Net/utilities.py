#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:43:56 2019

@author: cbrengman
"""

import cv2
import numpy as np
from PIL import Image
import tifffile as tiff
from misc.slice_join_image import slice_image



def load_image(filename,size=(224,224),op='downsample'):
    """
    Load an Image and
    Split an image into N smaller images of size (tuple)
    or downsample to size
    
    Args:
        filename (str): Filename of the image to split/downsample
        size (tuple): the size of the smaller images
        
    Kwargs:
        op (str): operation (downsample or split)
        
    returns:
        tuple of :class:`tile` instances
    """
    
    try:
        img = Image.open(filename).convert('L')
    except:
        print("PIL can't open image. Trying tiff.imread")
        try:
            img = tiff.imread(filename)
            if len(img) >= 1:
                img[0] = cv2.normalize(img[0],None,alpha=0,beta=255,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
                img[1] = cv2.normalize(img[1],None,alpha=0,beta=255,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
                img1 = Image.fromarray(img[0])
                img2 = Image.fromarray(img[1])
                img1 = img1.convert('L')
                img2 = img2.convert('L')
                img3 = Image.new('L',img1.size)
                img = Image.merge('RGB',[img1,img2,img3])
            else:
                img = cv2.normalize(img,None,alpha=0,beta=255,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
                img = Image.fromarray(img,'L')
            print('Image file loaded correctly')
        except:
            print('Cannot Load Image File')   
    if op == "downsample":
        img = img.resize(size,Image.ANTIALIAS)
    elif op == "slice":
        img = slice_image(img,size=size)
    else:
        raise Exception("Invalid option '{}'. Valid options are 'downsample' or 'slice'.".format(op))
    
    return img