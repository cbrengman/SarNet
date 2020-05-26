# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 11:26:18 2019

@author: cbrengman
"""

from skimage.transform import resize
import scipy.ndimage as snd
import numpy as np
import h5py
import os
import cv2
from PIL import Image

def gen_noise_corr_dem():
    np.random.seed()
    noise,noise1,dem1,dem2 = set_noise_params()
    dem_scaled1  = np.random.random(1)*(dem1/np.max([dem1.max(),np.abs(dem1.min())]))
    dem_scaled2  = np.random.random(1)*(dem1/np.max([dem1.max(),np.abs(dem1.min())]))
    dem_scaled3  = np.random.random(1)*(dem2/np.max([dem2.max(),np.abs(dem2.min())]))
    noisy_dem1   = dem_scaled1*noise
    noisy_dem2   = dem_scaled2*noise1
    noisy_dem3   = dem_scaled3*noise
    return noisy_dem1,noisy_dem2,noisy_dem3

def set_noise_params():
    ###############################################################################
    #Define Variables for noise
    xdim      = 10                      #Size of the x dimension of grid in arcseconds
    ydim      = 10                      #Size of the y dimension of grid in arcseconds
    fxdim     = 224                      #Size of the x dimension of grid in arcseconds
    fydim     = 224                      #Size of the y dimension of grid in arcseconds
    X         = np.arange(1,xdim,1)     #An array centered on 0 and evenly spaced
    Y         = np.arange(1,ydim,1)
    
    Xn1, Yn1  = np.meshgrid(X,Y)        #gridded data the size of the image
    Xn        = Xn1.ravel(order='F') #reshape from matrix to r*c by 1 length arrays
    Yn        = Yn1.ravel(order='F') #reshape from matrix to r*c by 1 length arrays
    
    nparams = {
            "xshape": Xn1,
            "x": Xn,
            "y": Yn,
            "Var": 0.01,
            "noisescale": 20,
            "numdatas": 1
            }
    
    ###############################################################################
    #Make DEM
    get_dem = True
    files = [f for f in os.listdir('../DEM/dem_h5/') if not f.startswith('.')] #get list of possible dem's
    while get_dem:
        f1 = np.random.randint(0,len(files))
        f2 = np.random.randint(0,len(files))
        with h5py.File('../DEM/dem_h5/'+files[f1],'r') as dem1, h5py.File('../DEM/dem_h5/'+files[f2],'r') as dem2:
        #dem1 = h5py.File('../DEM/dem_h5/'+files[f1],'r') #read dem file
        #dem2 = h5py.File('../DEM/dem_h5/'+files[f2],'r') #read dem file
            key1 = list(dem1.keys())[0] #get parts
            key2 = list(dem2.keys())[0] #get parts
            dem1 = np.array(dem1[key1]) #extract dem
            dem2 = np.array(dem2[key2]) #extract dem
            dem1diff = np.max([np.abs(np.min(dem1)),np.max(dem1)]) - np.mean(dem1)
            dem2diff = np.max([np.abs(np.min(dem2)),np.max(dem2)]) - np.mean(dem2)
            if dem1diff < 0.01 and dem2diff < 0.01:
                get_dem = False
    
    dem1 = resize(dem1,(fxdim,fydim),mode='constant',anti_aliasing=True) #resize to noise scale
    dem2 = resize(dem2,(fxdim,fydim),mode='constant',anti_aliasing=True) #resize to noise scale
    
    ###############################################################################
    #Make Noise
    noise = make_corr_noise(nparams)
    noise = snd.zoom(noise,24.9)
    noise1 = make_corr_noise(nparams)
    noise1 = snd.zoom(noise1,24.9)
    return noise,noise1,dem1,dem2

def make_corr_noise(nparams):
    x       = nparams['x']
    y       = nparams['y']
    var     = nparams['Var']
    ns      = nparams['noisescale']
    s    = x.size
    covd = np.diag(np.ones(s,)*var)
    for i in range(s):
        tmpx = x[(i+1):]
        tmpy = y[(i+1):]
        tmpsx = np.subtract(x[i],tmpx)
        tmpsy = np.subtract(y[i],tmpy)
        tmp = np.linalg.norm((tmpsx,tmpsy),axis=0)
        covd[i,(i+1):] = var*10**(-tmp/ns)
        covd[(i+1):,i] = covd[i,(i+1):]
    noise = corr_noise(covd,nparams["numdatas"])
    r,c   = nparams["xshape"].shape
    noise_reshape = noise.reshape(r,c)
    return noise_reshape

def corr_noise(covd,numdatas):
    npoints = len(covd)
    d,v     = np.linalg.eig(covd)
    d       = np.diag(d)
    orig    = np.random.randn(npoints,numdatas)
    orig    = orig.reshape(len(orig),1)
    noise   = np.matmul(np.matmul(v,np.sqrt(d)),orig)
    noise   = np.real(noise)
    return noise

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    n1,n2,d1,d2 = gen_noise_corr_dem()
    img = cv2.normalize(n1,None,alpha=0,beta=255,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
    img = Image.fromarray(img,'L')
    
     
    #plt.imshow(img,cmap='gray')
    fig,ax = plt.subplots()
    im = ax.imshow(img,cmap='jet')
    #fig.colorbar(im,ax=ax)
    #print(np.max(n1),np.max(n2),np.max(n3))
    #plt.colorbar()