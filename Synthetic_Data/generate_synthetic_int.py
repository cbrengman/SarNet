#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 15:35:07 2019

@author: Glarus
"""

from genrand_synth_displacements_smol import get_fault_parameters
from generate_synthetic_noise import gen_noise_corr_dem
import numpy as np
import cv2
from PIL import Image
import multiprocessing as mp
import matplotlib.pyplot as plt

def get_data_noise():
    data = get_fault_parameters()
    n1,n2,n3 = gen_noise_corr_dem()
    return data,n1,n2,n3

def wrap_images(img):
    img_wrapped = (((img - img.min()) * 4 * np.pi / 0.0555) % (2 * np.pi)) / 2 / np.pi
    return img_wrapped

def comb_data_noise(data,n1,n2,n3):
    wdata = wrap_images(data)
    wn1   = wrap_images(n1)
    wn2   = wrap_images(n2)
    wn3   = wrap_images(n3)
    
    wndata = (wdata+wn1)-wn2
    wndata = wndata - wndata.mean()
    wndata = wndata / max(abs(wndata.min()),abs(wndata.max()))
    
    ndata = (data+n1)-n2
    noise = n3
    nwdata = wrap_images(ndata) #(wdata+wn1)-wn2
    wnoise = wn3
    return ndata,noise,nwdata,wnoise

def make_greyscale(img):
    img = cv2.normalize(img,None,alpha=0,beta=255,norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
    img = Image.fromarray(img,'L')
    return img
    
def save_images(x,loc,ndatag,noiseg,ndata_wrapg,noise_wrapg,datag,data_wrapg):
    ndatag.save("data_smol/" + loc + "/ndata/data/sndata_image" + str(x) + ".tif")
    Image.Image.close(ndatag)
    noiseg.save("data_smol/" + loc + "/ndata/noise/snoise_image" + str(x) + ".tif")
    Image.Image.close(noiseg)
    ndata_wrapg.save("data_smol/" + loc + "/ndata_wrap/data/sndata_wrap_image" + str(x) + ".tif")
    Image.Image.close(ndata_wrapg)
    noise_wrapg.save("data_smol/" + loc + "/ndata_wrap/noise/snoise_wrap_image" + str(x) + ".tif")
    Image.Image.close(noise_wrapg)
    #datag.save("data_big/" + loc + "/ndata/orig/orig_image" + str(x) + ".tif")
    #Image.Image.close(datag)
    #data_wrapg.save("data_big/" + loc + "/ndata_wrap/orig/orig_image_wrap" + str(x) + ".tif")
    #Image.Image.close(data_wrapg)
    
def main_loop(x,loc='train'):
    data,n1,n2,n3 = get_data_noise()
    ndata,noise,ndata_wrap,noise_wrap = comb_data_noise(data,n1,n2,n3)
    ndatag = make_greyscale(ndata)
    noiseg = make_greyscale(noise)
    datag = make_greyscale(data)
    ndatag.save('ndata.tif')
    noiseg.save('noise.tif')
    datag.save('data.tif')
    
    data_wrapg = make_greyscale(wrap_images(data))
    ndata_wrapg = make_greyscale(ndata_wrap)
    noise_wrapg = make_greyscale(noise_wrap)
    save_images(x,loc,ndatag,noiseg,ndata_wrapg,noise_wrapg,datag,data_wrapg)
    

if __name__ == "__main__":
    with mp.Pool(mp.cpu_count()) as pool:
        for proc in range(100000):
            pool.starmap(main_loop,[(proc,'train')])
    pool.close()
    
    
    # for k in range(20):
    #     proc = [mp.Process(target=main_loop,args=(x,'train')) for x in range(k,(k+1))]
    #     for p in proc:
    #         p.start()
    #     for p in proc:
    #         p.terminate()
    #     for p in proc:
    #         p.join()
            
    
#    data,n1,n2,n3 = get_data_noise()
#    ndata,noise,ndata_wrap,noise_wrap = comb_data_noise(data,n1,n2,n3)
#    datag  = make_greyscale(data)
#    datag_wrap = make_greyscale(wrap_images(data))
#    ndatag = make_greyscale(ndata)
#    noiseg = make_greyscale(noise)
#    ndata_wrapg = make_greyscale(ndata_wrap)
#    noise_wrapg = make_greyscale(noise_wrap)
#    fig,ax = plt.subplots(3,2)
#    im1 = ax[0][0].imshow(datag)
#    im2 = ax[0][1].imshow(datag_wrap)
#    im3 = ax[1][0].imshow(ndatag)#,cmap='jet')
#    im4 = ax[1][1].imshow(ndata_wrapg)#,cmap='jet')
#    im5 = ax[2][0].imshow(noiseg)#,cmap='jet')
#    im6 = ax[2][1].imshow(noise_wrapg)#,cmap='jet')
#    fig.colorbar(im1,ax=ax[0][0])
#    fig.colorbar(im2,ax=ax[0][1])
#    fig.colorbar(im3,ax=ax[1][0])
#    fig.colorbar(im4,ax=ax[1][1])
#    fig.colorbar(im5,ax=ax[2][0])
#    fig.colorbar(im6,ax=ax[2][1])
        


