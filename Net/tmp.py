# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:03:21 2019

@author: cbrengman
"""

import cv2
import torch 
from torch import optim
import numpy as np
from PIL import Image
from tkinter import Tk
import tifffile as tiff
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from torch.nn import functional as F
from models.CNN.DD.sarnet import sarnet1
from tkinter.filedialog import askopenfilename
from mpl_toolkits.axes_grid1 import make_axes_locatable
from misc.slice_join_image import slice_image, join_image_heat


model = sarnet1()
optimizer = optim.SGD(model.parameters(),lr=0.01)
#Asks for filename and loads checkpoint model
root = Tk()
root.withdraw()
file = askopenfilename()
checkpoint = torch.load(file)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['opt_dict'])
finalconv_name = 'layer4'
net = model
net.eval()

    
def gen_heat(image):

    # hook the feature extractor
    features_blobs = []
    def hook_feature(module, input, output):
        features_blobs.append(output.data.cpu().numpy())
        
    net._modules.get(finalconv_name).register_forward_hook(hook_feature)
    
    # get the softmax weight
    params = list(net.parameters())
    weight_softmax = np.squeeze(params[-2].data.numpy())
    
    def returnCAM(feature_conv, weight_softmax, class_idx):
        # generate the class activation maps upsample to 224x224
        size_upsample = (224, 224)
        bz, nc, h, w = feature_conv.shape
        output_cam = []
        output_cam_og = []
        for idx in class_idx:
            cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, size_upsample,interpolation=cv2.INTER_CUBIC))
            output_cam_og.append(cam_img)
        return output_cam,output_cam_og
    
    preprocess = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    
    img_tensor = preprocess(image)
    img_variable = Variable(img_tensor.unsqueeze(0))
    if img_variable.shape[1] > 2:
        img_variable = img_variable.transpose(1,2)
    logit = net(img_variable)
    
    classes = {0: 'data',1: 'noise'}
    
    h_x = F.softmax(logit, dim=1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()
    
    # output the prediction
    for i in range(2):
        print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))
    out = []
    out.append(classes[idx[0]])
    out.append(classes[idx[1]])
    
    # generate class activation mapping for the top1 prediction
    CAMs,OGCAMs = returnCAM(features_blobs[0], weight_softmax, idx)
    
    height,width = image.size
    #CAMs[0] = cv2.resize(CAMs[0],(width,height))
    #CAMs[1] = cv2.resize(CAMs[1],(width,height))
    return CAMs,OGCAMs,out
    




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


if __name__ == "__main__":
    
    #image_loc = '/data/not_backed_up/cbrengman/Research/Artificial_Neural_Networks/SarNet/synthetic_data/spatial/1chan_data/val/ndata/data/'
    #image_loc = '/data/not_backed_up/cbrengman/Research/Artificial_Neural_Networks/SarNet/synthetic_data/spatial/1chan_data/val/ndata_wrap/data/'
    #image_loc = '/data/not_backed_up/cbrengman/Research/Artificial_Neural_Networks/SarNet/synthetic_data/spatial/1chan_data_smol/val/ndata_wrap/data/'
    #image_loc = '/data/not_backed_up/cbrengman/Research/Artificial_Neural_Networks/SarNet/synthetic_data/spatial/1chan_data_smol/val/ndata/data/'
    #image_loc = '/home/cbrengman/Pictures/SS/'
    #image_loc = '/data/not_backed_up/cbrengman/Research/Artificial_Neural_Networks/eq_ints/'
    #image_loc = '/home/cbrengman/Documents/Travel/AGU_2019/Talk/Figures/'
    image_loc = '/data/not_backed_up/cbrengman/Research/Artificial_Neural_Networks/SarNet_Final/Synthetic_Data/'
    #image_loc = '/data/not_backed_up/cbrengman/Research/Artificial_Neural_Networks/SarNet/synthetic_data/spatial/3chan_data/ndata_wrap/val/data/'
    #image_loc = '/data/not_backed_up/cbrengman/Research/Artificial_Neural_Networks/SarNet/synthetic_data/spatial/temp/'
    #image_loc = '/data/not_backed_up/cbrengman/Research/Artificial_Neural_Networks/SarNet/network/testfigs/'
    mode = 'downsample'  #downsample or slice
    #fname = 'ndata_wrap_image0.tif'
    #fname = 'filt_topophase.unw.geo.png'
    #fname = 'real_CAM_input.tif'
    fname = 'ndata.tif'
    fname1 = 'data.tif'
    fname2 = 'noise.tif'
    #fname = 'ndata_wrap_image155.tif'
    filename = image_loc + fname
    filename1 = image_loc + fname1
    filename2 = image_loc + fname2
    #filename2 = image_loc1 + fname2
    size = (224,224)
    img = load_image(filename,size,mode)
    img1 = load_image(filename1,size,mode)
    img2 = load_image(filename2,size,mode)
    #img2 = load_image(filename2,size,mode)
    
    if len(img.getbands()) > 1:
        img1 = np.array((np.array(img.getchannel(0)),np.array(img.getchannel(1))))
        img = img1
    CAMs,OGCAMs,classification = gen_heat(img)

    out = CAMs[0]
    out1 = CAMs[1]
    ndata = np.asarray(img,dtype='float64')-125
    data = np.asarray(img1,dtype='float64')-125
    noise = np.asarray(img2,dtype='float64')-125
    out = out.astype('float64')
    out *= 1.0/out.max()
    out1 = out1.astype('float64')
    out1 *= 1.0/out1.max()
    
    noise_norm = ndata*out1
    noise_res = noise_norm-noise
    ndata_norm = ndata*out
    ndata_res = ndata_norm-data
    
    fig,ax = plt.subplots(1,6)
    im = ax[0].imshow(data,cmap='jet')
    ax[0].title.set_text('data')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im,cax=cax)
    im1 = ax[1].imshow(noise,cmap='jet')
    ax[1].title.set_text('noise')
    ax[1].set_yticklabels('')
    divider = make_axes_locatable(ax[1])
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im1,cax=cax1)
    im2 = ax[2].imshow(ndata_norm,cmap='jet')
    ax[2].title.set_text('ndata_norm')
    ax[2].set_yticklabels('')
    divider = make_axes_locatable(ax[2])
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im2,cax=cax2)
    im3 = ax[3].imshow(noise_norm,cmap='jet')
    ax[3].title.set_text('noise_norm')
    ax[3].set_yticklabels('')
    divider = make_axes_locatable(ax[3])
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im3,cax=cax3)
    im4 = ax[4].imshow(ndata_res,cmap='jet')
    ax[4].title.set_text('ndata_res')
    ax[4].set_yticklabels('')
    divider = make_axes_locatable(ax[4])
    cax4 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im4,cax=cax4)
    im5 = ax[5].imshow(noise_res,cmap='jet')
    ax[5].title.set_text('noise_res')
    ax[5].set_yticklabels('')
    divider = make_axes_locatable(ax[5])
    cax5 = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im5,cax=cax5)
