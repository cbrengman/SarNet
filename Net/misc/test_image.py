# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:44:22 2019

@author: cbrengman
"""

# Import needed packages
import torch
from torchvision.transforms import transforms
from torch.autograd import Variable
from models.CNN.sarnet import sarnet1 as net
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from misc.slice_join_image import slice_image, join_image


checkpoint = torch.load("data_wrap_noise_final.model")
model = net(pretrained=True)
model.load_state_dict(checkpoint)
model.eval()

def predict_image(img):
    print("Prediction in progress")
    
    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])

    # Preprocess the image
    image_tensor = transformation(img).float()

    # Add an extra batch dimension since pytorch treats all images as batches
    image_tensor = image_tensor.unsqueeze_(0)

    # Turn the input into a Variable
    image = Variable(image_tensor)

    # Predict the class of the image
    output = model(image)

    index = output.data.numpy().argmax()

    return index

def predict_images(img):
    print("Prediction in progress")
    
    # Define transformations for the image, should (note that imagenet models are trained with image size 224)
    transformation = transforms.Compose([transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    
    index = []
    # Preprocess the image
    for im in img:
        image_tensor = transformation(im.image).float().unsqueeze_(0)

        # Turn the input into a Variable
        image = Variable(image_tensor)

        # Predict the class of the image
        output = model(image)

        index.append(output.data.numpy().argmax())
        
        #fig,ax = plt.subplots()
        #ax.imshow(np.asarray(im.image),cmap='gray')
        #if output.data.numpy().argmax()==0:
        #    ax.set_title("Class = Noise")
        #else:
        #    ax.set_title("Class = Data")

    return index

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
    
    if op == "downsample":
        img = Image.open(filename).convert('L')
        img = img.resize(size,Image.ANTIALIAS)
    elif op == "slice":
        img = slice_image(filename,size=size)
    else:
        raise Exception("Invalid option '{}'. Valid options are 'downsample' or 'slice'.".format(op))
    
    return img
    

if __name__ == "__main__":

    #imagefile = "../synth_data/tiff_files/dclean/noise/nimage_101.tif"
    filename = 't1.png'
    size = (224,224)

    # load image
    img = load_image(filename,size=size,op="downsample")
    
    #check if one or many images
    if hasattr(img,'__len__'):
        index = predict_images(img)
        for im,val in zip(img,index):
            im.score = val
        image,score = join_image(img)
        fig,ax = plt.subplots()
        ax.imshow(image,cmap='gray')
        sc = ax.imshow(score,alpha = 0.25,cmap='jet_r')
        ax.set_xticks(np.arange(0,image.size[0]+1,size[0]))
        ax.set_yticks(np.arange(0,image.size[1]+1,size[1]))
        ax.grid(color='k',linestyle='-',linewidth=2)
        cbar = fig.colorbar(sc,ticks=[0,255])
        cbar.ax.set_yticklabels(['Data','Noise'])
    else:
        index = predict_image(img)
        fig,ax = plt.subplots()
        ax.imshow(np.asarray(img),cmap='gray')
        if index==0:
            ax.set_title("Class = Noise (0)")
        else:
            ax.set_title("Class = Data (1)")