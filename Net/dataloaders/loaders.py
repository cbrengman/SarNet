# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 12:07:51 2019

@author: cbrengman
"""

import os
import sys
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.vision import VisionDataset

def ImageFolder(datadir):
    #Read data into a dataset which can be loaded into a pytorch dataloader
    #Data should be stored as an image format in folders based on class
    transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    image_data = datasets.ImageFolder(datadir, transform=transform)
    return image_data

class STdataset(Dataset):
    """
    A dataset for a folder of time dependent Interferograms in order from earliest to latest
    Expects a structure to be directory --> [train/test/val] --> label --> Image Sets. Where
    image sets are a single file of time dependent interferograms of format [x/y/z]. 
    
    Args:
        directory(str): parent directory where datasets are located
        mode (str,optional): which dataset is read. default: train
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.t
    """
    
    def __init__(self,directory,mode='train',transform=None):
        
        folder = Path(directory)/mode
        self.resize_height = 224
        self.resize_width = 224
        self.transform = transform
        
        #retrieve filenames
        self.fnames,labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder,label)):
                self.fnames.append(os.path.join(folder,label,fname))
                labels.append(label)
        
        #Map label names to indicies
        self.label2index = {label:index for index,label in enumerate(sorted(set(labels)))}
        #Convert labels to indicies
        self.label_array = np.array([self.label2index[label] for label in labels],dtype=int)
        
        
    def __getitem__(self,index):
        buffer = self.loadints(self.fnames[index])
        
        return buffer, self.label_array[index]
    
    def __len__(self):
        return len(self.fnames)
    
    def loadints(self,fname):
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 1), np.dtype('float32'))
        
        fcount = capture.get(cv2.CAP_PROP_POS_FRAMES)
        count = 0
        
        while True:
            flag, frame = capture.read()
            if flag:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = np.reshape(frame,(frame_width,frame_height,1))
                #Resize if not already correct size
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                        frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                if self.transform is not None:
                    frame = self.transform(frame)
                    frame = np.reshape(frame,(frame_width,frame_height,1))
                buffer[count] = frame
                count+=1
                fcount = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if fcount == capture.get(cv2.CAP_PROP_FRAME_COUNT):
                break
        
        capture.release()
        
        #convert to pytorch ordering 
        #e.g. [D,H,W,C] to [C,D,H,W]
        buffer = buffer.transpose((3,0,1,2))
        
        return buffer
    
# for 3DCNN
class Dataset_SarNet3D(Dataset):
    "Characterizes a dataset for PyTorch"
    def __init__(self, data_path, folders, labels, frames, transform=None):
        "Initialization"
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i))).convert('L')

            if use_transform is not None:
                image = use_transform(image)

            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        X = self.read_images(self.data_path, folder, self.transform).unsqueeze_(0)  # (input) spatial images
        y = torch.LongTensor([self.labels[index]])                             # (labels) LongTensor are for int64 instead of FloatTensor

        # print(X.shape)
        return X, y
    
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)

def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images
    
class MyDatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions=None, mode=None, transform=None, target_transform=None, is_valid_file=None):
        super(MyDatasetFolder, self).__init__(root)
        if mode==None or mode=='train':
            self.root = os.path.join(root,'train')
        elif mode=='test':
            self.root = os.path.join(root,'test')
        elif mode=='val':
            self.root = os.path.join(root,'val')
        self.transform = transform
        self.target_transform = target_transform
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
            
            Returns:tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
                
        return sample, target

    def __len__(self):
        return len(self.samples)
    
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

mode='train'
def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
    
def tiff_loader(path):
    import tifffile as tf
    return tf.imread(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        try:
            return pil_loader(path)
        except:
            try:
                return tiff_loader(path)
            except:
                print("Cannot load image using PIL or TIFFFILE")
    
class MyImageFolder(MyDatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid_file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root, mode='train', transform=None, target_transform=None,
                 loader=default_loader, is_valid_file=None):
        super(MyImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          mode = mode,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples