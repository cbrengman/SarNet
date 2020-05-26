# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 13:39:07 2019

@author: cbrengman
"""

# Import needed packages
import os
from math import ceil, sqrt, floor
from PIL import Image
import cv2
import numpy as np

class Tile(object):
    """Represents a single tile."""

    def __init__(self, image, number, position, coords, filename=None):
        self.image = image
        self.number = number
        self.position = position
        self.coords = coords
        self.filename = filename

    @property
    def row(self):
        return self.position[0]

    @property
    def column(self):
        return self.position[1]

    @property
    def basename(self):
        """Strip path and extension. Return base filename."""
        return get_basename(self.filename)

    def generate_filename(self, directory=os.getcwd(), prefix='tile',
                          format='png', path=True):
        """Construct and return a filename for this tile."""
        filename = prefix + '_{col:02d}_{row:02d}.{ext}'.format(
                      col=self.column, row=self.row, ext=format.lower().replace('jpeg', 'jpg'))
        if not path:
            return filename
        return os.path.join(directory, filename)

    def save(self, filename=None, format='png'):
        if not filename:
            filename = self.generate_filename(format=format)
        self.image.save(filename, format)
        self.filename = filename

    def __repr__(self):
        """Show tile number, and if saved to disk, filename."""
        if self.filename:
            return '<Tile #{} - {}>'.format(self.number,
                                            os.path.basename(self.filename))
        return '<Tile #{}>'.format(self.number)

def get_basename(filename):
    """Strip path and extension. Return basename."""
    return os.path.splitext(os.path.basename(filename))[0]


def calc_pad(im_size,size):
    """Calculate the padding to add to each image to make it divisible to size"""
    hpad = 224-(im_size[0] % size[0])
    wpad = 224-(im_size[1] % size[1])
    return (hpad,wpad)

def pad_img(im,pad_size):
    """Pad image to make it divisible by size"""
    pad_im = Image.new("L",(pad_size[0]+im.size[0],pad_size[1]+im.size[1]))
    pad_im.paste(im,(0,0))
    return pad_im
    
def calc_rc_split(im,size=None):
    """calculate resultant number of rows/columns"""
    ncol = im.size[0] / size[0]
    nrow = im.size[1] / size[1]
    return ncol, nrow

def calc_rc_join(n):
    """Calculate initial number of rows/columns"""
    ncol = int(ceil(sqrt(n)))
    nrow = int(ceil(n/float(ncol)))
    return (ncol,nrow)

def get_combined_size(tiles):
    """Calculate combined size of tiles."""
    columns, rows = calc_rc_join(len(tiles))
    tile_size = tiles[0].image.size
    return (tile_size[0] * columns, tile_size[1] * rows)

def join_image(tiles,size=(224,224)):
    """
    @param ``tiles`` - Tuple of ``Image`` instances.
    @return ``Image`` instance.
    """
    image = Image.new('L', get_combined_size(tiles), None)
    score = Image.new('L', get_combined_size(tiles), None)
    columns, rows = calc_rc_join(len(tiles))
    for tile in tiles:
        image.paste(tile.image, tile.coords)
        score.paste(Image.new('L',size,int(tile.score*255)),tile.coords)
        
    return image,score

def join_image_heat(tiles,size=(224,224)):
    """
    @param ``tiles`` - Tuple of ``Image`` instances.
    @return ``Image`` instance.
    """
    image = Image.new('L', get_combined_size(tiles), None)
    score = Image.new('L',get_combined_size(tiles),None)
    heat = Image.new('L', get_combined_size(tiles), None)
    columns, rows = calc_rc_join(len(tiles))
    for tile in tiles:
        image.paste(tile.image, tile.coords)
        heat.paste(Image.fromarray(tile.cam,'L'),tile.coords)
        score.paste(Image.new('L',size,int(tile.score*255)),tile.coords)
        
    return image,score,heat


def slice_image(im,size=(224,224),save=False):
    """
    Split an image into N smaller images of size (tuple)
    
    Args:
        filename (str): Filename of the image to split
        size (tuple): the size of the smaller images
        
    Kwargs:
        save (bool): save the new files or not
        
    returns:
        tuple of :class:`tile` instances
    """
    
    im_w,im_h = im.size
    
    columns = 0
    rows = 0
    
    pad_size = calc_pad(im.size,size)
    im = pad_img(im,pad_size)
    columns,rows = calc_rc_split(im,size)
    
    tiles = []
    number = 1
    for pos_y in range(0,im_h,size[1]):
        for pos_x in range(0,im_w,size[0]):
            area = (pos_x,pos_y,pos_x + size[0],pos_y + size[1])
            image = im.crop(area)
            position = (int(floor(pos_x / size[0])) + 1, int(floor(pos_y / size[1])) + 1)
            coords = (pos_x,pos_y)
            tile = Tile(image,number,position,coords)
            tiles.append(tile)
            number +=1
    #if save:
    #    save_tiles(tiles,prefix=get_basename(filename),directory = os.path.dirname(filename))
        
    return tuple(tiles)


def save_tiles(tiles, prefix='', directory=os.getcwd(), format='png'):
    """
    Write image files to disk. Create specified folder(s) if they
       don't exist. Return list of :class:`Tile` instance.
    Args:
       tiles (list):  List, tuple or set of :class:`Tile` objects to save.
       prefix (str):  Filename prefix of saved tiles.
    Kwargs:
       directory (str):  Directory to save tiles. Created if non-existant.
    Returns:
        Tuple of :class:`Tile` instances.
    """
    # TODO: insert command to remove dir if present and redo if not
    for tile in tiles:
        tile.save(filename=tile.generate_filename(prefix=prefix,
                                                  directory=directory,
                                                  format=format), 
                                                  format=format)
    return tuple(tiles)

if __name__ == "__main__":

    out =  slice_image('test.png',size=(224,224))