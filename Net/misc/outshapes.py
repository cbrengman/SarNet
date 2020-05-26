# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:28:18 2019

@author: cbrengman
"""

def conv(h_w, k=1, s=1, p=0, d=1):
    from math import floor
    if type(k) is not tuple:
        k = (k, k)
    h = floor( ((h_w[0] + (2 * p) - ( d * (k[0] - 1) ) - 1 )/ s) + 1)
    w = floor( ((h_w[1] + (2 * p) - ( d * (k[1] - 1) ) - 1 )/ s) + 1)
    return (h, w)

def pool(h_w, k=1, s=1, p=0, d=1):
    from math import floor
    if type(k) is not tuple:
        k = (k, k)
    h = floor( ((h_w[0] + (2 * p) - (d * (k[0] - 1)) - 1) / s ) + 1)
    w = floor( ((h_w[1] + (2 * p) - (d * (k[1] - 1)) - 1) / s ) + 1)
    return (h, w)