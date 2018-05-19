#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 00:24:10 2018

@author: rinzler
"""

from cv2 import warpAffine
import numpy as np

def shift(img, dx, dy):
    
    img = np.transpose(img, (1,2,0))
    r,c,_ = img.shape
#    print(r,c)
    trans = np.array([[1,0,dx],[0,1,dy]]).astype(np.float32)
    wimg = warpAffine(img, trans, (r,c)).reshape(r,c,1)
#    print(wimg.shape)
    rimg = np.transpose(wimg, (2,0,1))
#    print(rimg.shape)
    return rimg

def BatchShift(imbatch, dxdy = [-4,4]):
    dim = imbatch.shape
#    print(dim)
#    r = dim[2]
#    c = dim[3]
    
    
    R = np.random.randint(low=dxdy[0], high=dxdy[1], size=(dim[0],2))    
#    print(R.shape)
    for i in range(dim[0]):
        imbatch[i] = shift(imbatch[i], R[i][0], R[i][1])
    return imbatch, R
        