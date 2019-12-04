#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:25:28 2019

@author: hanxu8
"""

import numpy as np



def get_batch(Xi, Xv, y, batch_size, index):
    start = index * batch_size
    end = (index+1) * batch_size
    end = end if end < len(y) else len(y)
    return Xi[start:end], Xv[start:end], [[y_] for y_ in y[start:end]]
    
def shuffle_in_unison_scary( a, b, c):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    np.random.set_state(rng_state)
    np.random.shuffle(c)
