#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:38:53 2019

@author: hanxu8
"""

# coding: utf-8
import numpy as np

def decode(ins=None, field_size=21):
    #print("Loading data...")
    if len(ins) == "":
        raise ValueError("data is empty")
    xi_train = []
    xv_train = []
    y_train = []
    with open(ins,'r') as f:
        for line in f:
            #
            # 默认列数固定，否则报错
            # 0   1603 1708 1967 1700 2440 2639 2346 1240 830 932 951 964 606 2020 1100 2292 804 1695 1207 522 1770
            # 1   2040 1440 1542 345 2707 2639 517 1794 830 932 1636 964 606 2159 1100 2435 1480 1425 1760 939 1631
            #
            lis = line.strip('\t\n').split('\t')
            label= lis[0].split(' ')[0]
            #label= [int(item) for item in label]
            con  = lis[1].split(' ')
            con= [int(item) for item in con]
            value = [float(1) for i in range(field_size)]
            if len(con)!=field_size:
                con = con + [0]*(field_size-len(con))
            y_train.append(label)
            xv_train.append(value)
            xi_train.append(con)
 
    #print x_train.shape,x_train[0],type(x_train)
    #print y_train.shape,y_train[0],type(y_train)
    #print y_train.shape[1]
    return xi_train, xv_train, y_train


def decode_n(ins=None, field_size=37):
    #print("Loading data...")
    if len(ins) == "":
        raise ValueError("data is empty")
    xi_train = []
    xv_train = []
    y_train = []
    with open(ins,'r') as f:
        for line in f:
            #
            # 默认列数固定，否则报错
            # 0   1603:1 1708:1 1967:1 1700:1 2440:1 2639:1 2346:1 1240:1 522:1 1770:1
            # 1   2040:1 606:1 2159:1 1100:1 2435:1 1480:1 1425:1 1760:1 939:1 1631:1
            #
            lis = line.strip('\t\n').split('\t')
            label= lis[0].split(' ')[0]
            #label= [int(item) for item in label]
            con  = lis[1].split(' ')
            con = [int(item.split(':')[0]) for item in con]
            value = [float(1) for i in range(field_size)]
            if len(con)!=field_size:
                con = con + [0]*(field_size-len(con))
            y_train.append(label)
            xv_train.append(value)
            xi_train.append(con)
    #print x_train.shape,x_train[0],type(x_train)
    #print y_train.shape,y_train[0],type(y_train)
    #print y_train
    return xi_train, xv_train, y_train

def decode_valid(ins=None, field_size=37):
    #print("Loading data...")
    if len(ins) == "":
        raise ValueError("data is empty")
    xi_train = []
    xv_train = []
    y_train = []
    with open(ins,'r') as f:
        for line in f:
            #
            # 默认列数固定，否则报错
            # 0   1603:1 1708:1 1967:1 1700:1 2440:1 2639:1 2346:1 1240:1 522:1 1770:1
            # 1   2040:1 606:1 2159:1 1100:1 2435:1 1480:1 1425:1 1760:1 939:1 1631:1
            #
            lis = line.strip('\t\n').split('\1')
            label= lis[0].split(' ')[0]
            #label= [int(item) for item in label]
            con  = lis[1].split(' ')
            con = [int(item.split(':')[0]) for item in con]
            value = [float(1) for i in range(field_size)]
            if len(con)!=field_size:
                con = con + [0]*(field_size-len(con))
            y_train.append(label)
            xv_train.append(value)
            xi_train.append(con)
    xi_train = xi_train[0:int(len(xi_train)*0.8)]
    xv_train = xv_train[0:int(len(xv_train)*0.8)]
    y_train = y_train[0:int(len(y_train)*0.8)]
    xi_valid = xi_train[int(len(xi_train)*0.8):]
    xv_valid = xv_train[int(len(xv_train)*0.8):]
    y_valid = y_train[int(len(y_train)*0.8):]
    #print x_train.shape,x_train[0],type(x_train)
    #print y_train.shape,y_train[0],type(y_train)
    #print y_train
    return xi_train, xv_train, y_train, xi_valid, xv_valid, y_valid
#decode_n('/data3/ads_dm/hanxu8/DeepFm/src/t')
