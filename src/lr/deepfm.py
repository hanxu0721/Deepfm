#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 15:52:56 2019

@author: hanxu8
"""

import tensorflow as tf
import os
import sys
#from sklearn.metrics import roc_auc_score
import numpy as np
#import pandas as pd

class DeepFM:
    def __init__(self, feature_size, factor_size, field_size,l2_reg, loss_type):
        self.feature_size = feature_size # M
        self.factor_size  = factor_size # k
        self.field_size   = field_size  # F
	self.l2_reg = l2_reg
        self.loss_type = loss_type
      
       
   
    def linear_order_part(self,sparse_id, sparse_value):
        with tf.name_scope("linear_order"):
            W = tf.Variable(tf.random_normal([self.feature_size, 1],0.0,0.1,name ="linear_weights"))
            y_linear_part = tf.nn.embedding_lookup(W,sparse_id) # none*F*1
            y_linear_part = tf.reduce_sum(tf.multiply(y_linear_part,sparse_value),axis =1) # none*1
            
            return y_linear_part,W
    def forward(self, sparse_id, sparse_value):
        sparse_value = tf.reshape(sparse_value, shape=[-1, self.field_size, 1])
        sparse_value = tf.cast(sparse_value,dtype=tf.float32) 
        y_linear_part,W = self.linear_order_part(sparse_id, sparse_value)
        return y_linear_part,W 
    def loss(self, sparse_id, sparse_value, label):
        out,W = self.forward(sparse_id, sparse_value)
        if self.loss_type =='log_loss':
	    out = tf.nn.sigmoid(out,name="prob")

	    cost = tf.losses.log_loss(label, out)
            loss = tf.losses.log_loss(label, out)
        elif self.loss_type =='mse':
            loss = tf.nn.l2_loss(tf.subtract(label, out))
	if self.l2_reg > 0:
	    loss += tf.contrib.layers.l2_regularizer(
	                                    self.l2_reg)(W)
        auc_op, auc_value = tf.metrics.auc(label, out) 
        return out,cost,loss,auc_value
