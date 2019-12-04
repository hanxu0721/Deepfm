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
    def __init__(self, feature_size, factor_size, field_size, loss_type,
                 deep_layers=[20, 20],
                 deep_layers_activation=tf.nn.relu):
        self.feature_size = feature_size # M
        self.factor_size  = factor_size # k
        self.field_size   = field_size  # F
        self.loss_type = loss_type
        self.deep_layers  = deep_layers
        self.deep_layers_activation = deep_layers_activation
       
   
    def linear_order_part(self,sparse_id, sparse_value):
        with tf.name_scope("linear_order"):
            W = tf.Variable(tf.random_normal([self.feature_size, 1],0.0,0.1,name ="linear_weights"))
            y_linear_part = tf.nn.embedding_lookup(W,sparse_id) # none*F*1
            y_linear_part = tf.reduce_sum(tf.multiply(y_linear_part,sparse_value),axis =1) # none*1
            
            return y_linear_part
    def Second_order_part(self,sparse_id, sparse_value):   
        with tf.name_scope("Second_order"):
            V = tf.Variable(tf.random_normal([self.feature_size, self.factor_size],0.0,0.1,name ="cross_weights"))
            self.embedding = tf.nn.embedding_lookup(V,sparse_id) #none*F*K
            self.embedding = tf.multiply(self.embedding, sparse_value)
            square_sum = tf.square(tf.reduce_sum(self.embedding,1)) # none*K
            sum_square = tf.reduce_sum(tf.square(self.embedding),1) # none*K
            
            y_second_order = 0.5*tf.subtract(square_sum,sum_square)
            return y_second_order
        
    def deep(self,dropout_keep_deep):
        with tf.name_scope("deep_model"):
            y_deep = tf.reshape(self.embedding,[-1,self.factor_size * self.field_size]) #none*(F*K)
            y_deep = tf.nn.dropout(y_deep, dropout_keep_deep[0])
            #print y_deep
            for i in range(0,len(self.deep_layers)):
                y_deep = tf.contrib.layers.fully_connected(y_deep, self.deep_layers[i], activation_fn=self.deep_layers_activation, scope = 'fc%d' % i)
                y_deep = tf.nn.dropout(y_deep, dropout_keep_deep[1+i])
            return y_deep
    
    def forward(self, sparse_id, sparse_value, dropout_keep_deep):
        sparse_value = tf.reshape(sparse_value, shape=[-1, self.field_size, 1])
        sparse_value = tf.cast(sparse_value,dtype=tf.float32) 
        y_linear_part = self.linear_order_part(sparse_id, sparse_value)
        y_second_order = self.Second_order_part(sparse_id, sparse_value)
        y_deep = self.deep(dropout_keep_deep)
        with tf.name_scope("deep_fm"):
            output = tf.concat([y_linear_part, y_second_order, y_deep], axis = 1)
            output = tf.contrib.layers.fully_connected(y_deep, 1, activation_fn=tf.nn.sigmoid, scope = 'deepfm_out')
            return output
    
    def loss(self, sparse_id, sparse_value, label, dropout_keep_deep):
        predict_y = self.forward(sparse_id, sparse_value, dropout_keep_deep)
        if self.loss_type =='log_loss':
            loss = tf.losses.log_loss(label, predict_y)
        elif self.loss_type =='mse':
            loss = tf.nn.l2_loss(tf.subtract(label, predict_y))
        auc_op, auc_value = tf.metrics.auc(label, predict_y) 
        return loss,auc_value
