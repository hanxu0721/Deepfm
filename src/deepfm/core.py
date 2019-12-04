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
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
#import pandas as pd

class DeepFM:
    def __init__(self, feature_size, factor_size, field_size, loss_type,
                 deep_layers=[20, 20],
                 l2_reg=0.0,
		 mode='lr',
		 batch_norm=0,
		 batch_norm_decay=0.995,
                 deep_layers_activation=tf.nn.relu):
        self.feature_size = feature_size # M
        self.factor_size  = factor_size # k
        self.field_size   = field_size  # F
        self.loss_type = loss_type
        self.deep_layers  = deep_layers
        self.l2_reg = l2_reg
        self.mode = mode
        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay
        self.deep_layers_activation = deep_layers_activation
        self.weights = self._initialize_weights()      

    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights["Second_order"] = tf.Variable(
            tf.random_normal([self.feature_size, self.factor_size], 0.0, 0.01),
            name="cross_weights")  # feature_size * K
        weights["linear_order"] = tf.Variable(
            tf.random_uniform([self.feature_size, 1], 0.0, 1.0), name="linear_weights")  # feature_size * 1

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.factor_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        if self.mode=='deepfm':
            input_size = 1 + self.factor_size + self.deep_layers[-1]
        elif self.mode=='deep':
	    input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights
    def batch_norm_layer(self, x, train, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train, lambda: bn_train, lambda: bn_inference)
        return z 

    def linear_order_part(self,sparse_id, sparse_value):
        with tf.name_scope("linear_order"):
            #W = tf.Variable(tf.random_normal([self.feature_size, 1],0.0,0.1,name ="linear_weights"))
            y_linear_part = tf.nn.embedding_lookup(self.weights["linear_order"],sparse_id) # none*F*1
            y_linear_part = tf.reduce_sum(tf.multiply(y_linear_part,sparse_value),axis =1) # none*1
            
            return y_linear_part

    def Second_order_part(self,sparse_id, sparse_value):   
        with tf.name_scope("Second_order"):
            #V = tf.Variable(tf.random_normal([self.feature_size, self.factor_size],0.0,0.1,name ="cross_weights"))
            self.embedding = tf.nn.embedding_lookup(self.weights["Second_order"],sparse_id) #none*F*K
            self.embedding = tf.multiply(self.embedding, sparse_value)
            square_sum = tf.square(tf.reduce_sum(self.embedding,1)) # none*K
            sum_square = tf.reduce_sum(tf.square(self.embedding),1) # none*K
            
            y_second_order = 0.5*tf.subtract(square_sum,sum_square)
	    y_fm_out = tf.reduce_sum(y_second_order,axis=1)
            return y_second_order,y_fm_out
        
    def deep(self,dropout_keep_deep,train_phase):
        with tf.name_scope("deep_model"):
            y_deep = tf.reshape(self.embedding, shape=[-1, self.field_size * self.factor_size]) # None * (F*K)
            y_deep = tf.nn.dropout(y_deep, dropout_keep_deep[0])
            for i in range(0, len(self.deep_layers)):
                y_deep = tf.add(tf.matmul(y_deep, self.weights["layer_%d" %i]), self.weights["bias_%d"%i]) # None * layer[i] * 1
                if self.batch_norm:
                    y_deep = self.batch_norm_layer(y_deep, train=train_phase, scope_bn="bn_%d" %i)
                y_deep = self.deep_layers_activation(y_deep)
                y_deep = tf.nn.dropout(y_deep, dropout_keep_deep[1+i])
            return y_deep
        
    
    def forward(self, sparse_id, sparse_value, dropout_keep_deep, train_phase):
        sparse_value = tf.reshape(sparse_value, shape=[-1, self.field_size, 1])
        sparse_value = tf.cast(sparse_value,dtype=tf.float32) 
        y_linear_part = self.linear_order_part(sparse_id, sparse_value)
        y_second_order,y_fm_out = self.Second_order_part(sparse_id, sparse_value)
        y_deep = self.deep(dropout_keep_deep,train_phase)
        with tf.name_scope("forward"):
	    if self.mode=='lr':
	       output=y_linear_part
	    elif self.mode=='fm':
	       output = y_linear_part+y_fm_out
	    elif self.mode=='deep':
	       output = tf.add(tf.matmul(y_deep, self.weights["concat_projection"]), self.weights["concat_bias"],name="out_put")
	    elif self.mode=='deepfm':
               concat_input = tf.concat([y_linear_part, y_second_order, y_deep], axis=1)
               output = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"],name="out_put")
            return output
    
    def loss(self, sparse_id, sparse_value, label, dropout_keep_deep, train_phase):
        out = self.forward(sparse_id, sparse_value, dropout_keep_deep, train_phase)
        if self.loss_type =='log_loss':
            out = tf.nn.sigmoid(out,name="prob")
	    cost = tf.losses.log_loss(label, out)
            loss = tf.losses.log_loss(label, out)
        elif self.loss_type =='mse':
            loss = tf.nn.l2_loss(tf.subtract(label, out))
        if self.mode=='deepfm' and self.l2_reg > 0:
            loss += tf.contrib.layers.l2_regularizer(
                    self.l2_reg)(self.weights["concat_projection"])
            for i in range(len(self.deep_layers)):
                loss += tf.contrib.layers.l2_regularizer(
                            self.l2_reg)(self.weights["layer_%d"%i])
        elif self.mode=='lr' and self.l2_reg > 0:
	    loss += tf.contrib.layers.l2_regularizer(
	                        self.l2_reg)(self.weights["linear_order"])
        auc_op, auc_value = tf.metrics.auc(label, out) 
        return cost,loss,auc_value
