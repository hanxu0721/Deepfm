#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 17:24:17 2019

@author: hanxu8
"""

import tensorflow as tf
#from DataParser import decode_n
import os

#ins = '/Users/hanxu8/Documents/DL/fm/hx_deepfm/test'
data_dir='/data4/ads_dm/hanxu8/DeepFm/predict/tmp_hx_ctr_test_feas'
filenames = tf.gfile.ListDirectory(data_dir)
filenames = [os.path.join(data_dir, x) for x in filenames]
filenames.sort(key=lambda x : int(x.split('/')[-1].split('_')[0] ))
filenames = [x for x in filenames if tf.gfile.Exists(x)]
ins_list = []
for item in filenames:
    ins_list.append(item)
    #print (item)
    
def decode_n(ins=None, field_size=36):
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
            #print(xv_train,xi_train)

    #print x_train.shape,x_train[0],type(x_train)
    #print y_train.shape,y_train[0],type(y_train)
    #print y_train.shape[1]
    return xi_train, xv_train, y_train


def decode_test(ins=None, field_size=36):
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
            lis = line.strip('\t\n')
            #label= lis[0].split(' ')[0]
            #label= [int(item) for item in label]
            con  = lis.split(' ')
            con = [int(item.split(':')[0]) for item in con]
            value = [float(1) for i in range(field_size)]
            if len(con)!=field_size:
                con = con + [0]*(field_size-len(con))
            #y_train.append(label)
            xv_train.append(value)
            xi_train.append(con)
            #print(xv_train,xi_train)

    #print x_train.shape,x_train[0],type(x_train)
    #print y_train.shape,y_train[0],type(y_train)
    #print y_train.shape[1]
    return xi_train, xv_train
field_size=36
with tf.Session() as sess:
    # 导入模型
    saver = tf.train.import_meta_graph("/data4/ads_dm/hanxu8/DeepFm/predict/model.ckpt-1043135.meta", clear_devices=True)
    saver.restore(sess, '/data4/ads_dm/hanxu8/DeepFm/predict/model.ckpt-1043135')
    graph = tf.get_default_graph()# 获取当前图，为了后续训练时恢复变量
    #sess.run(tf.global_variables_initializer())
    tensor_name_list = [tensor.name for tensor in graph.as_graph_def().node]# 得到当前图中所有变量的名称
    #print (tensor_name_list)
    #model_file = tf.train.latest_checkpoint('/Users/hanxu8/Documents/DL/fm/hx_deepfm/')
    for i in ins_list:
        #Xi, Xv,y = decode_n(ins=i,field_size=field_size)
        #print i
        Xi, Xv = decode_test(ins=i,field_size=field_size)   
 #print (Xi)
        pred = graph.get_tensor_by_name('prob:0')# 获取网络输出值
        sparse_id  = graph.get_tensor_by_name('sparse_id:0')
        sparse_value = graph.get_tensor_by_name('sparse_value:0')
        #dropout_keep_deep = graph.get_tensor_by_name('dropout_keep_deep:0')
        #drop_out = [1.,1.,1.]
        feed_dict={sparse_id:Xi,
               sparse_value:Xv
               }
        pred_value=  sess.run(pred,feed_dict = feed_dict)
    #print(pred_value)
        for i in range(len(pred_value)):
            print (str(pred_value[i][0]))
    
