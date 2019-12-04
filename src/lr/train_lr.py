#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:07:26 2019

@author: hanxu8
"""



import numpy as np
import tensorflow as tf
import time
import os
import datetime
from DataParser import get_batch,shuffle_in_unison_scary
from deepfm import DeepFM
from deep_input import decode, decode_n

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("data_dir","./data","directory of data")
flags.DEFINE_string("log_dir","","directory of data")
flags.DEFINE_string('train_dir', '',
                           """Directory where to write event logs """
                           """and checkpoint.""")

# Flags for defining the tf.train.ClusterSpec
flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs, you can also specify pattern like ps[1-5].example.com")
flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs, you can also specify worker[1-5].example.co")

# Flags for defining the tf.train.Server

flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")

###
flags.DEFINE_integer("factor_size", 128, "Dimensionality of character embedding (default: 128)")
flags.DEFINE_integer("num_ins", 5003, "num ins (default: 64)")
flags.DEFINE_integer("num_runs",   1, "Batch Size (default: 64)")
flags.DEFINE_integer("feature_size", 5003, "feature Size (default: 64)")
flags.DEFINE_integer("field_size", 21, "field Size (default: 64)")
flags.DEFINE_string("deep_layers", "200-200-200", "Comma-separated filter sizes (default: '3,4,5')")
flags.DEFINE_string("dropout_out", "0.9-0.9-0.9", "Dropout keep probability (default: 0.5)")
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_float('l2_reg', 0.001, 'l2_reg')
flags.DEFINE_integer('max_epoch', 20, ' max train epochs')
flags.DEFINE_integer("batch_size", 20, "batch size for sgd")
#flags.DEFINE_string("checkpoint_dir", "./checkpoint/", "checkpoint dirctory")
flags.DEFINE_string("mode", "lr", "train_mode")
flags.DEFINE_string("optimizer", "adam", "optimization algorithm")
flags.DEFINE_string("loss", "log_loss", "loss function")



#if not os.path.exists(FLAGS.checkpoint_dir):
#    os.makedirs(FLAGS.checkpoint_dir)
 


   
def train(Xi, Xv, y, local_step=None, num_examples=None, startt=None):
    #drop_out = [float(item) for item in FLAGS.dropout_out.split('-')]
    feed_dict ={sparse_id:Xi,
                sparse_value:Xv,
                label:y
               }
    if local_step % 100 == 0 and FLAGS.task_index == 0:
        num_examples += len(y)
        _, prob,cost_value,loss_value, auc_value, step = sess.run([train_op,out, cost,loss, auc, global_step], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        endt = time.time()
        duration = endt - startt
        sec_per_batch = float(duration) / (100+1)
        examples_per_sec = num_examples / float(duration)
        print ("{}:\tstep={} , cost={:.5f}, loss={:.5f}, auc={:.5f}|\t{:g} examples/sec, {:g}sec/batch".format(time_str.split('.')[0],step,cost_value,loss_value,auc_value,examples_per_sec,sec_per_batch))
        #print ("predict:")
        for i in prob:
            print(i) 
    else:
        num_examples += len(y)*FLAGS.num_runs
        for i in xrange(FLAGS.num_runs):
            _, step = sess.run([train_op, global_step],feed_dict)
    return step,num_examples,startt



filenames = tf.gfile.ListDirectory(FLAGS.data_dir)
#filenames = [x for x in filenames if 'part' in x]
filenames = [os.path.join(FLAGS.data_dir, x) for x in filenames]
filenames = [x for x in filenames if tf.gfile.Exists(x)]
ins_list = []
for item in filenames:
    ins_list.append(item)
    print item

# Training
# ==================================================
ps_hosts = FLAGS.ps_hosts.split(",")
print(ps_hosts)
worker_hosts = FLAGS.worker_hosts.split(",")
print(worker_hosts)

# Create a cluster from the parameter server and worker hosts.
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
# Create and start a server for the local task.
server = tf.train.Server(cluster,
                           job_name=FLAGS.job_name,
                           task_index=FLAGS.task_index)

if FLAGS.job_name == "ps":
    print("Current process id: {}".format(os.getpid()))
    server.join()
elif FLAGS.job_name == "worker":
    print("Current process id: {}".format(os.getpid()))
    print("worker={}".format(FLAGS.task_index))
    with tf.device(tf.train.replica_device_setter(
        worker_device="/job:{}/task:{}".format(FLAGS.job_name, FLAGS.task_index),
            cluster=cluster)):
        print("Optimization algorithm: {}".format(FLAGS.optimizer))
        if FLAGS.optimizer   == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        elif FLAGS.optimizer == "adadelta":
            optimizer = tf.train.AdadeltaOptimizer(FLAGS.learning_rate)
        elif FLAGS.optimizer == "adagrad":
            optimizer = tf.train.AdagradOptimizer(FLAGS.learning_rate)
        elif FLAGS.optimizer == "adam":
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        else:
            print("Error: unknown optimizer: {}".format(FLAGS.optimizer))
            exit(1)
            
        global_step = tf.Variable(0, name="global_step", trainable=False)
        sparse_id = tf.placeholder(tf.int32, shape=[None, None], name="sparse_id")
        sparse_value = tf.placeholder(tf.float32,shape=[None, None], name="sparse_value")
        label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
        #dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
        model =  DeepFM(feature_size = FLAGS.feature_size, 
                    factor_size = FLAGS.factor_size, 
                    field_size = FLAGS.field_size,
                    loss_type = FLAGS.loss, 
                    l2_reg = FLAGS.l2_reg
                    )
    
        out,cost,loss,auc = model.loss(sparse_id,sparse_value,label) 
        train_op    = optimizer.minimize(loss, global_step = global_step)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        saver = tf.train.Saver()
        timestamp = str(int(time.time()))
        logdir = os.path.join(FLAGS.train_dir)
        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 init_op=init_op,
                                 logdir=logdir,
                                 summary_op=None,
                                 saver=saver,
                                 global_step=global_step,
                                 save_model_secs=60,
                                 save_summaries_secs=60)
    
        with sv.managed_session(server.target) as sess:
            sess.run(init_op)
            part_step, local_step, train_step = 0, 0, 0
            num_examples = 0
            startt = time.time()
            while (FLAGS.task_index==0 and train_step*FLAGS.batch_size < FLAGS.num_ins*FLAGS.max_epoch) or (FLAGS.task_index != 0):
                ins = ins_list[part_step%len(ins_list)]
                part_step += 1
                Xi_train, Xv_train, y_train = decode_n(ins=ins,field_size=FLAGS.field_size)
                
                for epoch in range(FLAGS.max_epoch):
                    print ("eopch={}".format(epoch))
                    shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
                    total_batch = int(len(y_train) / FLAGS.batch_size)
                    print ("total_batch={}".format(total_batch))
                    for i in range(total_batch):
                        Xi_batch, Xv_batch,y_batch = get_batch(Xi_train, Xv_train, y_train, FLAGS.batch_size, i)
                        train_step,num_examples,startt = train(Xi=Xi_batch, 
                                                               Xv = Xv_batch, y=y_batch, local_step=local_step, num_examples=num_examples, startt=startt)
                        local_step += 1
            print(str(datetime.datetime.now().isoformat()))
            print("结束 done")
