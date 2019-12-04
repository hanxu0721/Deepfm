#!/bin/bash
cur_dir=`pwd`
source ${cur_dir}/model.conf
source ${cur_dir}/deepfm.conf
#num_parts=`    hdfs dfs -ls   ${instance}              | grep part        |wc -l|awk '{print $1}'`
#num_ins=`     hdfs dfs -text ${merge_fea}/*            | grep     Belta0  | awk '{print $3}'`
#num_runs=1
#feature_size=`hdfs dfs -text ${feature}/*              | wc   -l          | awk '{print $1}'`


num_ins=10000
num_runs=1
feature_size=5345409
#8230571
#122638


# 超参
info="train_phase=${train_phase} batch_norm=${batch_norm} batch_norm_decay=${batch_norm_decay} mode=${mode} max_epoch=${max_epochs} batch_size=${batch_size} factor_size=${factor_size} dropout_out=${drop_out} "

# Deep隐藏层
deep="deep_layers=${deep_layers}"

# learning
learning="optimizer=${optimizer} learning_rate=${learning_rate}"

reg="l2_reg=${l2_reg}"
# 正则化参数
#reg="l1_reg_linear=${l1_reg_linear} l2_reg_linear=${l2_reg_linear} l1_reg_embedding=${l1_reg_embedding} l2_reg_embedding=${l2_reg_embedding} l1_reg_deep=${l1_reg_deep} l2_reg_deep=${l2_reg_deep}"
echo -e "${info} ${deep} ${learning} ${reg}"

train(){
    hdfs dfs -mkdir ${model};  hdfs dfs -chmod 777 ${model};  hdfs dfs -rmr ${model}/*
    hdfs dfs -mkdir ${log_dir};hdfs dfs -chmod 777 ${log_dir};hdfs dfs -rmr ${log_dir}/*
    TensorFlow_Submit  \
    --appName="${person}_${app_name}"  \
    --archives=hdfs://ns3-backup/dw_ext/ad/person/hanxu8/DL/Python_2.7.13_TF_1.2.0.zip#Python \
    --files=./DataParser.py,./core.py,./train.py,./deep_input.py \
    --worker_memory=${worker_memory} \
    --worker_cores=2  \
    --num_worker=${num_worker}  \
    --num_ps=${num_ps}   \
    --ps_memory=${ps_memory} \
    --appType=TensorFlow \
    --mode_local=true \
    --tensorboard=true \
    --data_dir=${instance} \
    --log_dir=${log_dir}/  \
    --train_dir=${model}/ \
    --command=Python/bin/python train.py num_runs=${num_runs} num_ins=${num_ins} field_size=${field_size} feature_size=${feature_size} ${info} ${reg} ${deep} ${learning} ${reg}
}
train

#cd ${cur_dir}/weights
#nohup sh -x run.sh
