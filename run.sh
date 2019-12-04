#!/bin/bash

cur_dir=`pwd`
#source "${cur_dir}/src/data.conf"

function training() {
    source ${cur_dir}/src/deepfm/deepfm.conf
    cd ${cur_dir}/src/deepfm
    nohup sh -x run.sh ${cur_time} ${person} > ${cur_dir}/log/${today}.${cur_time}.log.training 2>&1
}
training
function training_lr() {
    source ${cur_dir}/src/lr/ctr.conf
    cd ${cur_dir}/src/lr
    nohup sh -x run.sh ${cur_time} ${person} > ${cur_dir}/log/${today}.${cur_time}.log.training 2>&1
}
#training_lr
