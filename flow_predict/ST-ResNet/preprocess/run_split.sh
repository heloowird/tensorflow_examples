#!/bin/sh
source ~/.bashrc

data_dir=""

#filename="train_data_38_2018-07-29-2018-09-01"
#mode="train"
filename="test_data_38_2018-09-02-2018-09-08"
mode="test"

raw_data="${data_dir}/${filename}"
image_like_data="${raw_data}_image_like"
tf_record_data="${raw_data}_tfrecord"
tf_record_train="${data_dir}/split/${filename}_tfrecord_train_split"
tf_record_valid="${data_dir}/split/${filename}_tfrecord_valid_split"

train_hdfs=""
valid_hdfs=""

#cat $raw_data | python format_data_into_image_like.py > $image_like_data  2>stat_flow_num

cat $image_like_data | python split_large_into_small.py $mode $tf_record_train $tf_record_valid $train_hdfs $valid_hdfs

if [ $? -eq 0 ]
then
    echo "done"
else
    echo "fail"
fi
