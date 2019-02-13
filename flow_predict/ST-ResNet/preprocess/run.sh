#!/bin/sh

data_dir=""

#filename="train_data_38_2018-07-29-2018-09-01"
filename="test_data_38_2018-09-02-2018-09-08"
mode="test"

raw_data="${data_dir}/${filename}"
image_like_data="${raw_data}_image_like"
tf_record_data="${raw_data}_tfrecord"
tf_record_train="${tf_record_data}_train"
tf_record_valid="${tf_record_data}_valid"

rm -f $image_like_data
cat $raw_data | python format_data_into_image_like.py > $image_like_data  2>stat_flow_num_${mode}

rm -f $tf_record_train
rm -f $tf_record_valid
cat $image_like_data | python format_data_into_tfrecord.py $mode $tf_record_train $tf_record_valid 1>log_sample_cnt_${mode}

if [ $? -eq 0 ]
then
    echo "done"
else
    echo "fail"
fi
