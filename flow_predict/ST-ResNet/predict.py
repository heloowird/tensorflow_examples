#!/usr/bin/env python
#coding:utf-8
#author:zhujianqi

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
import numpy as np
from train import *

MIN_NUM = 0
MAX_NUM = 227
    
def main(argv):
    test_data_dir = "your_predict_data_directory"
    test_data_name = "your_predict_data_filename"

    model_dir="your_trained_model_directory"

    session_config = tf.ConfigProto(allow_soft_placement=True)
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=100,
        session_config=session_config)

    def pred_input_fn():
        return input_fn(test_data_dir, test_data_name, is_training=False, dtype=tf.float32, repeat_num=1)


    model = tf.estimator.Estimator(model_fn=demand_perdict_model_fn, model_dir=model_dir,
                                    config=run_config,
                                    params={'resnet_size':20,
                                            'resnet_version':2,
                                            'loss_filter_fn':None,
                                            'weight_decay':2e-3,
                                            'loss_scale':1000,
                                            'learning_rate':0.01,
                                            'dtype':tf.float32
                                    })

    predict_results = model.predict(pred_input_fn)
    for i, p in enumerate(predict_results):
        y_pred = reverse_transform(p["logits"])
        print(y_pred[0].flatten().tolist())
    
def reverse_transform(X):
    X = (X + 1.0) / 2.
    X = X * (MAX_NUM - MIN_NUM) + MIN_NUM
    return np.ceil(X)
    #return np.floor(X)
    #return np.around(X)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

