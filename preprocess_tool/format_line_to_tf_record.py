#coding=gbk

import sys
import os

import numpy as np

import tensorflow as tf

_byte_feature = lambda v: tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

def main(argv):
    writer = tf.python_io.TFRecordWriter(sys.argv[2])
    with open(sys.argv[1]) as f:
        for line in f:
            line = line.strip("\r\n")
            info = line.split("\t")[0]
            fs = info.split()
            
            label_np = np.array([np.int32(int(fs[0]) == i) for i in range(2)]).astype(np.float32)
            label = label_np.tostring()
    
            feat_np = np.array([np.float32(x) for x in fs[1:]])
            feat = feat_np.tostring()
    
            example = tf.train.Example(features=tf.train.Features(feature={
                'feat': _byte_feature(feat),
                'label': _byte_feature(label)
                }))
            writer.write(example.SerializeToString())

if __name__ == "__main__":
    tf.app.run()
