#coding=gbk

import sys
import os
import ast
import random

random.seed (1373)

import numpy as np
np.set_printoptions(threshold=np.nan)

import tensorflow as tf

_SEQ_LEN = [3, 1, 1]
_EXTEND_LEN = 26
_CHANNEL = 2
_WIDTH = 212
_HEIGHT = 177

MIN_NUM = 0
MAX_NUM = 227

_byte_feature = lambda v: tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

def transform(X):
    #X = 1. * (X - MIN_NUM) / (MAX_NUM - MIN_NUM)
    #X = X * 2. - 1.
    return X

def main(argv):
    mode = sys.argv[1]
    if mode not in ['train', 'test']:
        sys.stderr.write('bad mode\n')
        return
    train_writer = tf.python_io.TFRecordWriter(sys.argv[2])

    if mode == 'train':
        valid_writer = tf.python_io.TFRecordWriter(sys.argv[3])

    train_cnt = 0
    valid_cnt = 0
    for line in sys.stdin:
        line = line.strip("\r\n")
        info = line.split(" ", 1)
        info = ast.literal_eval(info[1])

        closeness, preriod, trend, external, label = info

        c_np = np.array(closeness, dtype=np.float32).reshape([_SEQ_LEN[0]*_CHANNEL, _HEIGHT, _WIDTH]) 
        #print(np.sum(c_np))
        c_np = transform(c_np)
        #print(np.sum(c_np))
        c_np = np.float32(c_np)
        c_feat = c_np.tostring()

        p_np = np.array(preriod, dtype=np.float32).reshape([_SEQ_LEN[1]*_CHANNEL, _HEIGHT, _WIDTH])
        #print(np.sum(p_np))
        p_np = transform(p_np)
        #print(np.sum(p_np))
        p_np = np.float32(p_np)
        p_feat = p_np.tostring()

        q_np = np.array(trend, dtype=np.float32).reshape([_SEQ_LEN[2]*_CHANNEL, _HEIGHT, _WIDTH])
        #print(np.sum(q_np))
        q_np = transform(q_np)
        #print(np.sum(q_np))
        q_np = np.float32(q_np)
        q_feat = q_np.tostring()

        e_np = np.array(external, dtype=np.float32).reshape([_EXTEND_LEN])
        #print(np.sum(e_np))
        e_np = np.float32(e_np)
        e_feat = e_np.tostring()

        label_np = np.array(label, dtype=np.float32).reshape([_CHANNEL, _HEIGHT, _WIDTH])
        #print(np.sum(label_np))
        label_np = transform(label_np)
        label_np = np.float32(label_np)
        #print(label_np)
        #print(np.sum(label_np))
        #print(np.max(label_np))
        #print(np.min(label_np))
        label = label_np.tostring()
    
    
        example = tf.train.Example(features=tf.train.Features(feature={
            'closeness': _byte_feature(c_feat),
            'preriod': _byte_feature(p_feat),
            'trend': _byte_feature(q_feat),
            'external': _byte_feature(e_feat),
            'label': _byte_feature(label)
            }))

        if random.random() < 0.9:
            train_writer.write(example.SerializeToString())
            train_cnt += 1
        else:
            if mode == "train":
                valid_writer.write(example.SerializeToString())
                valid_cnt += 1
            else:
                train_writer.write(example.SerializeToString())
                train_cnt += 1

    train_writer.flush()
    train_writer.close()

    if mode == 'train':
        valid_writer.flush()
        valid_writer.close()
    print("train smaples count: %d" % train_cnt)
    print("valid samples count: %d" % valid_cnt)

if __name__ == "__main__":
    tf.app.run()
