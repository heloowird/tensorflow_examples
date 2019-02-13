#coding=gbk

import sys
import os
import ast
import random

random.seed (1373)

import numpy as np

import tensorflow as tf

_SEQ_LEN = [3, 1, 1]
_EXTEND_LEN = 26
_CHANNEL = 2
_WIDTH = 212
_HEIGHT = 177

MIN_NUM = 0
MAX_NUM = 227

_TINY_WIDTH = 31
_TINY_HEIGHT = 31

_byte_feature = lambda v: tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))

def load_geo(filename):
    geo_index = {}
    with open(filename) as f:
        index = 0
        for line in f:
            line = line.strip('\r\n')
            geo_index[line] = index
            index += 1
    return geo_index

def load_target(filename):
    geo_index = load_geo("")
    print(len(geo_index))
    target_index = set()
    with open(filename) as f:
        for line in f:
            line = line.strip('\r\n')
            info = line.split('\t')
            if float(info[1]) > 100. and info[0] in geo_index:
                target_index.add(geo_index[info[0]])
    return target_index

target_index = load_target("")
print(len(target_index))

def transform(X):
    X = 1. * (X - MIN_NUM) / (MAX_NUM - MIN_NUM)
    X = X * 2. - 1.
    return X

mode = sys.argv[1]
if mode not in ['train', 'test']:
    sys.stderr.write('bad mode\n')
    sys.exit()

train_cnt = 0
train_filename = "%s_%d" % (sys.argv[2], train_cnt)
train_writer = tf.python_io.TFRecordWriter(train_filename)

valid_cnt = 0
valid_writer = None
valid_filename = "/tmp/tmp"
if mode == 'train':
    valid_filename = "%s_%d" % (sys.argv[3], valid_cnt)
    valid_writer = tf.python_io.TFRecordWriter(valid_filename)

def main(argv):
    mode = sys.argv[1]
    if mode not in ['train', 'test']:
        sys.stderr.write('bad mode\n')
        return

    for line in sys.stdin:
        line = line.strip("\r\n")
        info = line.split(" ", 1)
        info = ast.literal_eval(info[1])

        closeness, preriod, trend, external, label = info

        c_np = np.array(closeness).reshape([_SEQ_LEN[0]*_CHANNEL, _HEIGHT, _WIDTH]) 
        c_np = np.float32(c_np)

        p_np = np.array(preriod).reshape([_SEQ_LEN[1]*_CHANNEL, _HEIGHT, _WIDTH])
        p_np = np.float32(p_np)

        q_np = np.array(trend).reshape([_SEQ_LEN[2]*_CHANNEL, _HEIGHT, _WIDTH])
        q_np = np.float32(q_np)

        e_np = np.array(external).reshape([_EXTEND_LEN])
        e_np = np.float32(e_np)

        label_np = np.array(label).reshape([_CHANNEL, _HEIGHT, _WIDTH])
        label_np = np.float32(label_np)

        split_into_small(c_np, p_np, q_np, e_np, label_np, mode)
    
    train_writer.flush()
    train_writer.close()
    os.system("hadoop fs -put %s %s" % (train_filename, sys.argv[4]))
    #os.system("rm -f %s" % (train_filename))
    if mode == 'train':
        valid_writer.flush()
        valid_writer.close()
        os.system("hadoop fs -put %s %s" % (valid_filename, sys.argv[5]))
        #os.system("rm -f %s" % (valid_filename))

    print("train smaples count: %d" % train_cnt)
    print("valid samples count: %d" % valid_cnt)

def pad(arr, h_pad, w_pad):
    return np.pad(arr, ((0,0), (h_pad,h_pad), (w_pad,w_pad)), 'constant', constant_values=0)

def split_into_small(c, p, q, e, l, mode):
    global train_cnt
    global train_filename
    global train_writer
    global valid_cnt
    global valid_filename
    global valid_writer

    channel, height, width = l.shape
   
    pad_h, pad_w = _TINY_HEIGHT//2, _TINY_WIDTH//2
    c = pad(c, pad_h, pad_w)
    p = pad(p, pad_h, pad_w)
    q = pad(q, pad_h, pad_w)

    for i in range(height):
        for j in range(width):
            index = i*width+j
            if index not in target_index:
                continue

            c_ele = c[:, i:i+2*pad_h+1, j:j+2*pad_w+1]
            p_ele = p[:, i:i+2*pad_h+1, j:j+2*pad_w+1]
            q_ele = q[:, i:i+2*pad_h+1, j:j+2*pad_w+1]
            l_ele = l[:, i, j].reshape(2)
    
            c_ele = np.float32(c_ele)
            c_feat = c_ele.tostring()

            p_ele = np.float32(p_ele)
            p_feat = p_ele.tostring()

            q_ele = np.float32(q_ele)
            q_feat = q_ele.tostring()

            e = np.float32(e)
            e_feat = e.tostring()

            l_ele = np.float32(l_ele)
            label = l_ele.tostring() 

            example = tf.train.Example(features=tf.train.Features(feature={
                'closeness': _byte_feature(c_feat),
                'preriod': _byte_feature(p_feat),
                'trend': _byte_feature(q_feat),
                'external': _byte_feature(e_feat),
                'label': _byte_feature(label)
                }))
            
            is_train_changed, is_valid_changed = False, False

            if random.random() < 0.9:
                train_writer.write(example.SerializeToString())
                train_cnt += 1
                is_train_changed = True
            else:
                if mode == "train":
                    valid_writer.write(example.SerializeToString())
                    valid_cnt += 1
                    is_valid_changed = True
                else:
                    train_writer.write(example.SerializeToString())
                    train_cnt += 1
                    is_train_changed = True

            if train_cnt and train_cnt % 8000 == 0 and is_train_changed:
                train_writer.flush()
                train_writer.close()
                os.system("hadoop fs -put %s %s" % (train_filename, sys.argv[4]))
                #os.system("rm -f %s" % (train_filename))
                train_filename = "%s_%d" % (sys.argv[2], train_cnt)
                train_writer = tf.python_io.TFRecordWriter(train_filename)

            if mode == "train" and valid_cnt and valid_cnt % 8000 == 0 and is_valid_changed:
                valid_writer.flush()
                valid_writer.close()
                os.system("hadoop fs -put %s %s" % (valid_filename, sys.argv[5]))
                #os.system("rm -f %s" % (valid_filename))
                valid_filename = "%s_%d" % (sys.argv[3], valid_cnt)
                valid_writer = tf.python_io.TFRecordWriter(valid_filename)

if __name__ == "__main__":
    tf.app.run()
