# coding=utf-8

import sys
import io
import numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import tensorflow as tf

import data_helper

#test_file  = "./data/geo_poi_bj_1_sample_100"
test_file  = "./data/test_name"
feature_file = "../data_v2/poi_name_w2v.voc"
label_file = "../data_v2/class_2_dict_v2"

x_pre, y_pre, vocabulary, vocabulary_inv, label2id, id2label = data_helper.build_input_data(test_file, feature_file, label_file, mode="test")
raw_sentences, _ = data_helper.load_raw_data(test_file, "test")

"""
Restore the model
"""
model_dir = "./model/"
checkpoint_file = tf.train.latest_checkpoint(model_dir)
#checkpoint_file = "./model/model-1296000"
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0] # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]


def predict():
    predicted_results = sess.run(predictions, {input_x: x_pre, dropout_keep_prob: 1.0})
    for sen, ele in zip(raw_sentences, predicted_results):
        #print(ele)
        print("{}\t{}".format(sen, id2label[ele]))
        #print("{}".format(id2label[ele].encode('utf-8').decode('ascii')))
        #print("{}".format(id2label[ele].encode('utf-8').decode('unicode_escape')))

if __name__ == '__main__':
    predict()
