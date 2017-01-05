#coding=gbk

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf

import mlp_model

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'train', 'train | predict')

flags.DEFINE_integer('input_feature_dims', 1000, 'numbers of input features')
flags.DEFINE_integer('input_label_dims', 2, 'numbers of input labels')

flags.DEFINE_string('train_data_path', '../tf_record_for_train/', 'directory of train data')
flags.DEFINE_string('validate_data_path', '../tf_record_for_test/', 'directory of validate data')
flags.DEFINE_string('data_name', 'part', 'prefix of train or validation data')

flags.DEFINE_integer('eval_interval_steps', 200, 'how many steps we eval indice')
flags.DEFINE_integer('save_interval_steps', 17041, 'how many steps we save model')

flags.DEFINE_string('model_path', './saved_models', 'path of models saved ')
flags.DEFINE_integer('max_models_keep', 50, 'max models to keep')

flags.DEFINE_string('summary_dir', './summary_logs', 'summary directory')

flags.DEFINE_integer('train_batch_size', 256, 'batch size of training examples')
flags.DEFINE_integer('valid_batch_size', 20000, 'batch size of validation examples')
flags.DEFINE_boolean('one_hot_label', False, 'batch size of validation examples')
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate')
flags.DEFINE_float('decay_rate', 0.97, 'rate of learning rate decays')
flags.DEFINE_float('threshold', 0.5, 'threshold which jude accuracy')
flags.DEFINE_integer('max_epochs', 200, 'max epochs of training')

flags.DEFINE_string('predict_input_path', 'test_data', 'path of predict data')
flags.DEFINE_string('predict_output_path', 'test_data.predict', 'path of predict result')

feature_nums = FLAGS.input_feature_dims
hidden_nums = 25
label_nums = 1
activation_name = "sigmoid"
hyperparameters = (feature_nums, hidden_nums, label_nums, activation_name)

print("feature_nums: %d" %  feature_nums)
print("hidden_nums: %d" %  hidden_nums)
print("label_nums: %d" %  label_nums)
print("activation: %s" % activation_name)

print("train_batch_size: %d" %  FLAGS.train_batch_size)
print("valid_batch_size: %d" %  FLAGS.valid_batch_size)
print("learning_rate: %f" % FLAGS.learning_rate)
print("decay_rate: %f" % FLAGS.decay_rate)
print("threshold: %f" %  FLAGS.threshold)
print("model_path: %s" % FLAGS.model_path)
print("summary_dir: %s" % FLAGS.summary_dir)
	
def main():
	if FLAGS.mode == "train":
		model = mlp_model.MLP(FLAGS, hyperparameters)
		model.train()

def get_model():
	model = mlp_model.MLP(FLAGS, hyperparameters)
	return model

def get_parameters():
	return FLAGS

if __name__ == '__main__':
	main()
