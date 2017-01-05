#coding=gbk

from __future__ import absolute_import
from __future__ import print_function

import sys
import os

import numpy as np
import tensorflow as tf

import util

activation_map = {'sigmoid' :  tf.nn.sigmoid, 'tanh' : tf.nn.tanh, 'relu' : tf.nn.relu}

class MLP():
	def __init__(self, FLAGS, hyperparameters):
		self.FLAGS = FLAGS
		self.feature_nums, self.hidden_nums, self.label_nums, self.activation_name = hyperparameters
		self.hidden_w = self.init_weight([self.feature_nums, self.hidden_nums])
		self.hidden_b = self.init_bias([self.hidden_nums])
		self.output_w = self.init_weight([self.hidden_nums, self.label_nums])
		self.output_b = self.init_bias([self.label_nums])
		self.activation = activation_map[self.activation_name]

	def init_weight(self, shape):
		return tf.Variable(tf.random_normal(shape))

	def init_bias(self, shape):
		return tf.Variable(tf.zeros(shape))

	def mlp_forward(self, input, hidden_w, hidden_b, output_w, output_b):
		hidden_y =	self.activation(tf.matmul(input, hidden_w) + hidden_b)
		output_sum = tf.matmul(hidden_y, output_w) + output_b
		return output_sum

	def forward(self, features):
		return self.mlp_forward(features, self.hidden_w, self.hidden_b, self.output_w, self.output_b)

	def build_train_graph(self, features, labels, lr):
		logit = self.forward(features)
		loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logit, labels))
		train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
		return train_op, loss

	def predict_label(self, features):
		return self.activation(self.forward(features))

	def get_accuracy(self, features, labels):
		y_hat = self.predict_label(features)
		correct = tf.equal(tf.cast(tf.greater_equal(y_hat, self.FLAGS.threshold), tf.float32), labels) 
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		return accuracy

	def train(self):
		features = tf.placeholder(tf.float32, [None, self.feature_nums])
		labels = tf.placeholder(tf.float32, [None, self.label_nums])
		num_of_epoch = tf.placeholder(tf.float32)

		lr = self.FLAGS.learning_rate * (self.FLAGS.decay_rate ** num_of_epoch)

		train_op, loss = self.build_train_graph(features, labels, lr) 
		accuracy = self.get_accuracy(features, labels)

		tf.scalar_summary("learning rate", lr)
		tf.scalar_summary("loss", loss)
		tf.scalar_summary("accuracy", accuracy)

		# Read train data
		train_data = util.data_util(self.FLAGS, "train")
		batch_train_feature, batch_train_label = train_data.gen_batch(self.FLAGS.train_batch_size)

		# Read validate data
		valid_data = util.data_util(self.FLAGS, "valid")
		batch_valid_feature, batch_valid_label = valid_data.gen_batch(self.FLAGS.valid_batch_size)
	
		sess = tf.Session()
	
		saver = tf.train.Saver(max_to_keep=self.FLAGS.max_models_keep)
		ckpt = tf.train.get_checkpoint_state(self.FLAGS.model_path)
		pre_step = -1
		if ckpt and ckpt.model_checkpoint_path:
			print("load existing model from: %s" % ckpt.model_checkpoint_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("train from restored model: %s" % ckpt.model_checkpoint_path)
			pre_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
		else:
			sess.run(tf.initialize_all_variables())
		
		merged = tf.merge_all_summaries()
		train_writer = tf.train.SummaryWriter('%s/train' % self.FLAGS.summary_dir, sess.graph)
		test_writer = tf.train.SummaryWriter('%s/test' % self.FLAGS.summary_dir)
	
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)
		try:
			step = pre_step + 1
			while not coord.should_stop():
				batch_xs, batch_ys = sess.run([batch_train_feature, batch_train_label])
				if not self.FLAGS.one_hot_label:
					batch_ys = np.argmax(batch_ys, 1).reshape(self.FLAGS.train_batch_size, self.label_nums)
				train_feed_dict = {features:batch_xs, labels:batch_ys, num_of_epoch:step//self.FLAGS.save_interval_steps}

				if step % self.FLAGS.eval_interval_steps == 0:
					summary_train, train_loss, train_acc, cur_lr = sess.run([merged, loss, accuracy, lr], feed_dict=train_feed_dict)
					train_writer.add_summary(summary_train, step)
					print("step %d, train loss %g, train accuracy %g, cur lr %g" % (step, train_loss, train_acc, cur_lr))	  
	
				if step != 0 and step % self.FLAGS.save_interval_steps == 0:
					batch_valid_xs, batch_valid_ys = sess.run([batch_valid_feature, batch_valid_label])
					if not self.FLAGS.one_hot_label:
						batch_valid_ys = np.argmax(batch_valid_ys, 1).reshape(self.FLAGS.valid_batch_size, self.label_nums)
					valid_feed_dict = {features:batch_valid_xs, labels:batch_valid_ys, num_of_epoch:step//self.FLAGS.save_interval_steps}
	
					summary_test, valid_loss, valid_acc, next_lr = sess.run([merged, loss, accuracy, lr], feed_dict=valid_feed_dict)
					test_writer.add_summary(summary_test, step)
	
					print("step %d, valid loss %g, valid accuracy %g, next lr %g" % (step, valid_loss, valid_acc, next_lr))
	
					saver.save(sess, os.path.join(self.FLAGS.model_path, 'model.ckpt'), global_step = step)
	
				sess.run(train_op, feed_dict=train_feed_dict)

				if step == self.FLAGS.save_interval_steps * self.FLAGS.max_epochs:
					break
				step += 1
		except tf.errors.OutOfRangeError:
			pass
		finally:
			coord.request_stop()
		
		coord.join(threads)
		sess.close()

	def build_predict_graph(self):
		self.features = tf.placeholder(tf.float32, [None, self.feature_nums])
		self.labels = self.predict_label(self.features)

	def load(self, model_path, model_name=None):
		sess = tf.Session()
		saver = tf.train.Saver()
		model_path = util.get_model_path(model_path, model_name)
		saver.restore(sess, model_path)
		print("restore model done, path: %s" % model_path)
		self.sess = sess
		self.build_predict_graph()

	def predict(self, features):
		labels = self.sess.run(self.labels, feed_dict={self.features:features})
		return labels

	def close(self):
		self.sess.close()
