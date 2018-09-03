#coding=utf-8

import os
import sys

import numpy as np
import tensorflow as tf

class data_util():
	def __init__(self, FLAGS, name="train"):
		self.FLAGS = FLAGS
		if name == "train":
			self.filenames = self.init_filenames(self.FLAGS.train_data_path, self.FLAGS.data_name)	
		elif name == "valid":
			self.filenames = self.init_filenames(self.FLAGS.validate_data_path, self.FLAGS.data_name)	
		else:
			print >>sys.stderr, "init data_util failed"

	def init_filenames(self, data_path, data_name):
		tf_record_valid_pattern = os.path.join(data_path, '%s-*' % data_name)
		files = tf.gfile.Glob(tf_record_valid_pattern)
		return files

	def decode_read(self, filename_queue):
		reader = tf.TFRecordReader()
		key, record = reader.read(filename_queue)
		feature_data = tf.parse_single_example(
			record,
			features={
				'feat': tf.FixedLenFeature([], tf.string),
				'label': tf.FixedLenFeature([], tf.string)
				}
			)
	
		feat = tf.decode_raw(feature_data['feat'], tf.float32) 
		feat.set_shape([self.FLAGS.input_feature_dims])
	
		label = tf.decode_raw(feature_data['label'], tf.float32)
		label.set_shape([self.FLAGS.input_label_dims])
	
		return feat, label
	
	def gen_batch(self, batch_size, num_epochs=None):
		filename_queue = tf.train.string_input_producer(self.filenames, num_epochs=num_epochs, shuffle=True)
		feature, label = self.decode_read(filename_queue)
	
		min_after_dequeue = 100000
		capacity = min_after_dequeue + 3 * batch_size
		batch_feature, batch_label = tf.train.shuffle_batch(
			[feature, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue, num_threads=5)
	
		return batch_feature, batch_label

def get_model_path(model_path, model_name=None):
	ckpt = tf.train.get_checkpoint_state(model_path)
	if ckpt and ckpt.model_checkpoint_path:
		model_path = ckpt.model_checkpoint_path if model_name is None else os.path.join(model_path, model_name)
	return model_path

if __name__ == "__main__":
	# test tensorflow read
	flags = tf.app.flags
	FLAGS = flags.FLAGS
	flags.DEFINE_integer('input_label_dims', 2, 'numbers of labels')
	flags.DEFINE_integer('input_feature_dims', 1000, 'numbers of features')
	flags.DEFINE_string('train_data_path', './test_tf_data/', 'directory of train data')
	flags.DEFINE_string('train_data_name', 'part', 'prefix of train data')
	data_utiler = data_util(FLAGS, "train")
	batch_feature, batch_label = data_utiler.gen_batch(20)

	sess = tf.InteractiveSession()
	sess.run(tf.initialize_all_variables())
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	try:
		step = 1
		while not coord.should_stop():
			batch_xs, batch_ys = sess.run([batch_feature, batch_label])
			batch_ys = np.argmax(batch_ys, 1).reshape(20, 1)
			print batch_xs.shape
			print batch_xs[0:5]
			print batch_ys.shape
			print batch_ys[0:5]
			if step == 1:
				break
			step += 1
	except tf.errors.OutOfRangeError:
		print "done"
	finally:
		coord.request_stop()
	
	coord.join(threads)
	sess.close()
