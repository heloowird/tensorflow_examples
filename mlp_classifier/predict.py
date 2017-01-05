#coding=gbk

from __future__ import absolute_import
from __future__ import division

import sys
import os

import numpy as np

from classifier_main import *

FLAGS = get_parameters()
model_path = FLAGS.model_path

def init_model(model_path):
	model = get_model()
	model.load(model_path)
	return model

def predict(model, input_file, output_file):
	key_lst = []
	features = None

	# read predict data
	with open(input_file) as f:
		fea_lst = []
		for line in f:
			line = line.strip("\n\r")
			fs = line.split("\t")
			if len(fs) < 2:
				print >>sys.stderr, "error input: %s" % line
				continue

			key = fs[0]
			fea = fs[1].split()
			if len(fea) != 1000:
				print >>sys.stderr, "error feature len: %d" % len(fea)
				continue
			fea = [float(ele) for ele in fea]

			key_lst.append(key)
			fea_lst.append(fea)
			features = np.array(fea_lst)

	# predict
	predict_labels = model.predict(features)
	if len(predict_labels) != len(key_lst):
		print >>sys.stderr, "bad predict nums"

	# write predict result
	with open(output_file, "w") as f:
		for key, label in zip(key_lst, predict_labels):
			f.write("%s\t%f\n" % (key, label))

if __name__ == "__main__":
	model = init_model(model_path)
	predict(model, FLAGS.predict_input_path, FLAGS.predict_output_path)
