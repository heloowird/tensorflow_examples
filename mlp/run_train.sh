#!/bin/sh

# training with GPU
CUDA_VISIBLE_DEVICES=1 python classifier.py --mode train

# training with CPU
#python classifier.py --mode train
