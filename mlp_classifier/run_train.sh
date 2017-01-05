#!/bin/sh

# training with GPU
CUDA_VISIBLE_DEVICES=1 python classifier_main.py --mode train

# training with CPU
#python classifier_main.py --mode train
