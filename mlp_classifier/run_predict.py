#!/bin/sh

python predict_main.py --model_path ../trained_model/model.ckpt-16539692 --predict_input_path test_data --predict_output_path test_data.predict_result 1>log_predict 2>&1
