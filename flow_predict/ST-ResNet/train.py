#!/usr/bin/env python
#coding:utf-8
#author:zhujianqi

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
import resnet_model

_CLOSENESS_LEN = 3
_PRERIOD_LEN = 1
_TREND_LEN = 1
_EXTERNAL_LEN = 26

_INPUT_CHANNEL = 2
_HEIGHT = 177
_WIDTH = 212

_OUTPUT_CHANNEL = 1
_OUTPUT_SIZE = [_OUTPUT_CHANNEL, _HEIGHT, _WIDTH]

class DemandPredictModel(resnet_model.STResModel):
    def __init__(self, resnet_size, data_format='channels_first', output_size=_OUTPUT_SIZE,
               resnet_version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE):
        if resnet_size % 6 != 2:
            raise ValueError('resnet_size must be 6n + 2:', resnet_size)

        num_blocks = (resnet_size - 2) // 6

        super(DemandPredictModel, self).__init__(
            resnet_size=resnet_size,
            bottleneck=False,
            output_size=output_size,
            num_filters=64,
            kernel_size=3,
            conv_stride=1,
            first_pool_size=None,
            first_pool_stride=None,
            block_sizes=[num_blocks] * 1,
            block_strides=[1],
            final_size=0,
            resnet_version=resnet_version,
            data_format=data_format,
            dtype=dtype
        )

def parse_record_fn(value, is_training, dtype):
    features = {
        'closeness': tf.FixedLenFeature([], tf.string),
        'preriod': tf.FixedLenFeature([], tf.string),
        'trend': tf.FixedLenFeature([], tf.string),
        'external': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.string)
    }
    parsed_example = tf.parse_single_example(serialized=value, features=features)

    closeness = tf.decode_raw(parsed_example['closeness'], dtype)
    #closeness = tf.reshape(closeness, [_INPUT_CHANNEL*_CLOSENESS_LEN, _HEIGHT, _WIDTH])

    preriod = tf.decode_raw(parsed_example['preriod'], dtype)
    #preriod = tf.reshape(preriod, [_INPUT_CHANNEL*_PRERIOD_LEN, _HEIGHT, _WIDTH])

    trend = tf.decode_raw(parsed_example['trend'], dtype)
    #trend = tf.reshape(trend, [_INPUT_CHANNEL*_TREND_LEN, _HEIGHT, _WIDTH])

    external = tf.decode_raw(parsed_example['external'], dtype)
    #external = tf.reshape(external, [_EXTERNAL_LEN]) 

    label = tf.decode_raw(parsed_example['label'], dtype)
    label = tf.reshape(label, [_INPUT_CHANNEL, _HEIGHT, _WIDTH])
    label = label[0]
    return closeness, preriod, trend, external, label

def get_filenames(data_path, data_name):
    assert os.path.exists(data_path), ('data directory not found')

    tfrecord_valid_pattern = os.path.join(data_path, '%s' % data_name)
    files = tf.gfile.Glob(tfrecord_valid_pattern)
    return files

def input_fn(data_dir, data_name, is_training, dtype, batch_size=32, buffer_size=512, repeat_num=-1):
    filenames = get_filenames(data_dir, data_name)
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    dataset = dataset.map(lambda value: parse_record_fn(value, is_training, dtype))

    if is_training:
        dataset = dataset.shuffle(buffer_size=buffer_size)

    dataset = dataset.repeat(repeat_num)
    dataset = dataset.batch(batch_size=batch_size)

    iterator = dataset.make_one_shot_iterator()
    closeness_batch, preriod_batch, trend_batch, external_batch, label_batch = iterator.get_next()

    x = {'closeness': closeness_batch,
         'preriod': preriod_batch,
         'trend': trend_batch,
         'external': external_batch}
    y = label_batch
    return x, y

def demand_perdict_model_fn(features, labels, mode, params):
    model = DemandPredictModel(resnet_size=params['resnet_size'], 
                        resnet_version=params['resnet_version'], dtype=params['dtype'])

    sizes = (_CLOSENESS_LEN, _PRERIOD_LEN, _TREND_LEN, _EXTERNAL_LEN, _INPUT_CHANNEL, _HEIGHT, _WIDTH)
    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN, sizes)
    logits = tf.cast(logits, params['dtype'])

    predictions = {
        'logits': logits,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate loss, which includes mse loss and L2 regularization.
    labels = tf.reshape(labels, [-1, _OUTPUT_CHANNEL, _HEIGHT, _WIDTH])
    labels = tf.cast(labels, params['dtype'])
    #mse_loss = tf.losses.mean_squared_error(labels=labels, predictions=logits, 
    #            reduction=tf.losses.Reduction.MEAN)

    label_sum = tf.reduce_sum(labels)
    logit_sum = tf.reduce_sum(logits)
    tf.summary.scalar('label_sum', label_sum)
    tf.summary.scalar('logit_sum', logit_sum)

    sq_diff = tf.square(tf.subtract(labels, logits))
    sq_diff_sum = tf.reduce_sum(sq_diff)
    tf.summary.scalar('sq_diff_sum', sq_diff_sum)

    mse_loss = tf.reduce_mean(sq_diff)
    # Create a tensor named mse_loss for logging purposes.
    tf.identity(mse_loss, name='mse_loss')
    tf.summary.scalar('mse_loss', mse_loss)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def exclude_batch_norm(name):
        return 'batch_normalization' not in name
    loss_filter_fn = params['loss_filter_fn'] or exclude_batch_norm

    # Add weight decay to the loss.
    l2_loss = params['weight_decay'] * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()
        if loss_filter_fn(v.name)])
    tf.summary.scalar('l2_loss', l2_loss)
    loss = mse_loss + l2_loss
    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        #learning_rate = learning_rate_fn(global_step)

        # Create a tensor named learning_rate for logging purposes
        #tf.identity(learning_rate, name='learning_rate')
        #tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(
                learning_rate=params['learning_rate'])

        if params['loss_scale'] != 1:
            # When computing fp16 gradients, often intermediate tensor values are
            # so small, they underflow to 0. To avoid this, we multiply the loss by
            # loss_scale to make these tensor values loss_scale times bigger.
            scaled_grad_vars = optimizer.compute_gradients(loss * params['loss_scale'])
            #unscaled_grad_vars = [(grad / loss_scale, var)
            #                    for grad, var in scaled_grad_vars]
            #gradients, variables = zip(*unscaled_grad_vars)
            #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            cliped_scaled_grad_vars = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in scaled_grad_vars]
            minimize_op = optimizer.apply_gradients(cliped_scaled_grad_vars, global_step)
        else:
            grad_vars = optimizer.compute_gradients(loss)
            #gradients, variables = zip(*grad_vars)
            #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
            cliped_grad_vars = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in grad_vars]
            minimize_op = optimizer.apply_gradients(cliped_grad_vars, global_step)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
    else:
        train_op = None

    mse = tf.metrics.mean_squared_error(labels, logits)
    mae = tf.metrics.mean_absolute_error(labels, logits)

    metrics = {'mse': mse, 'mae': mae}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(mse[1], name='train_mse')
    tf.identity(mae[1], name='train_mae')
    tf.summary.scalar('train_mse', mse[1])
    tf.summary.scalar('train_mae', mae[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)

    
def main(argv):
    train_data_dir = ""
    train_data_name = ""
    test_data_dir = ""
    test_data_name = ""

    model_dir=""

    session_config = tf.ConfigProto(allow_soft_placement=True)
    run_config = tf.estimator.RunConfig(
        save_checkpoints_steps=100,
        session_config=session_config)

    def train_input_fn():
        return input_fn(train_data_dir, train_data_name, is_training=True, dtype=tf.float32)

    def test_input_fn():
        return input_fn(test_data_dir, test_data_name, is_training=False, dtype=tf.float32)

    model = tf.estimator.Estimator(model_fn=demand_perdict_model_fn, model_dir=model_dir,
                                    config=run_config,
                                    params={'resnet_size':20,
                                            'resnet_version':2,
                                            'loss_filter_fn':None,
                                            'weight_decay':2e-3,
                                            'loss_scale':1000,
                                            'learning_rate':0.01,
                                            'dtype':tf.float32
                                    })

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=1200)
    eval_spec = tf.estimator.EvalSpec(input_fn=test_input_fn, steps=3)
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

