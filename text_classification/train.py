#from __future__ import print_function
import sys
#reload(sys)
#sys.setdefaultencoding('utf-8') 
import tensorflow as tf
import numpy as np
import time
import datetime
import os
import data_helper
from cnn_lstm import CNN_LSTM

# Parameters
# ==================================================
flags = tf.app.flags
FLAGS = flags.FLAGS

# Model Hyperparameters
flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '3,5,8')")
flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
flags.DEFINE_integer("batch_size", 256, "Batch Size (default: 64)")
flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
flags.DEFINE_integer("evaluate_every", 1000, "Evaluate model on dev set after this many steps (default: 100)")
flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 100)")
flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
flags.DEFINE_string("model_path", "./model_v3/", "Path to save model")
flags.DEFINE_string("summary_path", "lstm_random_v3_summaries", "Path to write summary")

# Misc Parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

def main(agrv):
    # Data Preparation
    # ==================================================
    # Load data
    train_file = "../data_v3/v3_train"
    test_file  = "../data_v3/v3_test"
    feature_file = "../data_v3/w2v.voc"
    label_file = "../data_v3/class_2_dict_v3"
    #w2v_file = "../data_v2/poi_name_w2v.npy"
    
    x_train, y_train, vocabulary, vocabulary_inv, _, _ = data_helper.build_input_data(train_file, feature_file, label_file)
    x_dev, y_dev, _, _, _, _ = data_helper.build_input_data(test_file, feature_file, label_file)
    print("Vocabulary Size: {:d}".format(len(vocabulary)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    
    #pretrained_w2v = np.load(w2v_file)
    
    # Training
    # ==================================================
    
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            lstm = CNN_LSTM(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocabulary),
                embedding_size=FLAGS.embedding_dim,
                l2_reg_lambda=FLAGS.l2_reg_lambda)
    
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(lstm.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    
            # Keep track of gradient values and sparsity (optional)
            #grad_summaries = []
            #for g, v in grads_and_vars:
            #    if g is not None:
            #        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            #        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            #        grad_summaries.append(grad_hist_summary)
            #        grad_summaries.append(sparsity_summary)
            #grad_summaries_merged = tf.summary.merge(grad_summaries)
    
            # Output directory for models and summaries
            print("Writing model to {}\n".format(FLAGS.model_path))
            print("Writing summary to {}\n".format(FLAGS.summary_path))
    
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", lstm.loss)
            acc_summary = tf.summary.scalar("accuracy", lstm.accuracy)
    
            # Train Summaries
            #train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(FLAGS.summary_path, "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    
            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(FLAGS.summary_path, "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
    
            # Visualization for embedding
            # Write meta
            model_dir = os.path.abspath(FLAGS.model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            with open(os.path.join(model_dir, "metadata.tsv"), 'wt', encoding='utf-8') as tsv_file:
                for vocab in vocabulary_inv:
                    #tsv_file.write("%s\n" % vocab.decode('utf-8'))
                    tsv_file.write("%s\n" % vocab)
    
            config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = lstm.W.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = os.path.join(model_dir, 'metadata.tsv')
            tf.contrib.tensorboard.plugins.projector.visualize_embeddings(tf.summary.FileWriter(dev_summary_dir), config)
    
            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_prefix = os.path.join(model_dir, "model")
            saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and ckpt.model_checkpoint_path:
                print("load existing model from: %s" % ckpt.model_checkpoint_path)
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("train from restored model: %s" % ckpt.model_checkpoint_path)
            else:
                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                # Initialize word embedding 
                #sess.run([lstm.init_op], feed_dict={lstm.word_embedding: pretrained_w2v}) 
    
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    lstm.input_x: x_batch,
                    lstm.input_y: y_batch,
                    lstm.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, lstm.loss, lstm.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)
    
    
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    lstm.input_x: x_batch,
                    lstm.input_y: y_batch,
                    lstm.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, lstm.loss, lstm.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
    
    
            batches = data_helper.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
    
            dev_batches = data_helper.batch_iter(
                list(zip(x_dev, y_dev)), FLAGS.batch_size, FLAGS.num_epochs)
    
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_batch = next(dev_batches)
                    x_dev_batch, y_dev_batch = zip(*dev_batch)
                    dev_step(x_dev_batch, y_dev_batch, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
    
                    # Save the model for Embedding Visualization
                    saver.save(sess, os.path.join(dev_summary_dir, "model.ckpt"), global_step=current_step)
    
                    print("Saved model checkpoint to {}\n".format(path))

if __name__ == '__main__':
    tf.app.run()
