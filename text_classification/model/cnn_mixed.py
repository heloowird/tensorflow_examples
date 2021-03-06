import tensorflow as tf


class CNNClassifier(object):
    """
    A CNN model for text classification.
    Network structure: embedding layer > convolution layer > max-pooling > softmax
    """

    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int32, [None, num_classes], name='input_y')
        self.word_embedding = tf.placeholder(tf.float32, [vocab_size, embedding_size], name='embeding')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.NON_STATIC_W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="NON_STATIC_W")
            self.init_non_static_op = tf.assign(self.NON_STATIC_W, self.word_embedding) 

            self.STATIC_W = tf.Variable(
                tf.constant(0.0, shape=[vocab_size, embedding_size]),
                trainable=False, name="STATIC_W")
            self.init_static_op = tf.assign(self.STATIC_W, self.word_embedding) 

            no_static_embdded_chars = tf.nn.embedding_lookup(self.NON_STATIC_W, self.input_x)
            static_embdded_chars = tf.nn.embedding_lookup(self.STATIC_W, self.input_x)
            self.embdded_chars = tf.concat([tf.expand_dims(no_static_embdded_chars, -1), tf.expand_dims(static_embdded_chars, -1)], -1 )

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution layer
                filter_shape = [filter_size, embedding_size, 2, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_conv")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b_conv")
                conv = tf.nn.conv2d(
                    self.embdded_chars,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="pool")

                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # W = tf.get_variable(
            #     "W",
            #     shape=[num_filters_total, num_classes],
            #     initializer=tf.contrib.layers.xavier_initializer())
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
