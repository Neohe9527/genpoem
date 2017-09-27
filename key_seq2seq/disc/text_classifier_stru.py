# This code is mainly borrowed from https://github.com/dennybritz/cnn-text-classification-tf

import tensorflow as tf
#from tensorflow.python.ops import rnn
#from tensorflow.contrib.rnn.python.ops import rnn_cell

def linear(input_, output_size, scope=None):
    '''
    Linear map: output[k] = sum_i(Matrix[k, i] * args[i] ) + Bias[k]
    Args:
        args: a tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
    '''

    shape = input_.get_shape().as_list()
    if len(shape) != 2:
        raise ValueError("Linear is expecting 2D arguments: %s" % str(shape))
    if not shape[1]:
        raise ValueError("Linear expects shape[1] of arguments: %s" % str(shape))
    input_size = shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "SimpleLinear"):
        matrix = tf.get_variable("Matrix", [output_size, input_size], dtype=input_.dtype)
        bias_term = tf.get_variable("Bias", [output_size], dtype=input_.dtype)

    return tf.matmul(input_, tf.transpose(matrix)) + bias_term



# highway layer that borrowed from https://github.com/carpedm20/lstm-char-cnn-tensorflow
def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).

    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    output = input_
    for idx in xrange(layer_size):
        #output = f(rnn_cell._linear(output, size, 0, scope='output_lin_%d' % idx))
        output = f(linear(output, size,  scope='output_lin_%d' % idx))

        #transform_gate = tf.sigmoid(rnn_cell._linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)
        transform_gate = tf.sigmoid(linear(input_, size, scope='transform_lin_%d' % idx) + bias)
        carry_gate = 1. - transform_gate

        output = transform_gate * output + carry_gate * input_

    return output


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0,dropout_keep_prob=0.75 , max_grad_norm= 5,is_training= True, scope_name= "disc_cnn"):
        self.scope_name = scope_name

        # Placeholders for input, output and dropout
        with tf.variable_scope(self.scope_name):
            self.batch_size = tf.Variable(0,dtype=tf.int32,trainable=False)
            self.dropout_keep_prob =dropout_keep_prob
            #Placeholders for input, output and dropout
            self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
            #self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            self.new_batch_size = tf.placeholder(tf.int32,shape=[],name="new_batch_size")
            self._batch_size_update = tf.assign(self.batch_size,self.new_batch_size)

        # Keeping track of l2 regularization loss (optional)
            l2_loss = tf.constant(0.0)

        # Embedding layer
            with tf.device('/cpu:0'), tf.name_scope("embedding"):
                W = tf.Variable(
                        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                        name="W")
                self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
                #self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
                self.embedded_chars_expanded =tf.reshape(self.embedded_chars, [self.batch_size, 8, 4, embedding_size])

        # Create a convolution + maxpool layer for each filter size
            pooled_outputs = []
            for filter_size, num_filter in zip(filter_sizes, num_filters):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                ### conv
                    filter_shape = [filter_size, 4, embedding_size, num_filter]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
                    conv = tf.nn.conv2d(
                            self.embedded_chars_expanded,
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
                            padding='VALID',
                            name="pool")
                    pooled_outputs.append(pooled)

        # Combine all the pooled features
            num_filters_total = sum(num_filters)
            self.h_pool = tf.concat(pooled_outputs,3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Add highway
            with tf.name_scope("highway"):
                self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

        # Add dropout
            with tf.name_scope("dropout"):
                self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
                self.ypred_for_auc = tf.nn.softmax(self.scores)
                self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
                self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
            if not is_training:
                return
         ##add summary
            loss_summary = tf.summary.scalar("loss",self.loss)
            accuracy_summary=tf.summary.scalar("accuracy_summary",self.accuracy)

         ### train 
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.lr = tf.Variable(0.0,trainable=False)

            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_grad_norm)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            #for g, v in zip(grads, tvars):
             #   if g is not None:
              #      grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
               #     sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                #    grad_summaries.append(grad_hist_summary)                    
                 #   grad_summaries.append(sparsity_summary)
            #self.grad_summaries_merged = tf.summary.merge(grad_summaries)
            self.summary =tf.summary.merge([loss_summary,accuracy_summary])
            optimizer = tf.train.AdamOptimizer(self.lr)
            optimizer.apply_gradients(zip(grads, tvars))
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            self.new_lr = tf.placeholder(tf.float32,shape=[],name="new_learning_rate")            
            self._lr_update = tf.assign(self.lr,self.new_lr)
            all_variables = [k for k in tf.global_variables() if self.scope_name in k.name]
            self.saver = tf.train.Saver(all_variables)
    def assign_new_lr(self,session,lr_value):        
        session.run(self._lr_update,feed_dict={self.new_lr:lr_value})
    def assign_new_batch_size(self,session,batch_size_value):
        session.run(self._batch_size_update,feed_dict={self.new_batch_size:batch_size_value})


#def create_model(sess, disc_config):
#
 #   model = TextCNN(
  #          sequence_length = disc_config.seq_length,
   #         num_classes = disc_config.num_classes,
    #        vocab_size = disc_config.vocab_size,
     #       embedding_size = disc_config.dis_embedding_dim,
      #      filter_sizes = disc_config.dis_filter_sizes,
       #     num_filters = disc_config.dis_num_filters,
        #    l2_reg_lambda= disc_config.dis_l2_reg_lambda)
    #ckpt = tf.train.get_checkpoint_state(disc_config.train_dir)
    #if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
     #   print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
      #  model.saver.restore(session, ckpt.model_checkpoint_path)
   # else:
    #    print("Created disc_CNN model with fresh parameters.")
     #   session.run(tf.global_variables_initializer())
   # return model
