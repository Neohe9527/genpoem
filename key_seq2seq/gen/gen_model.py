from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import sys
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import *
from tensorflow.contrib.rnn.python.ops.core_rnn import *
import utils.data_utils as data_utils
import gen.seq2seq as rl_seq2seq
from tensorflow.contrib import rnn 
sys.path.append('../utils')

class Seq2SeqModel(object):

  def __init__(self,
               source_vocab_size, target_vocab_size, buckets, size, num_layers, max_gradient_norm, batch_size,
            learning_rate, learning_rate_decay_factor, use_lstm=False,num_samples = 8000,forward_only=False,
               scope_name='gen_seq2seq', keep_prob=0.5, dtype=tf.float32):
    
    self.scope_name = scope_name
    with tf.variable_scope(self.scope_name):
        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)
        self.batch_size = batch_size
        self.keep_prob = 0.5

        ### for reinforcement learning
        self.up_reward = tf.placeholder(tf.bool, name="up_reward")
        self.en_output_proj = tf.placeholder(tf.bool, name="en_output_proj")
        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None
        # Sampled softmax only makes sense if we sample less than vocabulary size.


        #def policy_gradient(logit, labels):
         #   def softmax(x):
          #      return tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=0)
           # prob = softmax(logit)
            #token = tf.argmax(logit, 0)
           # return tf.reduce_max(prob)
           # pass
        #softmax_loss_function = policy_gradient
       # pass

        if num_samples > 0 and num_samples < self.target_vocab_size:
            w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
            w = tf.transpose(w_t)
            b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
            output_projection = (w, b)

            def sampled_loss(inputs, labels):
                labels = tf.reshape(labels, [-1, 1])
                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                local_w_t = tf.cast(w_t, tf.float32)
                local_b = tf.cast(b, tf.float32)
                local_inputs = tf.cast(inputs, tf.float32)
                return tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,num_samples, self.target_vocab_size)
            softmax_loss_function = sampled_loss

        # softmax_loss_function = control_flow_ops.cond(self.up_reward,
        #                                       lambda:policy_gradient,
        #                                       lambda:sampled_loss)
        #softmax_loss_function = policy_gradient

        #loss_function = tf.select(self.up_reward, policy_gradient, softmax_loss_function)

        #softmax_loss_function = loss_function
        # Create the internal multi-layer cell for our RNN.
        emb_dim = 512
        encode_cell = GRUCell(size)
        decode_cell = GRUCell(size)
        #bw_cell = GRUCell(size)
        if use_lstm:
          print ("use lstm")
          encode_cell = BasicLSTMCell(size)
          decode_cell = BasicLSTMCell(size)
          #bw_cell = BasicLSTMCell(size)
        #encode_cell = DropoutWrapper(encode_cell, self.keep_prob)
        #decode_cell = DropoutWrapper(decode_cell, self.keep_prob)
        #cell = single_cell
        #b_cell = bw_cell
        if num_layers > 1:
          encode_cell = MultiRNNCell([encode_cell] * num_layers)
          #decode_cell = MultiRNNCell([decode_cell] * num_layers)
          #b_cell = MultiRNNCell([b_cell] * num_layers)
        cell = []
        cell.append(encode_cell)
        cell.append(decode_cell)
        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
          return rl_seq2seq.embedding_attention_seq2seq(
              encoder_inputs,
              decoder_inputs,
              cell,
              num_encoder_symbols=source_vocab_size,
              num_decoder_symbols=target_vocab_size,
              embedding_size=emb_dim,
              output_projection=output_projection,
              feed_previous=do_decode,
              dtype=dtype)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
          self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="encoder{0}".format(i)))
        for i in xrange(buckets[-1][1] + 1):
          self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                    name="decoder{0}".format(i)))
          self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                    name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i+1]
                   for i in xrange(len(self.decoder_inputs)-1)]

        # Training outputs and losses.
        if forward_only:
            print ("forward_only")
            self.outputs, self.losses, self.encoder_state = rl_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
                softmax_loss_function=softmax_loss_function)
            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in xrange(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses, self.encoder_state = rl_seq2seq.model_with_buckets(
                    self.encoder_inputs, self.decoder_inputs, targets,
                    self.target_weights, buckets,
                    lambda x, y: seq2seq_f(x, y, False),
                    softmax_loss_function=softmax_loss_function)

        # Gradients and SGD update operation for training the model.
        self.tvars = tf.trainable_variables()
        #if not forward_only:
        self.gradient_norms = []
        self.updates = []
        self.reward = [tf.placeholder(tf.float32, name="reward_%i" % i) for i in range(len(buckets))]
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        for b in xrange(len(buckets)):
            adjusted_losses = tf.multiply(self.losses[b], self.reward[b])
            gradients = tf.gradients(adjusted_losses, self.tvars)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
            self.gradient_norms.append(norm)
            self.updates.append(opt.apply_gradients(
                zip(clipped_gradients, self.tvars), global_step=self.global_step))

        # self.saver = tf.train.Saver(tf.all_variables())
        all_variables = [k for k in tf.global_variables() if k.name.startswith(self.scope_name)]
        self.saver = tf.train.Saver(all_variables, max_to_keep=1000)

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only=True, up_reward=False, reward=None, debug=True):
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {
      self.up_reward.name:  up_reward,
      self.en_output_proj:        (forward_only),
    }
    for l in xrange(len(self.buckets)):
      input_feed[self.reward[l].name] = reward if reward else 1
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only: # normal training
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else: # testing or reinforcement learning
      output_feed = [self.encoder_state[bucket_id], self.losses[bucket_id]]  # Loss for this batch.
      #output_feed = [ self.losses[bucket_id]]
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return outputs[0], outputs[1], outputs[2:]  # encoder_state, loss, outputs.
  def prepare_keydata(self,seqs_x, seqs_y, encoder_size, decoder_size, max_keyword_len=4):
    # x: a list of sentences
    n_samples = len(seqs_x)
    maxlen_x = encoder_size
    ####with go
    maxlen_y = decoder_size + 1
    print("pre key data")

    x = np.zeros((maxlen_x, n_samples)).astype('int32')
    y = np.zeros((maxlen_y, n_samples)).astype('int32')
    x_mask = np.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = np.zeros((maxlen_y, n_samples)).astype('float32')
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        split_idx = s_x.index(4)
        keyword = s_x[:split_idx+1]
        content = s_x[split_idx+1:]
        x[:len(keyword), idx] = keyword
        x_mask[:len(keyword), idx] = 1.
        if len(content)>=1:
            x[max_keyword_len:max_keyword_len+len(content),idx] = content
            x_mask[max_keyword_len:max_keyword_len+len(content),idx] = 1.
        y[:len(s_y), idx] = s_y
        y_mask[:len(s_y), idx] = 1.

    return x, x_mask, y, y_mask
  def get_batch(self, train_data, bucket_id, max_keyword_len=4,type=0):

    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []
    batch_size = self.batch_size
    #print("Batch_Size: %s" %self.batch_size)
    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    batch_source_encoder, batch_source_decoder, batch_decoder_size = [], [], []
    #print("bucket_id: %s" %bucket_id)
    #print (train_data[bucket_id])
    if type == 1:
        batch_size = 1
    for batch_i in xrange(batch_size):
      if type == 1:
        encoder_input, decoder_input= train_data[bucket_id][0]
      elif type == 2:
        #print("data[bucket_id]: ", data[bucket_id][0])
        #print (train_data)
        encoder_input, decoder_input= train_data[bucket_id][batch_i]
        print("train en: %s, de: %s" %(encoder_input, decoder_input))
        #encoder_input = encoder_input_a[batch_i]
      elif type == 0:
          encoder_input, decoder_input = random.choice(train_data[bucket_id])
      #batch_decoder_size.append(len(decoder_input)+1)
          print("train en: %s, de: %s" %(encoder_input, decoder_input))

      batch_source_encoder.append(encoder_input)
      batch_source_decoder.append(decoder_input)
      #print (decoder_input)
      batch_decoder_size.append(len(decoder_input)+1)
      # Encoder inputs are padded and then reversed.
      encoder_inputs.append(list(encoder_input))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_inputs.append([data_utils.GO_ID] + decoder_input)
      #decoder_inputs.append(decoder_input)
    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
    x, x_mask, y, y_mask = self.prepare_keydata(encoder_inputs,decoder_inputs, encoder_size, decoder_size, 4)
    #print(batch_decoder_size)

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      #print (encoder_size)
      #print (x.shape)
      batch_encoder_inputs.append(
          np.array([x[length_idx,batch_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([y[length_idx,batch_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(batch_size, dtype=np.float32)
      for batch_idx in xrange(batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = y[length_idx+1,batch_idx]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return (batch_encoder_inputs, batch_decoder_inputs, batch_weights, batch_source_encoder, batch_source_decoder, batch_decoder_size)
