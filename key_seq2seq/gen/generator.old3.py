from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import pickle
import heapq
import tensorflow.python.platform

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import utils.data_utils as data_utils
import utils.conf as conf
import gen.gen_model as seq2seq_model
import copy
from tensorflow.python.platform import gfile
sys.path.append('../utils')


# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(8,8),(12, 8), (20, 8), (28, 8)]
def read_data(source_path, max_size=None):
    data_set = [[] for _ in _buckets]
    print (source_path)
    source_data_set = []
    with gfile.GFile(source_path, mode= 'r') as source_file:
        pair = source_file.readline()
        counter = 0
        while pair and (not max_size or counter<max_size):
            pair = pair.strip().split('\t')
            source, target = pair[0].strip(), pair[1].strip()
            counter += 1
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            source_ids = [int(x) for x in source.split(' ')]        
            target_ids = [int(x) for x in target.split(' ')]
            len_target_ids = len(target_ids)
            #target_ids.append((data_utils.EOS_ID))
            source_data_set.append([source_ids, target_ids])
            for bucket_id, (source_size, target_size) in enumerate(_buckets): #[bucket_id, (source_size, target_size)]         
                if len(source_ids) <= source_size and len(target_ids) <= target_size:            
                    #source_ids.extend(len_target_ids)
                    data_set[bucket_id].append([source_ids, target_ids])            
                    break
            pair = source_file.readline()
    return data_set, source_data_set
def get_vocab(vocab_path):
    rev_vocab=[]
    vocab={}
    fr = open(vocab_path, 'r')
    for line in fr:
        word, _id = line.strip().decode('utf-8').split('\t')
        word= word.strip()
        _id = _id.strip()
        vocab[word]= int(_id)
        rev_vocab.append(word)
    print("get vocabu sucess!!")
    return vocab,  rev_vocab

def prepare_data(gen_config):    
    vocab_path = os.path.join(gen_config.data_dir, "vocab%d.all" % gen_config.vocab_size)    
    vocab, rev_vocab = get_vocab(vocab_path)    #for each in vocab:     #   print each, vocab[each]    
    print ("get vocab!")
    train_ids_path, dev_ids_path = data_utils.prepare_chitchat_data(gen_config.data_dir, vocab, gen_config.vocab_size)
    #gen_config.train_path = train_ids_path
    #gen_config.eval_path = dev_ids_path
    #print (train_ids_path, dev_ids_path)
    print ("Reading development and training data (limit: %d)."  % gen_config.max_train_data_size)
    train_set, _ = read_data(train_ids_path)    
    dev_set, _ = read_data(dev_ids_path)    
    return vocab, rev_vocab, train_set, dev_set, train_ids_path, dev_ids_path

def create_model(session, gen_config, forward_only):
    """Create translation model and initialize or load parameters in session."""
    #_buckets = [(8,decode_length),(12, decode_length), (20, decode_length), (28, decode_length)]
    gen_config.buckets = _buckets
    model = seq2seq_model.Seq2SeqModel(
      gen_config.vocab_size, gen_config.vocab_size, _buckets,
      gen_config.size, gen_config.num_layers, gen_config.max_gradient_norm, gen_config.batch_size,
      gen_config.learning_rate, gen_config.learning_rate_decay_factor, forward_only=forward_only)

    print (gen_config.train_dir)

    ckpt = tf.train.get_checkpoint_state(gen_config.train_dir)

    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created Gen_RNN model with fresh parameters.")

        #inject_pretrained_word2vec(sess, gen_config.word2vec_path, dict_dir, gen_config.vocab_size, gen_config.vocab_size)   
        session.run(tf.global_variables_initializer())
        vocab_path = os.path.join(gen_config.data_dir, "vocab%d.all" % gen_config.vocab_size)
        inject_pretrained_word2vec(session, gen_config.word2vec_path, vocab_path, gen_config.vocab_size, gen_config.vocab_size)
    return model

def softmax(x):
    prob = np.exp(x) / np.sum(np.exp(x), axis=0)
    return prob
def inject_pretrained_word2vec(session, word2vec_path, vocab_path, source_vocab_size, target_vocab_size):
  #SOURCE_EMBEDDING_KEY = "gen_seq2seq/embedding_attention_seq2seq/rnn/embedding_wrapper/embedding"
  #TARGET_EMBEDDING_KEY = "gen_seq2seq/embedding_attention_seq2seq/embedding_attention_decoder/embedding"
  SOURCE_EMBEDDING_KEY_1 = "gen_seq2seq/embedding_attention_seq2seq/keywords/fw/embedding_wrapper/embedding"
  SOURCE_EMBEDDING_KEY_2 ="gen_seq2seq/embedding_attention_seq2seq/keywords/bw/embedding_wrapper/embedding"
  SOURCE_EMBEDDING_KEY_3 = "gen_seq2seq/embedding_attention_seq2seq/text/fw/embedding_wrapper/embedding"
  SOURCE_EMBEDDING_KEY_4 ="gen_seq2seq/embedding_attention_seq2seq/text/bw/embedding_wrapper/embedding"
  TARGET_EMBEDDING_KEY = "gen_seq2seq/embedding_attention_seq2seq/embedding_attention_decoder/embedding"
  word2vec_model = {}
  with gfile.GFile(word2vec_path, mode="r") as word2vec_file:
      line = word2vec_file.readline().split('\t')
      word = line[0].strip()
      emb = line[1].strip()
      word2vec_model[word] = emb
  print("w2v model created!")
  session.run(tf.global_variables_initializer())

  assign_w2v_pretrained_vectors(session, word2vec_model, SOURCE_EMBEDDING_KEY_1, vocab_path, source_vocab_size)
  assign_w2v_pretrained_vectors(session, word2vec_model, SOURCE_EMBEDDING_KEY_2, vocab_path, source_vocab_size)
  assign_w2v_pretrained_vectors(session, word2vec_model, SOURCE_EMBEDDING_KEY_3, vocab_path, source_vocab_size)
  assign_w2v_pretrained_vectors(session, word2vec_model, SOURCE_EMBEDDING_KEY_4, vocab_path, source_vocab_size)
  assign_w2v_pretrained_vectors(session, word2vec_model, TARGET_EMBEDDING_KEY, vocab_path, target_vocab_size)


def assign_w2v_pretrained_vectors(session, word2vec_model, embedding_key, vocab_path, vocab_size):

  vectors_variable = [v for v in tf.trainable_variables() if embedding_key in v.name]

  if len(vectors_variable) != 1:
      print(len(vectors_variable))
      print("Word vector variable not found or too many key: " + embedding_key)
      print("Existing embedding trainable variables:")
      print([v.name for v in tf.trainable_variables() if "embedding" in v.name])
      sys.exit(1)

  vectors_variable = vectors_variable[0]
  vectors = vectors_variable.eval()

  with gfile.GFile(vocab_path, mode="r") as vocab_file:
      counter = 0
      while counter < vocab_size:
          line = vocab_file.readline().strip().split('\t')
          vocab_w = line[0].strip()
          _id = int(line[1].strip())
          # for each word in vocabulary check if w2v vector exist and inject.
          # otherwise dont change the value.
          if word2vec_model.has_key(vocab_w):
              w2w_word_vector = [float(each) for each in (word2vec_model[vocab_w].split(' '))]
              vectors[_id] = w2w_word_vector
          else:
              vectors[_id] = w2w_word_vector
          counter += 1

  session.run([vectors_variable.initializer],
            {vectors_variable.initializer.inputs[1]: vectors})
  print ("intit the model with word2vec done")
def train(gen_config):
    """Train a en->fr translation model using WMT data."""
    vocab, rev_vocab, train_set, dev_set, _ , _ = prepare_data(gen_config)
    print("train_set: %d dev_set %d" % (len(train_set),len(dev_set)))

    with tf.Session() as sess:
    #with tf.device("/gpu:1"):
        # Create model.
        print("Creating %d layers of %d units." % (gen_config.num_layers, gen_config.size))
        model = create_model(sess, gen_config,False)
        ###initiate the emb with word2vec
        #inject_pretrained_word2vec(sess, gen_config.word2vec_path, dict_dir, gen_config.vocab_size, gen_config.vocab_size)

        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))
        dev_bucket_sizes = [len(dev_set[b]) for b in xrange(len(_buckets))]
        dev_total_size = int(sum(dev_bucket_sizes))
        print ("total size %d"  %train_total_size)
        print ("total dev size %d"  %dev_total_size)
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        print(train_buckets_scale)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []

        #step_loss_summary = tf.Summary()
        #merge = tf.merge_all_summaries()
        #writer = tf.summary.FileWriter("../logs/", sess.graph)
        train_file_path = os.path.join(gen_config.train_log, "train.loss")
        f_train = gfile.GFile(train_file_path, mode="wb")

        eval_file_path = os.path.join(gen_config.eval_log, "eval.loss")
        f_eval = gfile.GFile(eval_file_path, mode="wb")
        dev_sum_epoch = int(dev_total_size/gen_config.batch_size/len(_buckets))
        print("dev_sum_epoch %d\n" %dev_sum_epoch)


        #for iter in xrange(200*gen_config.steps_per_checkpoint):
         #   random_number_01 = np.random.random_sample()
          #  bucket_id = min([i for i in xrange(len(train_buckets_scale))
           #                 if train_buckets_scale[i] > random_number_01])
           # print(bucket_id)
        for iter in xrange(80*gen_config.steps_per_checkpoint):
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights, batch_source_encoder, batch_source_decoder,_ = model.get_batch(
                train_set, bucket_id, 4, 0)

            _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, forward_only=False)

            step_time += (time.time() - start_time) / gen_config.steps_per_checkpoint
            loss += step_loss / gen_config.steps_per_checkpoint
            current_step += 1
            # Once in a while, we save checkpoint, print statistics, and run evals.

            if current_step % gen_config.steps_per_checkpoint == 0:

                #bucket_value = step_loss_summary.value.add()
                #bucket_value.tag = "loss"
                #bucket_value.simple_value = float(loss)
                #writer.add_summary(step_loss_summary, current_step)
                #print ("train_loss\t"(float(loss))+'\t'+current_step)
                print ("train_loss \t %f current_step %d\n" %(float(loss), current_step))
                f_train.write(str(float(loss))+'\t'+str(current_step)+'\n')

                # Print statistics for the previous epoch.
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity "
                       "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                 step_time, perplexity))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(gen_config.train_dir, "poem_ancient.symbol.model")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                #dev_sum_epoch = int(dev_total_size/gen_config.batch_size/len(_buckets))
                #print("dev_sum_epoch %d\n" %dev_sum_epoch)
                sum_eval_ppx = 0.0
                sum_eval_loss = 0.0
                for eval_epoch in xrange(dev_sum_epoch):
                    for bucket_id in xrange(len(_buckets)):
                        encoder_inputs, decoder_inputs, target_weights, _ , _, _ = model.get_batch(dev_set, bucket_id, 4, 0)
                        _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                target_weights, bucket_id, True)
                    #print (eval_loss)

                        eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                        sum_eval_ppx += eval_ppx
                        sum_eval_loss += eval_loss
                        #print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
                f_eval.write(str(float(sum_eval_loss)/(dev_sum_epoch*len(_buckets)))+'\t'+str(current_step)+'\n')
                #print ("eval_loss"+'\t'+(float(sum_eval_loss))+'\t'+str(current_step))
                avg_eval_loss = float(sum_eval_loss)/(dev_sum_epoch*len(_buckets))
                print ("eval_loss \t %f current_step %d\n" %(avg_eval_loss,current_step))
                sys.stdout.flush()

def get_predicted_sentence(sess, input_token_ids, vocab, model,beam_size,buckets, mc_search=True,debug=True,history_alpha=5, normalize=True,dump_remaining=True):
   def model_step(enc_inp, dec_inp, dptr, target_weights, bucket_id):
        #model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
       _, _, logits = model.step(sess, enc_inp, dec_inp, target_weights, bucket_id, True)
      
       prob = softmax(logits[dptr][0])
       print("prob")
       print(prob)
      # print("model_step @ %s" % (datetime.now()))
       return prob
   def greedy_dec(output_logits):
       selected_token_ids=[]
       print(output_logits)
       print(len(output_logits))
       selected_token_ids = [(np.argmax(logit, axis=1)) for logit in output_logits]
       print (selected_token_ids)
      #if data_utils.EOS_ID in selected_token_ids:
          #eos = selected_token_ids.index(data_utils.EOS_ID)
          #selected_token_ids = selected_token_ids[:eos]
      #output_sentence = ' '.join([rev_vocab[t] for t in selected_token_ids])
       return selected_token_ids
   #if beam_size == 1 or (not mc_search):
    #    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
     #   return [{"dec_inp": greedy_dec(output_logits), 'prob': 1}]	
        
   sample = []
   sample_score = []
   live_hyp = 1
   dead_hyp = 0
   hyp_samples = [[]] * live_hyp
   hyp_scores = np.zeros(live_hyp).astype('float32')
   # Get a 1-element batch to feed the sentence to the model
   len_in = len(input_token_ids)
   outputs =[[]]*len_in
   form_inputs=[]
   for _in, _out in zip(input_token_ids, outputs):
       form_inputs.append((_in, _out))
   bucket_id = min([b for b in range(len(buckets)) if buckets[b][0] >= len(input_token_ids[0])])
   feed_data = {bucket_id: form_inputs}
        
   print('get batch')
   encoder_inputs, decoder_inputs, target_weights, _, _, _ = model.get_batch(feed_data, bucket_id, 4, 2)
   print('get_bacth done')
   if debug: print("\n[get_batch]\n", encoder_inputs, decoder_inputs, target_weights)
   if beam_size == 1 or (not mc_search):
       _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
       return [{"dec_inp": greedy_dec(output_logits), 'prob': 1}]

   # we must retrieve the last state to feed the decoder run
   for ii in xrange(len(decoder_inputs)-1):
       all_prob_ts= model_step(encoder_inputs, decoder_inputs, ii, target_weights, bucket_id)
       next_p=all_prob_ts
       print(next_p)
       cand_scores = next_p
       cand_flat = cand_scores.flatten()
       print(len(cand_flat))
       ranks_flat = (cand_flat.argsort()[:(beam_size - dead_hyp)]).astype("int64")
       print(ranks_flat)
       voc_size = 7938
       print(voc_size)
       trans_indices = (ranks_flat / voc_size).astype("int64")
       print(trans_indices)
       word_indices = (ranks_flat % voc_size).astype("int64")
       print(word_indices)
       costs = cand_flat[ranks_flat]
       new_hyp_samples = []
       new_hyp_scores = np.zeros(beam_size - dead_hyp).astype('float32')
       for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
           new_hyp_samples.append(hyp_samples[ti] + [wi])
           new_hyp_scores[idx] = copy.copy(costs[ti])

            # check the finished samples
       new_live_k = 0
       hyp_samples = []
       hyp_scores = []
       for idx in xrange(len(new_hyp_samples)):
           if new_hyp_samples[idx][-1] == data_utils.EOS_ID:
               sample.append(new_hyp_samples[idx])
               sample_score.append(new_hyp_scores[idx])
               dead_hyp += 1
           else:
               new_live_k += 1
               hyp_samples.append(new_hyp_samples[idx])
               hyp_scores.append(new_hyp_scores[idx])

       hyp_scores = np.array(hyp_scores)
       live_hyp = new_live_k
       if new_live_k < 1:
           break
       if dead_hyp >= beam_size:
           break

       decoder_inputs = np.array([w[-1] for w in hyp_samples]+[0]*(len(decoder_inputs)-len(hyp_samples)))
        # dump every remaining one
   if dump_remaining:
       if live_hyp > 0:
           for idx in xrange(live_hyp):
               sample.append(hyp_samples[idx])
               sample_score.append(hyp_scores[idx])

   # normalize scores according to sequence lengths
   if normalize:
       lengths = np.array([len(s) for s in sample])
       sample_score = sample_score / lengths

        # sort the samples by score (it is in log-scale, therefore lower is better)
   sidx = np.argsort(sample_score)
   sample = np.array(sample)[sidx]
   sample_score = np.array(sample_score)[sidx]
   
   return sample.tolist()
def gen_sample(sess, gen_config, model, vocab, source_inputs, source_outputs, history_alpha, mc_search=True):
    #gen_config.batch_size=1
    sample_inputs = []
    sample_labels =[]
    rep = []
    rep_same = []
    index = 0
    print("get pre start")
    responses = get_predicted_sentence(sess, source_inputs, vocab,model, gen_config.beam_size, _buckets, mc_search,debug=True,history_alpha=history_alpha)
    print("get pre end")
    print("responses")
    print(responses)
    form_resp=[]
    for resp in responses:
        if gen_config.beam_size == 1 or (not mc_search):
            dec_inp = [dec for dec in resp['dec_inp']]
            dec_inp=dec_inp[:]
        else:
            dec_inp = [dec.tolist()[0] for dec in resp['dec_inp'][:]]
            dec_inp = dec_inp[1:]
        seg=[]
        for value in dec_inp:
            #print (value)
            seg.append(value)
        form_resp.append(seg)
    if mc_search==False:
        form_resp=(np.transpose(np.array(form_resp, dtype = int))).tolist()
    print('form_resp')
    print(form_resp)
    index=0        
    for source_query, source_answer in zip(source_inputs, source_outputs):
        sample_inputs.append(source_query+source_answer[:])
        len_dec_size=len(source_answer)
        sample_labels.append([0, 1])
        len_source_answer = len(source_answer[:-1])
        #responses = get_predicted_sentence(sess, source_query, vocab,
         #                                  model, gen_config.beam_size, _buckets, dec_size, mc_search,debug=True,history_alpha=history_alpha)
            ### the same length of input and output
            #dec_inp = dec_inp[:len_source_answer]
        dec_inp=[]
        for item in form_resp[index]:
            if mc_search==True:
                dec_inp.append(item)
            else:
                dec_inp.extend(item)
        print("dec_inp")
        print (dec_inp)


        rep.append(dec_inp[:len_dec_size])
        rep_same.append(dec_inp)
            #print ("  (%s) -> %s" % (resp['prob'], dec_inp))
        sample_neg = source_query + dec_inp
        sample_inputs.append(sample_neg)
        sample_labels.append([1, 0])
        index +=1

    return sample_inputs, sample_labels, rep, rep_same
    pass
