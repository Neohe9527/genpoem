import os
import sys
import tensorflow as tf
import numpy as np
import time
import gen.generator as gens
import disc.discriminator as discs
import math
import utils.conf as conf
import utils.data_utils as data_utils
from tensorflow.python.platform import gfile
gen_config = conf.gen_config
disc_config = conf.disc_config
evl_config = conf.disc_config
gan_config = conf.gan_config
#sys.path.append("./disc")
#sys.path.appedn("./utils")
# pre train discriminator
_PAD= b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK] 
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
def sentence_to_token_ids(sentence, vocab):
    pairs = sentence.strip().strip('\n').decode('utf-8').split('\t')
    res = []
    for each in pairs:
        each = each.strip().split(" ")
        res.append(" ".join([str(vocab.get(w, UNK_ID)) for w in each]))
    return res
def data_toids(data_path,target_path, vocab):
    if not gfile.Exists(target_path):
       print("Tokenizing data in %s" % data_path)
       with gfile.GFile(data_path, mode="rb") as data_file:
         with gfile.GFile(target_path, mode="w") as tokens_file:
            counter = 0
            for line in data_file:
                counter += 1
                if counter % 100000 == 0:
                   print("  tokenizing line %d" % counter)
                res = sentence_to_token_ids(line, vocab)
                tokens_file.write("\t".join(res)+'\n')
def disc_pre_train(vocab):
    #disc_config.train_neg_file = train_neg_file
    #disc_config.valid_neg_file = valid_neg_file
    data_toids(disc_config.train_pos_file, disc_config.train_pos_id,vocab)
    data_toids(disc_config.valid_pos_file, disc_config.valid_pos_id,vocab)
    data_toids(disc_config.train_neg_file, disc_config.train_neg_id,vocab)
    data_toids(disc_config.valid_neg_file, disc_config.valid_neg_id,vocab)
    discs.train_step(disc_config, disc_config)

# pre train generator
def gen_pre_train():
    gens.train(gen_config)

# prepare data for discriminator and generator
def disc_train_data(sess, gen_model, vocab, source_inputs, source_outputs, mc_search=False):
    decoder_size=[]
    for each in source_outputs:
        decoder_size.append(len(each))
    sample_inputs, sample_labels=[],[]
    #for (each_in, each_out) in zip(source_inputs, source_outputs):
     #   print(each_in)
      #  print (each_out)
       # each_sample_inputs, each_sample_labels, responses, rep_same = gens.gen_sample(sess, gen_config, gen_model, vocab,each_in, each_out, decoder_size, history_alpha= gen_config.history_alpha, mc_search=mc_search)
        #sample_inputs.append(each_sample_inputs)
        #sample_labels.append(each_sample_labels)
    print("disc_train_data, mc_search: ", mc_search)
    print(source_inputs)
    print(source_outputs)
    sample_inputs, sample_labels, responses, rep_same = gens.gen_sample(sess, gen_config, gen_model, vocab, source_inputs, source_outputs, history_alpha= gen_config.history_alpha, mc_search=mc_search)
    re_sample_inputs=[]
    print("sample_inputs")
    print(sample_inputs)
    print("responses")
    print (responses)
    for _sample_input in sample_inputs:
        split_idx = _sample_input.index(4)
        content = _sample_input[split_idx+1:]
        num=0
        for each in content:
            if each==7 or each==6:
                num+=1
        if num%2==0:
            content.append(6)
        else:
            content.append(7)
        re_sample_inputs.append(content)
    #print(re_sample_inputs[0])
    for input, label in zip(re_sample_inputs, sample_labels):
        print(str(label) + "\t" + str(input))

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))
    sorted_index = len_argsort(re_sample_inputs)
    train_set_x = [re_sample_inputs[i] for i in sorted_index]
    train_set_y = [sample_labels[i] for i in sorted_index]
    train_set=(train_set_x,train_set_y)
    new_train_set_x=np.zeros([len(train_set[0]),disc_config.max_len])
    #print("new_train_set: ", np.shape(new_train_set_x))
    new_train_set_y=np.zeros([len(train_set[0]), disc_config.num_classes])
    #print("new_train_set_y: ", np.shape(new_train_set_y))
    mask_train_x=np.zeros([disc_config.max_len,len(train_set[0])])

    def padding_and_generate_mask(x,y,new_x,new_y,new_mask_x):
        for i,(x_i,y_i) in enumerate(zip(x,y)):
            #whether to remove sentences with length larger than maxlen
            if len(x_i)<=disc_config.max_len:
                new_x[i,0:len(x_i)]=x_i
                new_mask_x[0:len(x_i),i]=1
                new_y[i]=y_i
            else:
                new_x[i]=(x_i[0:disc_config.max_len])
                new_mask_x[:,i]=1
                new_y[i]=y_i
        new_set =(new_x,new_y,new_mask_x)
        del new_x,new_y
        return new_set

    train_inputs, train_labels, train_masks =padding_and_generate_mask(train_set[0],train_set[1],
                                                                     new_train_set_x,new_train_set_y,mask_train_x) 
    return train_inputs, train_labels, train_masks, responses
def gen_samples(sess, gen_config, gen_model,vocab,source_data_path, mc_search=False, output_file= ""):
    start_time = int(time.time())
    _,  source_data = gens.read_data(source_data_path)
    #print source_data
    #source_inputs = source_data[:][0]
    #source_outputs = source_data[:][1]
    source_inputs = []
    source_outputs = []
    #for i in xrange(len(source_data)):
    for i in xrange(len(source_data)):
        source_inputs.append(source_data[i][0])
        source_outputs.append(source_data[i][1])
    print(len(source_inputs))
    ####same length of output and input
    _, _, responses, rep_same = gens.gen_sample(sess, gen_config, gen_model, vocab,source_inputs, source_outputs, mc_search=mc_search)
    end_time = time.time()
    print 'Sample generation time:', (end_time - start_time)
    with open(output_file, 'w') as fout:
        for poem in rep_same:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)
    fout.close()
# discriminator api
def disc_step(sess, disc_model, train_inputs, train_labels):
    #### one example
    feed_dict={}
    feed_dict[disc_model.input_x]=train_inputs
    feed_dict[disc_model.input_y]=train_labels
    #feed_dict[disc_model.mask_x]=train_masks
    disc_model.assign_new_batch_size(sess,len(train_inputs))
    fetches = [disc_model.loss,disc_model.accuracy,disc_model.train_op]
    #state = sess.run(disc_model._initial_state)

    cost,accuracy,_, = sess.run(fetches,feed_dict)
    print("the train cost is: %f and the train accuracy is %f ."%(cost, accuracy))
    return accuracy

# Adversarial Learning for Neural Dialogue Generation
def al_train():
    #gen_config.batch_size = 1 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    ##pre_train for D
    vocab, rev_vocab, train_set, dev_set, train_path, eval_path = gens.prepare_data(gen_config)
    #print (train_set)

    #sess = tf.Session(config=config)
    #sess.run(tf.global_variables_initializer())
    gen_batch_size = gen_config.batch_size


    with tf.Session(config = config) as sess:

        #sess.run(tf.global_variables_initializer())
        #disc_model = discs.create_model(sess, disc_config, is_training=True)
        #disc_model = discs.create_model(sess, disc_config)
        gen_model = gens.create_model(sess, gen_config, forward_only=True)
        #gen_samples(sess, gen_config, gen_model,vocab,source_data_path, mc_search=1, output_file= gen.config.outputfile)
        #gen_samples(sess, gen_config, gen_model, vocab, train_path, mc_search=False, output_file= gen_config.generate_train_file)
        #gen_samples(sess, gen_config, gen_model, vocab, eval_path, mc_search=False, output_file= gen_config.generate_eval_file )
        print ("D pre_train")
        #discs.train_neg_file = gen_config.generate_train_file
        #discs.valid_neg_file = gen_config.generate_eval_file
        #disc_pre_train(vocab)

        ### disc model
        disc_model = discs.create_model(sess, disc_config, is_training=True)
        print("load d model ok!!!")
        _buckets = gen_config.buckets
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(gen_config.buckets))]
        train_total_size = int(sum(train_bucket_sizes))
        print ("total train sets: %d\n" %(train_total_size))
        dev_bucket_sizes = [len(dev_set[b]) for b in xrange(len(gen_config.buckets))]
        dev_total_size = int(sum(dev_bucket_sizes))
        print ("total dev sets: %d\n" %(dev_total_size))
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / float(train_total_size)
                               for i in xrange(len(train_bucket_sizes))]
        current_step = 0
        step_time, loss = 0.0, 0.0
        previous_losses = []
        dev_sum_epoch =  int(dev_total_size/gan_config.batch_size/len(_buckets))
        ## add loss
        train_file_path = os.path.join(gan_config.train_log, "train.loss")
        f_train = gfile.GFile(train_file_path, mode="wb")
        eval_file_path = os.path.join(gan_config.eval_log, "eval.loss")
        f_eval = gfile.GFile(eval_file_path, mode="wb")
        for iter in xrange(100*gan_config.steps_per_checkpoint):
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in xrange(len(train_buckets_scale))
                         if train_buckets_scale[i] > random_number_01])
            start_time = time.time()

            print("===========================Update Discriminator================================")
            # 1.Sample (X,Y) from real data
            _, _, _, source_inputs, source_outputs,decoder_size  = gen_model.get_batch(train_set, bucket_id, 4, 0)
            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X)
            train_inputs, train_labels,  _, _ = disc_train_data(sess,gen_model,vocab,
                                                        source_inputs,source_outputs, mc_search=False)
            # 3.Update D using (X, Y ) as positive examples and(X, ^Y) as negative examples
            disc_step(sess, disc_model, train_inputs, train_labels)

            print("===============================Update Generator================================")
            # 1.Sample (X,Y) from real data
            #gen_config.batch_size= gen_batch_size
            update_gen_data = gen_model.get_batch(train_set, bucket_id, 4, 0)
            encoder, decoder, weights, source_inputs, source_outputs, decoder_size = update_gen_data
            print("encoder")
            print(encoder)

            # 2.Sample (X,Y) and (X, ^Y) through ^Y ~ G(*|X) with Monte Carlo search
            train_inputs, train_labels, _, responses = disc_train_data(sess,gen_model,vocab,
                                                        source_inputs,source_outputs, mc_search=True)
            # 3.Compute Reward r for (X, ^Y ) using D.---based on Monte Carlo search
            reward = disc_step(sess, disc_model, train_inputs, train_labels)

            # 4.Update G on (X, ^Y ) using reward r
            dec_gen =[responses[i][:gen_config.buckets[bucket_id][1]] for i in xrange(len(responses)) if i%gen_config.beam_size==0]
            print("dec_gen")
            print(dec_gen)
            re_dec_gen=[]
            for each in dec_gen:
                if len(each)< gen_config.buckets[bucket_id][1]:
                    each = each + [0]*(gen_config.buckets[bucket_id][1] - len(each))
                re_dec_gen.append(each)
            #re_dec_gen = np.reshape(re_dec_gen, (-1,1))
            re_dec_gen=np.transpose(re_dec_gen)
            print(re_dec_gen)
            #gen_config.batch_size= gen_batch_size
            gen_model.step(sess, encoder, re_dec_gen, weights, bucket_id, forward_only=False,
                   up_reward=True, reward=reward, debug=True)

            # 5.Teacher-Forcing: Update G on (X, Y )
            _, step_loss, _ = gen_model.step(sess, encoder, decoder, weights, bucket_id, forward_only=False, up_reward=False)
            print("loss: ", step_loss)
            step_time = (time.time()-start_time)/gan_config.steps_per_checkpoint
            loss += step_loss/gan_config.steps_per_checkpoint
            current_step += 1
            if current_step % gan_config.steps_per_checkpoint == 0:
                ppl = math.exp(loss) if loss<300 else float('inf')
                print ("global step %d learning rate %.4f step-time %.2f perplexity " "%.2f" % (gen_model.global_step.eval(), gen_model.learning_rate.eval(), step_time, ppl))
                if len(previous_losses)>2 and loss >max(previous_losses[-3:]):
                    sess.run(gen_model.learning_rate_decay_op)
                previous_losses.append(loss)
                f_train.write(str(float(loss))+'\t'+str(current_step)+'\n')
                step_time, loss = 0.0, 0.0
                ### dev set to test G
                sum_eval_ppx, sum_eval_loss = 0.0, 0.0
                for eval_epoch in xrange(dev_sum_epoch):
                    for bucket_id in xrange(len(_buckets)):
                        encoder_inputs, decoder_inputs, target_weights, _ , _, _ = gen_model.get_batch(dev_set, bucket_id, 4, 0)
                        _, eval_loss, _ = gen_model.step(sess, encoder_inputs, decoder_inputs,target_weights, bucket_id, True)
                        eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                        sum_eval_ppx += eval_ppx
                        sum_eval_loss += eval_loss
                f_eval.write(str(float(sum_eval_loss))+'\t'+str(current_step)+'\n')
                print ("eval_loss \t %f current_step %d\n" %(float(sum_eval_loss), current_step))
                checkpoint_path = os.path.join(gan_config.train_log, "poem_gan.model")
                gen_model.saver.save(sess, checkpoint_path, global_step=gen_model.global_step)
                sys.stdout.flush()
        f_eval.close()
        f_train.close()




        #add checkpoint
        #checkpoint_dir = os.path.abspath(os.path.join(disc_config.out_dir, "checkpoints"))
        #checkpoint_prefix = os.path.join(checkpoint_dir, "disc.model")
        #if not os.path.exists(checkpoint_dir):
         #   os.makedirs(checkpoint_dir)
        #pass

def main(_):
    #gen_pre_train()
    #disc_pre_train()
    al_train()

if __name__ == "__main__":
  tf.app.run()
