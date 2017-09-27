import os

import tensorflow as tf
import numpy as np
import time
import gen.generator as gens
import disc.discriminator as discs

import utils.conf as conf

gen_config = conf.gen_config
disc_config = conf.disc_config
evl_config = conf.disc_config
#sys.path.append("./disc")
#sys.path.appedn("./utils")
# pre train discriminator
def gen_poems(sess, gen_config, gen_model,vocab,source_data_path, mc_search=False, output_file= ""):
    start_time = int(time.time())
    _,  source_data = gens.read_data(source_data_path)
    #print source_data
    #source_inputs = source_data[:][0]
    #source_outputs = source_data[:][1]
    #source_inputs = []
    #source_outputs = []
    poem = []
    for i in xrange(len(source_data)):
        source_inputs = source_data[i][0].strip()
        source_outputs = source_data[i][1].strip()
        _, _, responses = gens.gen_sample(sess, gen_config, gen_model, vocab,source_inputs, source_outputs, mc_search=mc_search)
        for res in responses:
            source_inputs += " "+res
            _, _, responses = gens.gen_sample(sess, gen_config, gen_model, vocab,source_inputs, source_outputs, mc_search=mc_search)
        for res in responses:
            source_inputs += " "+res
            _, _, responses = gens.gen_sample(sess, gen_config, gen_model, vocab,source_inputs, source_outputs, mc_search=mc_search)
        for res in response:
            source_inputs += " "+res
            poem.append(source_inputs)


            
    end_time = time.time()
    print 'Sample generation time:', (end_time - start_time)
    
    with open(output_file, 'w') as fout:
        for poem in responses:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)
    fout.close()
def decode():
    gen_config.batch_size = 1 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)
    #sess.run(tf.global_variables_initializer())


    with tf.Session(config = config) as sess:

        sess.run(tf.global_variables_initializer())
        gen_model = gens.create_model(sess, gen_config, forward_only=True)
        vocab, rev_vocab, train_set, dev_set, train_path, eval_path = gens.prepare_data(gen_config)
        #gen_samples(sess, gen_config, gen_model, vocab, train_path, mc_search=False, output_file= gen_config.generate_train_file)
        gen_samples(sess, gen_config, gen_model, vocab, eval_path, mc_search=False, output_file= gen_config.generate_eval_file )

def main(_):
    #gen_pre_train()
    #disc_pre_train()
    al_train()

if __name__ == "__main__":
  tf.app.run()
