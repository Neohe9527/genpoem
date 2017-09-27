import os
import tensorflow as tf
import numpy as np
import time
import gen.generator as gens
import disc.discriminator as discs

import utils.conf as conf
from utils import data_utils
import sys
gen_config = conf.gen_config
disc_config = conf.disc_config
evl_config = conf.disc_config
reload(sys)
sys.setdefaultencoding('utf-8')
#sys.path.append('utils')
# pre train discriminator
# pre train generator
def get_vocab(vocab_path):
    rev_vocab=[]
    vocab={}
    fr = open(vocab_path, 'r')
    for line in fr:
        word, _id = line.strip().decode('utf-8').split('\t')
        word= word.strip()
        _id = _id.strip()
        vocab[word]= _id
        rev_vocab.append(word)
    return vocab,  rev_vocab

def prepare_data(gen_config):
    vocab_path = os.path.join(gen_config.data_dir, "vocab%d.all" % gen_config.vocab_size)
    vocab, rev_vocab = get_vocab(vocab_path)
    #for each in vocab:
     #   print each, vocab[each]
    train_ids_path, dev_ids_path = data_utils.prepare_chitchat_data(gen_config.data_dir, vocab, gen_config.vocab_size)
    train_set = gens.read_data(train_ids_path)
    dev_set = gens.read_data(dev_ids_path)
    return vocab, rev_vocab, train_set, dev_set
def gen_pre_train():
    gens.train(gen_config)


def main(_):
    gen_pre_train()

if __name__ == "__main__":
  tf.app.run()
