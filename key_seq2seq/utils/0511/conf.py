import os
class disc_config(object):
    batch_size = 1
    lr = 0.1
    lr_decay = 0.6
    vocabulary_size = 20000
    embed_dim = 128
    hidden_neural_size = 128
    hidden_layer_num = 1
    train_dir = 'data/subj0.pkl'
    max_len = 40
    valid_num = 100
    checkpoint_num = 1000
    init_scale = 0.1
    class_num = 2
    keep_prob = 0.5
    num_epoch = 60
    max_decay_epoch = 30
    max_grad_norm = 5
    out_dir = os.path.abspath(os.path.join(os.path.curdir,"runs"))
    checkpoint_every = 10

class gen_config(object):
    beam_size = 5
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size = 128
    #emb_dim = 512
    size = 512
    num_layers = 2
    vocab_size = 6004
    train_dir = "/home/chenyan/poem/seq2seq/gan_seq2seq/pre_train/0511/model/"
    data_dir = "/home/chenyan/poem/seq2seq/gan_seq2seq/train_data/ancient/symbol_data"
    max_train_data_size = 0
    steps_per_checkpoint = 5000
    buckets = [(8, 9), (16, 9), (24, 9)]
    train_log = "/home/chenyan/poem/seq2seq/gan_seq2seq/pre_train/0511/"
    eval_log = "/home/chenyan/poem/seq2seq/gan_seq2seq/pre_train/0511/"
    writer_file = "/home/chenyan/poem/seq2seq/gan_seq2seq/pre_train/0511/log"

    


