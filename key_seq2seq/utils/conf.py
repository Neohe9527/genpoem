import os
class disc_config(object):
   #batch_size = 1
   lr = 0.5
   lr_decay = 0.6
   batch_size = 128
   seq_length = 32
   max_len = 32
   vocab_size = 7938
   embedding_size = 512
   num_classes = 2
   #filter_sizes = [1,2, 3, 4,6,8,10, 12, 16, 18, 24, 32]
   #filter_sizes = [1,2, 3, 4,6,8,10, 12, 16, 18, 24, 32]
   #num_filters = [200, 200, 200, 100, 100, 100, 160, 160, 160, 160, 200, 200]
   filter_sizes = [1,2]
   num_filters = [200, 200]
   l2_reg_lambda = 0.2
   train_dir ="/home/work/chenyan/poem/key_seq2seq/0615/0615/disc_model"
   keep_prob = 0.75
   num_epoch = 30
   max_decay_epoch = 1
   max_grad_norm = 5
   out_dir = "/home/work/chenyan/poem/key_seq2seq/0615/0615/disc_model"
   checkpoint_every =1 
   init_scale = 0.1
   train_log = "/home/work/chenyan/poem/key_seq2seq/0615/0615/disc_model"
   eval_log = "/home/work/chenyan/poem/key_seq2seq/0615/0615/disc_model"

   train_pos_file = "/home/work/chenyan/poem/key_seq2seq/gan_data/train.true"
   train_neg_file = "/home/work/chenyan/poem/key_seq2seq/gan_data/train.gen"
   valid_pos_file = "/home/work/chenyan/poem/key_seq2seq/gan_data/valid.true"
   valid_neg_file = "/home/work/chenyan/poem/key_seq2seq/gan_data/valid.gen"

   train_pos_id = "/home/work/chenyan/poem/key_seq2seq/gan_data/true.ids7938.train"
   train_neg_id = "/home/work/chenyan/poem/key_seq2seq/gan_data/gen.ids7938.train"
   valid_pos_id = "/home/work/chenyan/poem/key_seq2seq/gan_data/true.ids7938.valid"
   valid_neg_id = "/home/work/chenyan/poem/key_seq2seq/gan_data/gen.ids7938.valid"

   
class gen_config(object):
    beam_size = 5
    learning_rate = 0.5
    learning_rate_decay_factor = 0.99
    max_gradient_norm = 5.0
    batch_size =128
    emb_dim = 512
    size = 512
    num_layers = 2
    vocab_size = 7938
    data_dir = "/home/work/chenyan/poem/key_seq2seq/gan_data"
    train_dir = "/home/work/chenyan/poem/key_seq2seq/0615/0615/gan_model_now/"
    train_file = "/home/work/chenyan/poem/key_seq2seq/gan_data/train.true"
    eval_file = "/home/work/chenyan/poem/key_seq2seq/gan_data/valid.true"
    
    train_log = "/home/work/chenyan/poem/key_seq2seq/0615/0615/gan_model_now/"
    eval_log = "/home/work/chenyan/poem/key_seq2seq/0615/0615/gan_model_now/"
    max_train_data_size = 0
    steps_per_checkpoint = 5000
    buckets = [(8,8),(12, 8), (20, 8), (28, 8)]
    writer_path = "/home/chenyan/poem/seq2seq/key_seq2seq/pre_train/0615/0813/log"
    generate_train_file = "/home/work/chenyan/poem/key_seq2seq/gan_data/data.train.big.gan"
    generate_eval_file = "/home/work/chenyan/poem/key_seq2seq/gan_data/data.valid.big.gan"
    word2vec_path = "/home/work/chenyan/poem/key_seq2seq/train_data/ancient/symbol_data/train_big/dict7938.emb"
    history_alpha= 0.005
    maxlen = 7
    minlen = 7

class gan_config(object):
    steps_per_checkpoint = 2000
    train_log = "/home/work/chenyan/poem/key_seq2seq/0615/0615/gan_model_now"
    eval_log = "/home/work/chenyan/poem/key_seq2seq/0615/0615/gan_model_now"
    batch_size = 128
