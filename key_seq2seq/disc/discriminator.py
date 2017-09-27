import tensorflow as tf                                                                                                                              
import numpy as np
import os
import time
import datetime
from text_classifier import TextCNN 
import utils.data_helper as data_helper
import utils.conf as conf
import sys
from utils.dis_dataloader import Dis_dataloader
sys.path.append('../utils')
def create_model(session, disc_config, is_training = True, name = "d_model"):
     with tf.variable_scope(name):
         model = TextCNN(
                 sequence_length = disc_config.seq_length,
                 num_classes = disc_config.num_classes,
                 vocab_size = disc_config.vocab_size,
                 embedding_size = disc_config.embedding_size,
                 filter_sizes = disc_config.filter_sizes,
                 num_filters = disc_config.num_filters,
                 l2_reg_lambda = disc_config.l2_reg_lambda,
                 dropout_keep_prob = disc_config.keep_prob,
                 is_training = is_training
               )
         ckpt = tf.train.get_checkpoint_state(disc_config.train_dir)
         if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
             print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
             model.saver.restore(session, ckpt.model_checkpoint_path)
         else:
             print("Created disc_CNN model with fresh parameters.")
             session.run(tf.global_variables_initializer())
     return model 
def evaluate(model, session, data, batch_size, summary_writer=None):
    correct_num = 0
    total_num = len(data)
    print ("total_num: %d" %total_num)
    for step, batch in enumerate(data_helper.batch_iter(data,batch_size=batch_size)):
        x,y = zip(*batch)
        feed = {model.input_x: x,
                model.input_y: y
                }
        model.assign_new_batch_size(session,len(x))
        fetches = model.correct_num
        count = session.run(fetches, feed)
        correct_num += count
    acc = float(correct_num)/total_num
    dev_summary = tf.summary.scalar('dev_accuracy',acc)
    dev_summary = session.run(dev_summary)
    if summary_writer:
        summary_writer.add_summary(dev_summary, model.global_step.eval())
        summary_writer.flush()

    return acc

def run_epoch(model, session, data,batch_size, train_summary_writer=None):
    sum_acc = 0.0
    sum_cost = 0.0
    dis_batches = data_helper.batch_iter(data,batch_size=batch_size)
    for step, batch in enumerate(dis_batches):
        x, y = zip(*batch)
        feed ={
                model.input_x: x,
                model.input_y: y
                }
        model.assign_new_batch_size(session,len(x))
        fetches= [model.train_op, model.loss, model.accuracy,model.scores,model.predictions,model.summary, model.losses, model.ypred_for_auc]
        #fetches = [model.loss,model.accuracy,model.train_op, model.summary]
        _, cost, accuracy, scores, predictions, summary, losses, ypred_for_auc= session.run(fetches,feed)
        sum_cost += cost
        sum_acc += accuracy
        #print scores
        #print predictions
        print("acc:%f, loss: %f" %(accuracy, cost))
        #print (correct_num)
        train_summary_writer.add_summary(summary,model.global_step.eval())
        train_summary_writer.flush()

    step_size = len(data)/batch_size
    avg_acc = sum_acc/step_size
    avg_cost = sum_cost/step_size

    return avg_acc, avg_cost
def train_step(config_disc, config_eval):
    print("loading the disc train set")
    config = config_disc 
    eval_config=config_eval
    eval_config.keep_prob=1.0
    dis_data_loader = Dis_dataloader(config_disc.vocab_size, config_disc.seq_length)

    train_data_x, train_data_y = dis_data_loader.load_data_and_labels(config_disc.train_pos_id, config_disc.train_neg_id)
    valid_data_x, valid_data_y = dis_data_loader.load_data_and_labels(config_disc.valid_pos_id, config_disc.valid_neg_id)
    train_data = zip(train_data_x, train_data_y)
    valid_data = zip(valid_data_x, valid_data_y)
    #train_data = dis_data_loader.load_train_data(config_disc.train_pos_file, config_disc.train_neg_file)
    #valid_data = dis_data_loader.load_train_data(config_disc.train_pos_file, config_disc.train_neg_file)
    print("begin training")
    #gpu_config=tf.ConfigProto()  
    #gpu_config.gpu_options.allow_growth=True
    #sess.run(tf.global_variables_initializer())

    with tf.Graph().as_default(), tf.Session() as session:        
        print("model training")
####load model can not initial here
        #initializer = tf.random_uniform_initializer(-1*config.init_scale,1*config.init_scale)
        #session.run(tf.global_variables_initializer())
        model = create_model(session, config, is_training = True)
        #with tf.variable_scope("d_model",reuse=True,initializer=initializer):
         #   valid_model = create_model(session, eval_config, is_training = False)
            #test_model = create_model(session, eval_config, is_training=False)
         
        #checkpoint_dir = os.path.abspath(os.path.join(config.out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(config.train_dir, "disc.model")
        if not os.path.exists(config.train_dir):            
            os.makedirs(config.train_dir)

        #tf.global_variables_initializer().run()       
        global_steps=1        
        begin_time=int(time.time())
        train_file_path = os.path.join(config_disc.train_log, "train.loss")
        f_train = file(train_file_path, 'wb')
        eval_file_path = os.path.join(config_disc.train_log, "eval.loss")
        f_eval = file(eval_file_path, 'wb')
        ## add summary
        train_summary_dir = os.path.join(config.out_dir,"summaries","train")
        train_summary_writer =  tf.summary.FileWriter(train_summary_dir,session.graph)

        dev_summary_dir = os.path.join(eval_config.out_dir,"summaries","dev")
        dev_summary_writer =  tf.summary.FileWriter(dev_summary_dir,session.graph)
        for i in range(config.num_epoch):
            print("the %d epoch training..."%(i+1))            
            lr_decay = config.lr_decay ** max(i-config.max_decay_epoch,0.0)           
            model.assign_new_lr(session,config.lr*lr_decay)
            start_time = int(time.time())
            avg_acc, avg_loss =run_epoch(model,session,train_data,config_disc.batch_size, train_summary_writer)
            step_time = (time.time() - start_time) / config_disc.batch_size
            print ("global step %d learning rate %.4f step-time %.2f acc %.2f and loss %.2f" % (model.global_step.eval(), model.lr.eval(), step_time, avg_acc, avg_loss))
            f_train.write(str(avg_loss)+'\t'+str(avg_acc)+'\n')
            if i% config.checkpoint_every==0: 
                ##dev set
                avg_acc =evaluate(model,session,valid_data, config_disc.batch_size, dev_summary_writer)
                f_eval.write(str(avg_acc)+'\n')
                print("avg_acc:%f\n" %avg_acc)
                path = model.saver.save(session,checkpoint_prefix,model.global_step)                
                print("Saved model chechpoint to{}\n".format(path))
        print("the train is finished")       
        end_time=int(time.time())       
        print("training takes %d seconds already\n"%(end_time-begin_time))        
            #test_accuracy=evaluate(test_model,session,test_data, config_disc.batch_size)        
            #print("the test data accuracy is %f"%test_accuracy)        
        print("program end!")
        f_train.close()
        f_eval.close()
def main():
    train_step()
if __name__ == "main":
    tf.app.run()

