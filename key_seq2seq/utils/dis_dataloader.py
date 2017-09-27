import numpy as np
from re import compile as _Re
import cPickle
import utils.data_utils as data_utils

def split_unicode_chrs(text):
    _unicode_chr_splitter = _Re('(?s)((?:[\ud800-\udbff][\udc00-\udfff])|.)').split
    return [chr for chr in _unicode_chr_splitter(text) if chr]


class Dis_dataloader():
    def __init__(self, vocab_size, seq_length):
        self.vocab_size = vocab_size
        self.seq_length = seq_length

    def load_data_and_labels(self, positive_file, negative_file):
        """
        Loads MR polarity data from files, splits the data into words and generates labels.
        Returns split sentences and labels.
        """
        positive_examples = []
        negative_examples = []
        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split('\t')
                #source = line[0].strip().split(' ')
                #target = line[1].strip().split(' ')
                #source = [int(x) for x in source]
                #sources.append(source)
                #target = [int(x) for x in target]
                source_list = [each.strip().split(' ') for each in line]
                sources =[]
                for each in source_list:
                    #print " ".join(each)
                    for item in each:
                        sources.append(int(item))
                parse_line=sources+[data_utils.PAD_ID]*(self.seq_length-len(sources))
                positive_examples.append(parse_line)

        with open(negative_file)as fin:
            index = 0
            for line in fin:
                line = line.strip()
                line = line.split('\t')
                #parse_line = [int(x) for x in line]
                neg_list = [each.strip().split(' ') for each in line]
                neg = []
                for each in neg_list:
                    for item in each:
                        neg.append(int(item))


                parse_line_1=neg+[data_utils.PAD_ID]*(self.seq_length-len(neg))
                negative_examples.append(parse_line_1)
                index += 1
        # Split by words
        x_text = positive_examples + negative_examples
        print ("x_test:%d" %(len(x_text)))

        # Generate labels
        positive_labels = [[1,0] for _ in positive_examples]
        negative_labels = [[0,1] for _ in negative_examples]
        print ("negative_labels:%d" %(len(negative_labels)))
        #y = positive_labels + negative_labels
        y = np.concatenate([positive_labels, negative_labels], 0)

        x_text = np.array(x_text)
        y = np.array(y)
        return [x_text, y]

    def load_train_data(self, positive_file, negative_file):
        """
        Returns input vectors, labels, vocabulary, and inverse vocabulary.
        """
        # Load and preprocess data
        sentences, labels = self.load_data_and_labels(positive_file, negative_file)
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        x_shuffled = sentences[shuffle_indices]
        y_shuffled = labels[shuffle_indices]
        #self.sequence_length = 28
        return [x_shuffled, y_shuffled]

    def load_test_data(self, positive_file, test_file):
        test_examples = []
        test_labels = []
        with open(test_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                test_examples.append(parse_line)
                test_labels.append([1, 0])

        with open(positive_file)as fin:
            for line in fin:
                line = line.strip()
                line = line.split()
                parse_line = [int(x) for x in line]
                test_examples.append(parse_line)
                test_labels.append([0, 1])

        test_examples = np.array(test_examples)
        test_labels = np.array(test_labels)
        shuffle_indices = np.random.permutation(np.arange(len(test_labels)))
        x_dev = test_examples[shuffle_indices]
        y_dev = test_labels[shuffle_indices]

        return [x_dev, y_dev]
    def batch_iter_epoch(self, data, batch_size):
        data = np.array(data)
        data_size = len(data)
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        num_batches_per_epoch = int(data_size/batch_size)
        for batch_num in range(num_batches_per_epoch):
             start_index = batch_num * batch_size
             end_index = min((batch_num + 1) * batch_size, data_size)
             yield shuffled_data[start_index:end_index]  

    def batch_iter(self, data, batch_size, num_epochs):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        x, y = data
        print (len(x))
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data_x = x[shuffle_indices]
            shuffled_data_x = y[shuffle_indices]
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield (shuffled_data_x[start_index:end_index], shuffled_data_y[start_index:end_index])
