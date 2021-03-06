import data_utils
import tree_rnn
import graph_lstm 

import numpy as np
import theano
from theano import tensor as T
import random, time, os, sys, gc
import pickle
from train_util import read_matrix_from_file
#from memory_profiler import profile

#DIR = 'data' 
#DIR = '/home/t-napeng/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds'
DIR = '/home/t-napeng/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_triple/feature_folds'
GLOVE_DIR = '../treelstm/data'  # should include .npy files of glove vecs and words
EMB_DIR = '../spinn/glove/glove.6B.100d.txt'
FINE_GRAINED = False
DEPENDENCY = True #False
SEED = 88

NUM_EPOCHS = 30
LEARNING_RATE = 0.01

EMB_DIM = 100 #300
HIDDEN_DIM = 150


class SentimentModel(graph_lstm.ChildAvgGraphLSTM):
    #@profile
    def train_step_inner(self, x, y, child_exists, *entt_ids): #, y_exists):
        #print 'here!! input dep:', dep[:, :]
        loss, pred_y = self._train(x, child_exists, y, *entt_ids) #, y_exists)
        #self._update(LEARNING_RATE)
        return loss, pred_y

    #@profile
    def train_step(self, sentence, dependency, label, child_exists, *entt_ids):
        x = np.array(sentence, dtype='int32')
        #dep = np.array(dependency, dtype='int32')
        new_entt_idx = []
        for eid in entt_ids:
            new_entt_idx.append(np.array(eid, dtype='int32'))
        # Note: this was intended to check the wellformness of the dependency input, but I change it to transfer the input to be wellformness
        #print dep[:, -1]
        #self._check_input(child_exists)
        y = label
        loss, pred_y = self.train_step_inner(x, y, child_exists, *new_entt_idx) #, labels_exist)
        return loss, pred_y

    def loss_fn_multi(self, y, pred_y, y_exists):
        return T.sum(T.nnet.categorical_crossentropy(pred_y, y) * y_exists)

    #@profile
    def predict(self, sentence, dependency, child_exists, *entt_ids):
        x = np.array(sentence, dtype='int32')
        #dep = np.array(dependency, dtype='int32')
        new_entt_idx = []
        for eid in entt_ids:
            new_entt_idx.append(np.array(eid, dtype='int32'))
        # Note: this was intended to check the wellformness of the dependency input, but I change it to transfer the input to be wellformness
        #self._check_input(dep)
        '''print 'IN PREDICTION!!'
        print 'words:', [self.vocab[idx] for idx in x]
        print 'index:', indices
        print 'idx_dict:', idx_dict
        print 'original entity idx:', entt_ids
        print 'new entity idx:', new_entt_idx
        return [0,1]
        '''
        return self._predict(x, child_exists, *new_entt_idx)
    


def get_model(num_emb, output_dim, max_degree=2, num_entities=2, arc_type=3):
    return SentimentModel(
        num_emb, EMB_DIM, HIDDEN_DIM, output_dim,
        degree=max_degree, dep_types=arc_type, learning_rate=LEARNING_RATE,
        trainable_embeddings=True,
        labels_on_nonroot_nodes=False,
        num_entities=num_entities) #True)


def train():
    ''' load vocabulary, datasets(train, dev, test) and the maximum degree '''
    num_entities = int(sys.argv[2])
    train_set, dev_set, dicts, max_degree, arc_type_dict = data_utils.read_RE_dataset_graph(DIR, 5, int(sys.argv[1]), num_entities)

    #train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    print 'train', len(train_set)
    print 'dev', len(dev_set)
    #print 'test', len(test_set)

    num_emb = len(dicts['words2idx'])
    num_labels = len(dicts['labels2idx']) #5 if FINE_GRAINED else 3
    #for _, dataset in data.items():
    #    labels = [label for _, label in dataset]
    #    assert set(labels) <= set(xrange(num_labels)), set(labels)
    print 'num emb', num_emb
    print 'num labels', num_labels

    random.seed(SEED)
    np.random.seed(SEED)
    ''' Initialize the model '''
    model = get_model(num_emb, num_labels, max_degree, num_entities, len(arc_type_dict))
    print 'Finish initializing the model!!!'
    ''' initialize model embeddings to glove '''
    #embeddings = model.embeddings.get_value()
    #glove_vecs = np.load(os.path.join(GLOVE_DIR, 'glove.npy'))
    #glove_words = np.load(os.path.join(GLOVE_DIR, 'words.npy'))
    #glove_word2idx = dict((word, i) for i, word in enumerate(glove_words))
    #for i, word in enumerate(vocab.words):
    #for word, i in dicts['words2idx'].iteritems():
    #    if word in glove_word2idx:
    #        embeddings[i] = glove_vecs[glove_word2idx[word]]
    #glove_vecs, glove_words, glove_word2idx = [], [], []
    M_emb, _ = read_matrix_from_file(EMB_DIR, dicts['words2idx'])
    model.embeddings.set_value(M_emb)
    
    model.vocab = dict((v, k) for k,v in dicts['words2idx'].iteritems())
    print 'Finish loading embeddings!!!'
    for epoch in xrange(NUM_EPOCHS):
        print 'epoch', epoch
        tic = time.time() 
        avg_loss = train_dataset(model, train_set)
        print '\n>> Epoch completed in %.2f (sec) << avg loss: %.2f' % (time.time() - tic,  avg_loss)
        train_score = evaluate_dataset(model, train_set)
        print 'train score', train_score
        dev_score = evaluate_dataset(model, dev_set)
        print 'dev score', dev_score

    print 'finished training'
    #test_score = evaluate_dataset(model, test_set)
    #print 'test score', test_score

def train_dataset(model, data):
    losses = []
    avg_loss = 0.0
    total_data = len(data)
    tic = time.time() 
    print 'before shuffle the data, data[0]:', data[0][-2:]
    random.shuffle(data)
    print 'after shuffle the data, data[0]:', data[0][-2:]
    print type(data), type(data[0])
    correct = 0
    for i, (sentence, dependency, child_exists, label, idx) in enumerate(data):
        loss, pred_y = model.train_step(sentence, dependency, label, child_exists, *idx)  
        losses.append(loss)
        avg_loss = avg_loss * (len(losses) - 1) / len(losses) + loss / len(losses)
        correct += (label == np.argmax(pred_y))
        if i%10 == 0:

            #print 'avg loss %.2f at example %d of %d, label: %d' % (avg_loss, i, total_data, label),
            print 'avg loss %.2f at example %d of %d, label: %d, prediction: %d, accuracy: %.2f' % (avg_loss, i, total_data, label, np.argmax(pred_y), float(correct)/len(losses)),
            print '[learning] >> %2.2f%%' % ( (i + 1) * 100. / total_data),
            print 'completed in %.2f (sec) <<\r' % (time.time() - tic),
            sys.stdout.flush()
    return np.mean(losses)


def evaluate_dataset(model, data):
    num_correct = 0
    for sentence, dep, child_exists, label, idx in data:
        pred_y = model.predict(sentence, dep, child_exists, *idx)[-1]  # root pred is final row
        #print 'in evaluation, pred_y:', pred_y, 'label:', label
        #print 'in evaluation, label:', label, 'prediction:', np.argmax(pred_y)
        num_correct += (label == np.argmax(pred_y))
    return float(num_correct) / len(data)


if __name__ == '__main__':
    train()
