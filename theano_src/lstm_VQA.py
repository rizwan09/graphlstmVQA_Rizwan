#!/usr/bin/python

import argparse, os, random, subprocess, sys, time
import theano, numpy
import theano.tensor as T
from copy import copy
from neural_lib import StackConfig, ArrayInit
from vqa_architectures import VQA, CNNRelation, LSTMRelation, LSTMRelation_multitask, GraphLSTMRelation, WeightedGraphLSTMRelation, WeightedAddGraphLSTMRelation, WeightedGraphLSTMRelation_multitask, WeightedAddGraphLSTMRelation_multitask, ArcPredAddGraphLSTMRelation
from train_util import dict_from_argparse, shuffle, create_relation_circuit, convert_id_to_word, add_arg, add_arg_to_L, conv_data
from vqa_architectures import *
from train_util import sgd, adadelta, rmsprop, read_matrix_from_gzip, read_matrix_from_file, read_matrix_and_idmap_from_file, batch_run_func, get_minibatches_idx, save_parameters, load_params
from vqa.data_process import load_data, load_data_cv, load_data_withAnnos, prepare_data
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh
import json
import numpy as np

''' Convert the entity index: from indexing to dense-vector(but many zero entries) multiplication.'''
def conv_idxs(idxs, length):
    new_idxs = [numpy.zeros(length).astype(theano.config.floatX) for i in range(len(idxs))]
    for i, idx in enumerate(idxs):
        new_idxs[i][idx] = 1.0
    return new_idxs

''' For prediction, both batch version and non-batch version'''
def predict(_args, f_classify, *data, **kwargs): #batchsize=1, graph=False, dep=None, weighted=False, print_prediction=False, prediction_file=None):
    ''' On the test set predict the labels using f_classify.
    Compare those labels against groundtruth.

    It returns a dictionary 'results' that contains
    f1 : F1 or Accuracy
    p : Precision
    r : Recall
    '''
    batchsize = kwargs.pop('batchsize', 1)
    dep = kwargs.pop('dep', None)
    weighted = kwargs.pop('weighted', False)
    print_prediction = kwargs.pop('print_prediction', False)
    prediction_file = kwargs.pop('prediction_file', None)
    groundtruth_test = data[-1]


    predictions_test = []
    if print_prediction:
        assert prediction_file is not None
        pred_file = open(prediction_file, 'w')
    if batchsize > 1:
        nb_idxs = get_minibatches_idx(len(data[0]), batchsize, shuffle=False)
        for i, tr_idxs in enumerate(nb_idxs):
            #words = [lex_test[ii] for ii in tr_idxs]
            #eidxs = [idxs_test[ii] for ii in tr_idxs]
            #labels = [groundtruth_test[ii] for ii in tr_idxs]
            #orig_eidxs = eidxs
            batch_data = [[elem[ii] for ii in tr_idxs] for elem in data]
            if _args.graph:
                assert dep is not None
                masks = [dep[ii] for ii in tr_idxs]
            else:
                masks = None
            x, x_masks, obj, obj_masks = prepare_data(batch_data[0], batch_data[1], masks, None, maxlen=200)
            if weighted or not _args.graph:
                pred_all = f_classify( x, obj, x_masks, obj_masks)
                #print len(pred_all)
                predictions_test.extend(list(numpy.argmax(pred_all, axis=1))) #[0]))
                '''print pred_all[1].shape, len(words), len(orig_eidxs)
                for iii, (line, att, idx, pred) in enumerate(zip(words, pred_all[1].T, orig_eidxs, pred_all[0])):
                    wds = [_args.idx2word[wd[0]] for wd in line]
                    try:
                        assert len(wds) <= len(att)
                    except:
                        print len(wds), len(att)
                    print wds, pred, groundtruth_test[i*batchsize + iii]
                    print att
                '''
            else:
                #print f_classify(x, masks.sum(axis=-1), *eidxs)
                pred_all = f_classify(x,  obj,  masks.sum(axis=-1) )
                predictions_test.extend(list(numpy.argmax(pred_all, axis=1)))
            if print_prediction:
                for idx, p in zip(tr_idxs, pred_all):
                    pred_file.write(str(idx) + '\t' + str(p[1]) + '\n')
    else:
        print "ERRRRRRRRRRRO"
        pass
    print 'in predict,', len(predictions_test), len(groundtruth_test)
    if print_prediction:
        pred_file.close()
    predictions_test = map(lambda k: _args.idx2label[k].split('(')[0], predictions_test)

    if groundtruth_test[0] == 0 or groundtruth_test[0] == 1:
        groundtruth_test = map(lambda k: _args.idx2label[k].split('(')[0], groundtruth_test)

    #eval_logitReg_F1(predictions_test, groundtruth_test)
    results = eval_logitReg_accuracy(predictions_test, groundtruth_test)
    print "Results:", results
    return results, predictions_test

def eval_logitReg_accuracy(predictions, goldens):
    assert len(predictions) == len(goldens)
    correct = 0.0
    for p, g in zip(predictions, goldens):
        # print 'in eval_logitReg_accuracy,', p, g
        if p == g:
            correct += 1.0
    return correct/len(predictions)

def eval_logitReg_F1(predictions, goldens):
    # the dictionary that stores scores for each type.
    # key=label, value=(tp, fp, fn)
    F1_dict = dict()
    assert len(predictions) == len(goldens)
    for p, g in zip(predictions, goldens):
        #print 'in eval_logitReg_accuracy,', p, g
        if g not in F1_dict:
            F1_dict[g] = [0.0, 0.0, 0.0]
        if p not in F1_dict:
            F1_dict[p] = [0.0, 0.0, 0.0]
        if p == g:
            F1_dict[g][0] += 1.0
        else:
            F1_dict[p][1] += 1.0
            F1_dict[g][2] += 1.0
    avg_prec, avg_recall, avg_F1 = 0.0, 0.0, 0.0
    all_tp, all_fp, all_fn = 0.0, 0.0, 0.0
    for k, v in F1_dict.items():
        if k == 'Other' or k == 'None':
            del F1_dict[k]
            continue
        tp, fp, fn = v
        all_tp += tp
        all_fp += fp
        all_fn += fn
        prec = tp / (tp + fp) if (tp != 0 or fp != 0) else 0
        recall = tp / (tp + fn) if (tp != 0 or fn != 0) else 0
        F1 = 2*prec*recall / (prec + recall) if (prec != 0 or recall != 0) else 0
        print 'label', k, 'precision:', prec, 'recall:', recall, 'F1:', F1
        avg_prec += prec
        avg_recall += recall
        avg_F1 += F1
    print 'macro average precision:', avg_prec/len(F1_dict), 'recall:', avg_recall/len(F1_dict), 'F1:', avg_F1/len(F1_dict)
    prec = all_tp / (all_tp + all_fp) if (all_tp != 0 or all_fp != 0) else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp != 0 or all_fn != 0) else 0
    F1 = 2*prec*recall / (prec + recall) if (prec != 0 or recall != 0) else 0
    print 'micro average precision:', prec, 'recall:', recall, 'F1:', F1
    #return prec, recall, F1
    return avg_prec/len(F1_dict), avg_recall/len(F1_dict), avg_F1/len(F1_dict)

''' For training on single-task setting, both batch and non-batch version'''
def train_single(_args, f_cost, f_update, epoch_id, learning_rate, nsentences, *data, **kwargs): #train_lex, train_idxs, train_y, batchsize=1, dep=None, weighted=False):
    ''' This function is called from the main method. and it is primarily responsible for updating the
    parameters. Because of the way that create_relation_circuit works that creates f_cost, f_update etc. this function
    needs to be flexible and can't be put in a lib.
    Look at lstm_dependency_parsing_simplification.py for more pointers.
    '''
    batchsize = kwargs.pop('batchsize', 1)
    dep = kwargs.pop('dep', None)
    weighted = kwargs.pop('weighted', False)
    # None-batched version
    def train_instance(learning_rate, f_cost, f_update, *inputs):
        ' Since function is called only for side effects, it is likely useless anywhere else'
        if inputs[0].shape[0] < 2:
            return 0.0
        #inputs = idxs + [words, label]
        iter_cost = f_cost(*inputs) #words, id1, id2, labels)
        f_update(learning_rate)
        return iter_cost

    # Mini-batch version
    '''def train_batch(words, masks, idxs, label, learning_rate, f_cost, f_update):
        if words.shape[0] < 2:
            return 0.0
        inputs = idxs + [words, masks, label]
        iter_cost = f_cost(*inputs) #words, id1, id2, labels)
        f_update(learning_rate)
        return iter_cost
    '''
    ## main body of train
    #print type(data)
    data = list(data)
    if dep:
        #shuffle([train_lex, train_idxs, train_y, dep], _args.seed)
        shuffle(data + [dep], _args.seed)
    else:
        shuffle(data, _args.seed)
    if nsentences < len(data[0]):
        data = [elem[:nsentences] for elem in data]
    tic = time.time()
    aggregate_cost = 0.0
    temp_cost_arr = [0.0] * 2

    # make the judge on whether use mini-batch or not.
    # No mini-batch
    if batchsize == 1:
        print "Error: batch size cannot be 1"
        pass
    # Mini-batch
    else:
        nb_idxs = get_minibatches_idx(len(data[0]), batchsize, shuffle=False)
        nbatches = len(nb_idxs)
        for i, tr_idxs in enumerate(nb_idxs):
            #words = [train_lex[ii] for ii in tr_idxs]
            #eidxs = [train_idxs[ii] for ii in tr_idxs]
            #labels = [train_y[ii] for ii in tr_idxs]
            #print [len(elem) for elem in data]
            batch_data = [[elem[ii] for ii in tr_idxs] for elem in data]
            #orig_eidxs = eidxs
            if _args.graph:
                assert dep is not None
                masks = [dep[ii] for ii in tr_idxs]
            else:
                masks = None
            x, x_masks, obj, obj_masks = prepare_data(batch_data[0], batch_data[1], masks, None, maxlen=200)
            '''print x.shape, len(words)
            for elem, wd in zip(numpy.transpose(x, (1,0,2)), words):
                print 'words:', wd
                print 'converted words:', elem
            '''
            if weighted or dep is None:
                iter_cost = train_instance(learning_rate, f_cost, f_update, x, obj, batch_data[-1], x_masks, obj_masks )
                print len(x), len(x_masks), len(obj), len(batch_data[-1]), len(obj_masks)
                print x
                print obj
                print x_masks
                print obj_masks
                print batch_data[-1]
                print iter_cost
                #for ii, c in enumerate(iter_cost):
                #    temp_cost_arr[ii] += c
                aggregate_cost += iter_cost#[0]

            else:
                aggregate_cost += train_instance(learning_rate, f_cost, f_update, x, obj, batch_[-1], masks.sum(axis=-1) )
            if _args.verbose == 2 :
                print '[learning] epoch %i >> %2.2f%%' % (epoch_id, (i + 1) * 100. / nbatches),
                print 'completed in %.2f (sec). << avg loss: %.2f <<\r' % (time.time() - tic, aggregate_cost/(i+1)),
                #print 'completed in %.2f (sec). << avg loss: %.2f <<%%' % (time.time() - tic, aggregate_cost/(i+1)),
                #print 'average cost for each part: (%.2f, %.2f) <<\r' %(temp_cost_arr[0]/(i+1), temp_cost_arr[1]/(i+1)),
                sys.stdout.flush()
    if _args.verbose == 2:
        print '\n>> Epoch completed in %.2f (sec) <<' % (time.time() - tic), 'training cost: %.2f' % (aggregate_cost)

''' Initialize some parameters for training and prediction'''
def prepare_corpus(_args):
    numpy.random.seed(_args.seed)
    random.seed(_args.seed)
    if _args.win_r or _args.win_l:
        _args.emb_win = _args.win_r - _args.win_l + 1
    word2idx = _args.dicts['words2idx']
    obj2idx = _args.dicts['objs2idx']
    _args.label2idx = _args.dicts['labels2idx']
    _args.idx2label = dict((k, v) for v, k in _args.label2idx.iteritems())
    #_args.idx2word = dict((k, v) for v, k in word2idx.iteritems())
    _args.nsentences = len(_args.train_set[1])
    #!!!!!!!!!!!!!!!
    _args.o_voc_size = len(obj2idx)
    _args.q_voc_size = len(word2idx)
    if 'arcs2idx' in _args.dicts:
        _args.lstm_arc_types = len(_args.dicts['arcs2idx'])
        print 'lstm arc types =', _args.lstm_arc_types
    #del _args.dicts
    #_args.groundtruth_valid = convert_id_to_word(_args.valid_set[-1], _args.idx2label)
    #_args.groundtruth_test = convert_id_to_word(_args.test_set[-1], _args.idx2label)

    _args.logistic_regression_out_dim = len(set(_args.label2idx.values()))
    eval_args(_args)
    _args.lstm_go_backwards = True #False
    try:
        print 'Circuit:', _args.circuit.__name__
        #print 'Chkpt1', len(_args.label2idx), _args.nsentences, _args.train_set[1][0], _args.voc_size, _args.train_set[2][0], _args.valid_set[1][0], _args.valid_set[2][0]
        #print 'Chkpt2', _args.emb_T_initializer.matrix.shape, _args.emb_T_initializer.matrix[0]
    except AttributeError:
        pass

''' Compile the architecture.'''
def compile_circuit(_args):
    ### build circuits. ###
    (_args.f_cost, _args.f_update, _args.f_classify, cargs) = create_relation_circuit(_args, StackConfig)
    _args.train_func = train_single
    print "Finished Compiling"
    return cargs

def convert_args(_args, prefix):
    from types import StringType
    for a in _args.__dict__:  #TOPO_PARAM + TRAIN_PARAM:
        try:
            if type(a) is StringType and a.startswith(prefix):
                _args.__dict__[a[len(prefix)+1:]] = _args.__dict__[a]
                del _args.__dict__[a]
        except:
            pass

def eval_args(_args):
    for a in _args.__dict__:  #TOPO_PARAM + TRAIN_PARAM:
        try:
            _args.__dict__[a] = eval(_args.__dict__[a])
        except:
            pass

def run_wild_prediction(_args):
    best_f1 = -numpy.inf
    param = dict(clr = _args.lr, ce = 0, be = 0, epoch_id = -1)
    cargs = compile_circuit(_args)
    while param['epoch_id']+1 < _args.nepochs:
        param['epoch_id'] += 1
        run_training(_args, param)
    train_lex, train_y, train_idxs, train_dep = _args.train_set
    valid_lex, valid_y, valid_idxs, valid_dep = _args.valid_set
    res_train, _ = predict(_args, _args.f_classify, train_lex, train_idxs, train_y, batchsize=_args.batch_size, dep=train_dep, weighted=_args.weighted)
    res_valid, _ = predict(_args, _args.f_classify, valid_lex, valid_idxs, valid_y, batchsize=_args.batch_size, dep=valid_dep, weighted=_args.weighted, print_prediction=_args.print_prediction, prediction_file=_args.prediction_file)
    if _args.verbose:
        print('TEST: epoch', param['epoch_id'],
                'train performances'   , res_train,
                'valid performances'   , res_valid)
    print('Training accuracy', res_train,
          )


def run_training(_args, param, multi_input=False):
    train_lex, train_y, train_obj, train_dep = _args.train_set
    if multi_input:
        _args.train_func(_args, _args.f_cost, _args.f_update, param['epoch_id'], param['clr'], _args.nsentences, *(train_lex + [train_idxs, train_y]), batchsize=_args.batch_size, dep=train_dep, weighted=_args.weighted)
    else:
        _args.train_func(_args, _args.f_cost, _args.f_update, param['epoch_id'], param['clr'], _args.nsentences, train_lex, train_obj, train_y, batchsize=_args.batch_size, dep=train_dep, weighted=_args.weighted)


def run_epochs(_args, multi_input=False, test_data=True, best_f1 = -numpy.inf):
    #best_f1 = -numpy.inf

    param = dict(clr = _args.lr, ce = 0, be = 0, epoch_id = -1)
    cargs = compile_circuit(_args)
    settings = {"lr:": _args.lr, "emb_dropout_rate": _args.emb_dropout_rate, "lstm_dropout_rate": _args.lstm_dropout_rate, "MLP_hidden_out_dim": _args.MLP_hidden_out_dim, "minimum_lr": _args.minimum_lr, 'L2Reg_reg_weight': _args.L2Reg_reg_weight, 'nepochs': _args.nepochs, "decay_epochs": _args.decay_epochs, 'current': _args.current, 'total': _args.total}

    train_accuracy_record = []
    valid_accuracy_record = []

    test_accuracy_record = []

    while param['epoch_id']+1 < _args.nepochs:
        param['epoch_id'] += 1
        #print settings

        run_training(_args, param, multi_input)
        train_lex, train_y, train_obj, train_dep = _args.train_set
        valid_lex, valid_y, valid_obj, valid_dep = _args.valid_set

        if test_data:
            test_lex, test_y, test_obj, test_dep = _args.test_set
        if multi_input:
            res_train, _ = predict(_args, _args.f_classify, *(train_lex + [train_idxs, train_y]), batchsize=_args.batch_size, dep=train_dep, weighted=_args.weighted)
            res_valid, _ = predict(_args, _args.f_classify, *(valid_lex + [valid_idxs, valid_y]), batchsize=_args.batch_size, dep=valid_dep, weighted=_args.weighted)
        else:
            res_train, _ = predict(_args, _args.f_classify, train_lex, train_obj, train_y, batchsize=_args.batch_size, dep=train_dep, weighted=_args.weighted)
            res_valid, _ = predict(_args, _args.f_classify, valid_lex, valid_obj, valid_y, batchsize=_args.batch_size, dep=valid_dep, weighted=_args.weighted)
        if _args.verbose:
            print('TEST: epoch', param['epoch_id'],
                  'train performances'   , res_train,
                  'valid performances'   , res_valid)
        # If this update created a 'new best' model then save it.


        if type(res_valid) is tuple:
            curr_f1 = res_valid[-1]
        else:
            curr_f1 = res_valid
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            param['be']  = param['epoch_id']
            param['last_decay'] = param['be']
            param['vf1'] = res_valid
            param['trainf1'] = res_train
            param['best_classifier'] = _args.f_classify
            param['decay_epochs'] = _args.decay_epochs
            param['MLP_hidden_out_dim'] = _args.MLP_hidden_out_dim
            param['emb_dropout_rate'] = _args.emb_dropout_rate
            param['lstm_dropout_rate'] = _args.lstm_dropout_rate

            #cargs['f_classify'] = _args.f_classify
            #save_parameters(_args.parameters_file, cargs)
            if test_data:
                if multi_input:
                    res_test, _ = predict(_args, _args.f_classify, *(test_lex + [test_idxs, test_y]), batchsize=_args.batch_size, dep=test_dep, weighted=_args.weighted, print_prediction=_args.print_prediction, prediction_file=_args.prediction_file)
                else:
                    res_test, _ = predict(_args, _args.f_classify, test_lex, test_obj, test_y, batchsize=_args.batch_size, dep=test_dep, weighted=_args.weighted, print_prediction=_args.print_prediction, prediction_file=_args.prediction_file)
                # get the prediction, convert and write to concrete.
                param['tf1'] = res_test
                print '\nEpoch:%d'%param['be'], 'Test accuracy:', res_test, '\n'
                ############## Test load parameters!!!! ########
                #cargs = {}
                #print "loading parameters!"
                #load_params(_args.parameters_file, cargs)
                #f_classify = cargs['f_classify']
                #res_test, _ = predict(_args, test_lex, test_idxs, f_classify, test_y, _args.batch_size, _args.graph, test_dep, _args.weighted)
                #print 'Load parameter test accuracy:', res_test, '\n'
        train_accuracy_record.append(res_train)
        valid_accuracy_record.append(res_valid)
        test_accuracy_record.append(res_test)

        #print "train accuracy: ", train_accuracy_record.__len__()
        #print "valid accuracy: ", valid_accuracy_record.__len__()
        #print "test accuracy: ", test_accuracy_record.__len__()
                ############## End Test ##############
        if _args.decay and (param['epoch_id'] - param['last_decay']) >= _args.decay_epochs:
            print 'learning rate decay at epoch', param['epoch_id'], '! Previous best epoch number:', param['be']
            param['last_decay'] = param['epoch_id']
            param['clr'] *= 0.5
        # If learning rate goes down to minimum then break.
        if param['clr'] < _args.minimum_lr:
            print "\nLearning rate became too small, breaking out of training"
            break

    print('BEST RESULT: epoch', param['be'],
          'valid accuracy', param['vf1'], 'lr', _args.lr, 'emb_dropout_rate' , _args.emb_dropout_rate, 'lstm_dropout_rate',_args.lstm_dropout_rate
          )




def combine_word_dicts(dict1, dict2):
    print 'the size of the two dictionaries are:', len(dict1), len(dict2)
    combine_dict = dict1.copy()
    for k, v in dict2.items():
        if k not in combine_dict:
            combine_dict[k] = len(combine_dict)
    print 'the size of the combined dictionary is:', len(combine_dict)
    return combine_dict

# load and setup embedding shared variable for multiple embeddings
def init_emb_multi(_args):
    _args.emb_matrices = []
    # initialize word embeddings
    if _args.emb_dir != 'RANDOM':
        print 'started loading embeddings from file', _args.emb_dir
        M_emb, _ = read_matrix_from_file(_args.emb_dir, _args.global_word_map)
        print 'global map size:', len(M_emb), len(_args.global_word_map)
        ## load pretrained embeddings
    else:
        print 'random initialize the embeddings!'
        M_emb = numpy.random.rand(len(_args.global_word_map)+2, _args.emb_out_dim)
    _args.emb_matrices.append(theano.shared(M_emb, name='wemb_matrix') )
    # add pos embeddings
    P_emb = numpy.random.rand(len(_args.dicts['poss2idx']), _args.pos_emb_dim)
    _args.emb_matrices.append(theano.shared(P_emb, name='pemb_matrix') )
    _args.emb_out_dim = M_emb.shape[1] + P_emb.shape[1]
    if _args.fine_tuning :
        print 'fine tuning!!!!!'
        for matrix in _args.emb_matrices:
            matrix.is_trainable= True

# load and setup embedding shared variable
def init_emb(_args):

    if _args.emb_dir != 'RANDOM':
        print 'started loading embeddings from file', _args.emb_dir

        #print 'global map size:', len(M_emb), len(_args.global_word_map)
        ## load pretrained embeddings
    #_args.global_word_map = _args.dicts['objs2idx']
        Q_emb, _ = read_matrix_from_file(_args.emb_dir, _args.dicts['words2idx'])
        _args.qemb_matrix = theano.shared(Q_emb, name='qemb_matrix')
        O_emb, _ = read_matrix_from_file(_args.emb_dir, _args.dicts['objs2idx'])
        _args.oemb_matrix = theano.shared(O_emb, name='oemb_matrix')
        _args.emb_dim = len(Q_emb[0])
        _args.emb_out_dim = _args.emb_dim
        _args.question_emb_out_dim = _args.emb_dim
        _args.object_emb_out_dim = _args.emb_dim
        _args.attention_out_dim = _args.question_lstm_out_dim*2 + _args.object_emb_out_dim
        if _args.fine_tuning :
            print 'fine tuning!!!!!'
            _args.qemb_matrix.is_trainable= True
            _args.oemb_matrix.is_trainable= True

### convert the old word idx to the new one ###
def convert_word_idx(corpus_word, idx2word_old, word2idx_new):
    # print type(corpus_word), len(corpus_word)
    #if len(corpus_word) == 0:
    #    print 'empty word array!!!!'
    #    return corpus_word
    if type(corpus_word[0]) is int:
        # print [(idx, idx2word_old[idx], word2idx_new[idx2word_old[idx]]) for idx in corpus_word]
        return [word2idx_new[idx2word_old[idx]] for idx in corpus_word]
    else:
        return [convert_word_idx(line, idx2word_old, word2idx_new) for line in corpus_word]


def data_prep_shareLSTM(_args):
    _args.rng = numpy.random.RandomState(_args.seed)
    dataSets, _args.lr_arr, dataset_map = load_all_data_multitask(_args)
    if 'arcs2idx' in dataSets[0][-1]:
        _args.lstm_arc_types = len(dataSets[0][-1]['arcs2idx'])
        print 'lstm arc types =', _args.lstm_arc_types
    ## re-map words in the news cws dataset
    idx2word_dicts = [dict((k, v) for v, k in ds[-1]['words2idx'].iteritems()) for ds in dataSets]
    _args.idx2label_dicts = [dict((k, v) for v, k in ds[-1]['labels2idx'].iteritems()) for ds in dataSets]
    for i, ds in enumerate(dataSets):
        # ds structure: train_set, valid_set, test_set, dicts
        print len(ds[0]), len(ds[0][0][-1]), ds[0][1][-1], ds[0][2][-1]
        print len(ds[1]), len(ds[1][0][-1]), ds[1][1][-1], ds[1][2][-1]
        print len(ds[2]), len(ds[2][0][-1]), ds[2][1][-1], ds[2][2][-1]
        ds[0][0], ds[1][0], ds[2][0] = batch_run_func((ds[0][0], ds[1][0], ds[2][0]), convert_word_idx, idx2word_dicts[i], _args.global_word_map)
        ## convert word, feature and label for array to numpy array
        ds[0], ds[1], ds[2] = batch_run_func((ds[0], ds[1], ds[2]), conv_data_graph, _args.win_l, _args.win_r)
        check_input(ds[0][:3], len(_args.global_word_map))
        check_input(ds[1][:3], len(_args.global_word_map))
    '''Probably want to move part of the below code to the common part.'''
    _args.trainSet = [ds[0] for ds in dataSets]
    _args.devSet = [ds[1] for ds in dataSets]
    _args.testSet = [ds[2] for ds in dataSets]
    _args.nsentences_arr = [len(ds[0]) for ds in _args.trainSet]
    #train_lex, train_idxs, train_y = _args.trainSet[0]
    #print 'in data_prep_shareLSTM', train_lex[0], train_idxs[0], train_y[0]
    if _args.sample_coef != 0:
        _args.nsentences_arr[0] = int(_args.sample_coef * _args.nsentences_arr[-1])
        #_args.nsentences_arr[1] = int(_args.sample_coef * _args.nsentences_arr[-1])
        #_args.nsentences_arr[2] = int(_args.sample_coef * _args.nsentences_arr[-1])


def check_input(dataset, voc_size, multi_input=False):
    lex, label, idxv = dataset
    if multi_input:
        elems = lex + [label, idxv]
    else:
        elems = dataset
    for i, els in enumerate(zip(*elems)):
        sent_len = len(els[0])
        for ii in els[-1]:
            try:
            	assert numpy.all(numpy.array(ii) < sent_len) and numpy.all(ii > -1)
            except:
                print 'abnormal index:', ii, 'at instance', i, 'sentence length:', sent_len
        #try:
        #    assert (els[-2] == 0 or els[-2] == 1)
        #except:
        #    print 'abnormal label:', els[-2]
        try:
            assert numpy.all(numpy.array(els[0]) < voc_size)
        except:
            print 'abnormal input:', els[0]

def run_single_corpus(_args):
    _args.rng = numpy.random.RandomState(_args.seed)
    _args.loaddata = load_data #_withAnnos #_cv
    if 'Graph' in _args.circuit:
        _args.graph = True
    if 'Add' in _args.circuit:
        _args.add = True
    if 'Weighted' in _args.circuit:
        _args.weighted = True
    # For the GENIA experiment
    _args.train_set, _args.valid_set, _args.test_set, _args.dicts = _args.loaddata(_args)

    #return

	#.train_path, _args.valid_path, num_entities=_args.num_entity, dep=_args.graph, train_dep=_args.train_graph, valid_dep=_args.valid_graph, add=_args.add)
    # For the n-ary experiments
    #_args.train_set, _args.valid_set, _args.test_set, _args.dicts = _args.loaddata(_args.data_dir, _args.total_fold, _args.dev_fold, _args.test_fold, num_entities=_args.num_entity, dep=_args.graph, content_fname=_args.content_file, dep_fname=_args.dependent_file, add=_args.add)
    # convert the data from array to numpy arrays

    _args.train_set, _args.valid_set, _args.test_set = batch_run_func((_args.train_set, _args.valid_set, _args.test_set), conv_data, _args.win_l, _args.win_r) #, True)

    _args.train_set[1] = map(lambda x: _args.dicts['labels2idx'][x], _args.train_set[1])
    _args.valid_set[1] = map(lambda x: _args.dicts['labels2idx'][x], _args.valid_set[1])

    _args.test_set[1] = map(lambda x: _args.dicts['labels2idx'][x], _args.test_set[1])

    print 'word dict size:', len(_args.dicts['words2idx'])
    #print 'checking training data!'
    #check_input(_args.train_set[:3], len(_args.dicts['words2idx']), multi_input=True)
    #print 'checking test data!'
    #check_input(_args.valid_set[:3], len(_args.dicts['words2idx']), multi_input=True)
    #print 'finish check inputs!!!'
    #_args.global_word_map = _args.dicts['objs2idx']
    prepare_corpus(_args)
    #for k, v in word2idx.iteritems():
    #    print k, v
    init_emb(_args) #_multi(_args)
    run_epochs(_args) #, multi_input=True)


def run_corpora_multitask(_args):
    if 'Graph' in _args.circuit:
        _args.graph = True
    if 'Add' in _args.circuit:
        _args.add = True
    if 'Weighted' in _args.circuit:
        _args.weighted = True
    data_prep_shareLSTM(_args)
    prepare_params_shareLSTM(_args)
    _args.f_costs_and_updates, _args.f_classifies, cargs = create_multitask_relation_circuit(_args, StackConfig, len(_args.trainSet))
    print "Finished Compiling"
    run_multi_task(_args, cargs, 1, len(_args.trainSet), mode=_args.train_mode) #, test_data=True)


def create_arg_parser(args=None):
    _arg_parser = argparse.ArgumentParser(description='LSTM')
    add_arg.arg_parser = _arg_parser
    ## File IO
    # For single task
    add_arg('--data_dir'        , '.')
    # For multitask
    add_arg('--drug_gene_dir'        , '.')
    add_arg('--drug_variant_dir'        , '.')
    add_arg('--drug_gene_variant_dir'         , '.')
    # End for multitask
    # For wild prediction
    add_arg('--train_path'         , '.')
    add_arg('--valid_path'         , '.')
    add_arg('--featurelist_path'         , '.')
    add_arg('--train_graph'         , '.')
    add_arg('--valid_graph'         , '.')
    add_arg('--content_file'         , 'sentences')
    add_arg('--dependent_file'         , 'graph_arcs')
    add_arg('--parameters_file'         , 'best_parameters')
    add_arg('--prediction_file'         , 'prediction')
    add_arg('--drug_gene_prediction_file'         , '.')
    add_arg('--drug_var_prediction_file'         , '.')
    add_arg('--triple_prediction_file'         , '.')
    add_arg('--num_entity'        , 2)
    add_arg('--total_fold'        , 10)
    add_arg('--dev_fold'        , 0)
    add_arg('--test_fold'        , 1)
    add_arg('--circuit'         , 'LSTMRelation')
    add_arg('--emb_dir'       , '../treelstm/data', help='The initial embedding file name for cws')
    add_arg('--emb_dropout_rate'    , 0.0, help='Dropout rate for the input embedding layer')
    add_arg('--lstm_dropout_rate'    , 0.0, help='Dropout rate for the lstm output embedding layer')
    add_arg('--representation'      , 'charpos', help='Use which representation')
    add_arg('--fine_tuning'   , True)
    add_arg('--feature_thresh'     , 0)
    add_arg('--graph'    , False)
    add_arg('--weighted'         , False)
    add_arg('--add'         , False)
    add_arg('--print_prediction'    , True)
    add_arg('--label_path'    , "vqa_data")
    add_arg('--current'    , 0)
    add_arg('--total'    , 0)
    ## Task
    add_arg('--task'    , 'news_cws')
    add_arg('--oovthresh'     , 0    , help="The minimum count (upto and including) OOV threshold for NER") # Maybe 1 ?
    ## Training
    add_arg_to_L(TRAIN_PARAM, '--cost_coef'           , 0.0)
    add_arg_to_L(TRAIN_PARAM, '--sample_coef'           , 0.0)
    add_arg_to_L(TRAIN_PARAM, '--batch_size'           , 1)
    add_arg_to_L(TRAIN_PARAM, '--train_mode'           , 'alternative')
    add_arg_to_L(TRAIN_PARAM, '--lr'           , 0.01)
    add_arg_to_L(TRAIN_PARAM, '--dg_lr'           , 0.005)
    add_arg_to_L(TRAIN_PARAM, '--dv_lr'           , 0.005)
    add_arg_to_L(TRAIN_PARAM, '--dgv_lr'           , 0.005)
    add_arg_to_L(TRAIN_PARAM, '--nepochs'      , 50)
    add_arg_to_L(TRAIN_PARAM, '--optimizer'    , 'sgd', help='sgd or adadelta')
    add_arg_to_L(TRAIN_PARAM, '--seed'         , 1) #int(random.getrandbits(10)))
    add_arg_to_L(TRAIN_PARAM, '--decay'        , True,  help='whether learning rate decay')
    add_arg_to_L(TRAIN_PARAM, '--decay_epochs' , 5)
    add_arg_to_L(TRAIN_PARAM, '--minimum_lr'   , 1e-5)
    ## Topology
    add_arg_to_L(TOPO_PARAM, '--emission_trans_out_dim',    -1)
    add_arg_to_L(TOPO_PARAM, '--crf_viterbi',   False)
    add_arg_to_L(TOPO_PARAM, '--lstm_win_size',               5)
    add_arg_to_L(TOPO_PARAM, '--emb_out_dim',               300)
    add_arg_to_L(TOPO_PARAM, '--pos_emb_dim',               25)
    add_arg_to_L(TOPO_PARAM, '--lstm_out_dim',                150)
    add_arg_to_L(TOPO_PARAM, '--question_lstm_out_dim',                150)
    add_arg_to_L(TOPO_PARAM, '--CNN_out_dim',                 500)
    add_arg_to_L(TOPO_PARAM, '--lstm_type_dim',                50)
    add_arg_to_L(TOPO_PARAM, '--MLP_hidden_out_dim',                1000)
    add_arg_to_L(TOPO_PARAM, '--MLP_activation_fn',                'tanh')
    add_arg_to_L(TOPO_PARAM, '--L2Reg_reg_weight',             0.0)
    add_arg_to_L(TOPO_PARAM, '--win_l',                          0)
    add_arg_to_L(TOPO_PARAM, '--win_r',                          0)
    ## DEBUG
    add_arg('--verbose'      , 2)

    return _arg_parser


if __name__ == "__main__":
    #######################################################################################
    ## PARSE ARGUMENTS, BUILD CIRCUIT, TRAIN, TEST
    #######################################################################################
    TOPO_PARAM = []
    TRAIN_PARAM = []
    _arg_parser = create_arg_parser()
    args = _arg_parser.parse_args()
    #run_wild_test(args)
    run_single_corpus(args)
    #run_corpora_multitask(args)
