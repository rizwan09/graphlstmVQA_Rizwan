#!/bin/bash

DATA_PATH=ACE05

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math,exception_verbosity=high time python2.7 theano_src/lstm_RE.py --train_path ${DATA_PATH}/split_train.relations.originText --train_graph ${DATA_PATH}/split_train.docgraph --valid_path ${DATA_PATH}/split_dev.relations.originText --valid_graph ${DATA_PATH}/split_dev.docgraph --emb_dir glove.6B.100d.txt --num_entity 2  --circuit ${1}Relation --batch_size 8 --lr 0.01 | tee ${1}_multiclass_origText_result
