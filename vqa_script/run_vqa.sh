#!/bin/bash

DATA_PATH=vqa_data

#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math time \
#    python2.7 theano_src/lstm_VQA.py --train_path ${DATA_PATH}/train_ --valid_path ${DATA_PATH}/val_ --featurelist_path ${DATA_PATH}/featureList.txt \
#    --emb_dir ${DATA_PATH}/glove.6B.100d.txt  --batch_size 8 --lr 0.01 --circuit Question  | tee Question_vqa_results

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math,exception_verbosity=high time \
    python2.7 theano_src/lstm_VQA.py --train_path ${DATA_PATH}/train_ --valid_path ${DATA_PATH}/val_ --featurelist_path ${DATA_PATH}/featureList.txt \
    --emb_dir /if1/kc2wc/data/glove/glove.6B.300d_w_header.txt  --batch_size 8 --lr 0.01 --emb_dropout_rate 0.03 --lstm_dropout_rate 0.03   --circuit VQA | tee Object_vqa_results

#THEANO_FLAGS=mode=DebugMode,device=gpu,floatX=float32,nvcc.flags=-use_fast_math,exception_verbosity=high time \
#    python2.7 theano_src/lstm_VQA.py --train_path ${DATA_PATH}/train_ --valid_path ${DATA_PATH}/val_ --featurelist_path ${DATA_PATH}/featureList.txt \
#    --emb_dir ${DATA_PATH}/glove.6B.100d.txt  --batch_size 8 --lstm_out_dim 100 --lr 0.01 --circuit VQA  | tee Object_vqa_results
#--win_l -1 --win_r 1

#--win_l -1 --win_r 1
