#!/bin/bash

cd /home/npeng/graphLSTM/graphLSTM

DATA_PATH=/export/a12/npeng/RelationExtraction/jointERE/semeval_concrete4.4
OUT_DIR=/export/a12/npeng/RelationExtraction/jointERE/results/semeval_tune_20161212

THEANO_FLAGS=mode=FAST_RUN,device=$1,floatX=float32,nvcc.flags=-use_fast_math,exception_verbosity=high time python theano_src/lstm_RE.py --train_path ${DATA_PATH}/SemEval.train.ner.comm.relations.${3} --train_graph ${DATA_PATH}/SemEval.train.ner.comm.docgraph.origText --valid_path ${DATA_PATH}/SemEval.test.ner.comm.relations.${3} --valid_graph ${DATA_PATH}/SemEval.test.ner.comm.docgraph.origText --emb_dir /export/a12/npeng/bioNLP/glove/glove.6B.100d.txt --lstm_out_dim 150 --num_entity 2  --circuit ${2}Relation --batch_size 1 --lr ${4} --L2Reg_reg_weight ${5} --pos_emb_dim ${6} --emb_dropout_rate ${7} --lstm_dropout_rate ${8} | tee ${OUT_DIR}/${2}/${2}_semeval_${3}_9class_result.emb100.h150.lr${4}.reg${5}.tagemb${6}.embdo${7}.lstmdo${8}
