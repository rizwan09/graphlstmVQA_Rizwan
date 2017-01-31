#!/bin/bash

cd /home/npeng/graphLSTM/graphLSTM

PP_DIR=/export/a12/npeng/bioNLP/PubMed_param_and_predictions

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math,exception_verbosity=high time python theano_src/lstm_RE.py --train_path /export/a12/npeng/bioNLP/features/candidate_sliding-3/triple_sentences.train --train_graph /export/a12/npeng/bioNLP/features/candidate_sliding-3/triple_graph_arcs.train --valid_path /export/a12/npeng/bioNLP/features/candidate_sliding-3/triple_sentences --valid_graph /export/a12/npeng/bioNLP/features/candidate_sliding-3/triple_graph_arcs --emb_dir /export/a12/npeng/bioNLP/glove/glove.6B.100d.txt --total_fold 5 --num_entity 3 --circuit $1 --batch_size 8 --lr 0.02 --lstm_type_dim 2 --parameters_file ${PP_DIR}/all_triple_best_params_$1.lr0.02.bt8 --prediction_file ${PP_DIR}/all_triple_$1.allwords.predictions --nepochs 7 > ${PP_DIR}/triple.${1}.PubMed.allwords.results
