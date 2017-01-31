#!/bin/bash

cd /home/npeng/graphLSTM/graphLSTM

PP_DIR=/export/a12/npeng/bioNLP/post_internship_experiments/Nary_param_and_predictions_20160923

#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math,exception_verbosity=high time python theano_src/lstm_RE.py --data_dir /export/a12/npeng/bioNLP/n_ary_relations/data_drug_var/feature_folds_5fld/  --emb_dir /export/a12/npeng/bioNLP/glove/glove.6B.100d.txt --total_fold 5 --dev_fold $1 --test_fold $1 --num_entity 2 --circuit $2 --batch_size 8 --lr 0.02 --lstm_type_dim 2 --content_file sentences_single --dependent_file graph_arcs_single --cost_coef 0.0 > /export/a12/npeng/bioNLP/Nary_results_20160920/all_drug_var.accuracy.$2.cv$1.lr0.02.bt8
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math,exception_verbosity=high time python theano_src/lstm_RE.py --data_dir /export/a12/npeng/bioNLP/n_ary_relations/data_drug_var/feature_folds_5fld/  --emb_dir /export/a12/npeng/bioNLP/glove/glove.6B.100d.txt --total_fold 5 --dev_fold $1 --test_fold $1 --num_entity 2 --circuit $2 --batch_size 8 --lr 0.01 --lstm_type_dim 3 --content_file sentences_2nd --dependent_file graph_arcs --parameters_file ${PP_DIR}/all_drug_var_best_params_$2.cv$1.lr0.02.bt8 --prediction_file ${PP_DIR}/all_drug_var_$2.cv$1.arc3.predictions > /export/a12/npeng/bioNLP/post_internship_experiments/Nary_results_20160923/all_drug_var.accuracy.$2.cv$1.lr0.02.bt8
