#!/bin/bash

cd /home/npeng/graphLSTM/graphLSTM

PP_DIR=/export/a12/npeng/bioNLP/Nary_multitask_param_and_predictions_20160926

#THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math,exception_verbosity=high time python theano_src/lstm_RE.py --drug_gene_dir /export/a12/npeng/bioNLP/n_ary_relations/data_drug_gene/feature_folds_5fld/ --drug_variant_dir /export/a12/npeng/bioNLP/n_ary_relations/data_drug_var/feature_folds_5fld/ --drug_gene_variant_dir /export/a12/npeng/bioNLP/n_ary_relations/data_triple/feature_folds_5fld/  --emb_dir /export/a12/npeng/bioNLP/glove/glove.6B.100d.txt --total_fold 5 --dev_fold $1 --test_fold $1  --circuit $2 --batch_size 8  --dg_lr 0.002 --dv_lr 0.002 --dgv_lr 0.002  --lstm_type_dim 2 --content_file sentences_withName --dependent_file graph_arcs_withName --parameters_file ${PP_DIR}/all_triple_best_params_$2.cv$1.lr0.02.bt8 --prediction_file ${PP_DIR}/all_triple_$2.cv$1.predictions --print_prediction True --sample_coef 1.0 > /export/a12/npeng/bioNLP/Nary_multitask_results_20160927/all_triple.accuracy.$2.cv$1.lr0.02.bt8

THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math,exception_verbosity=high time python theano_src/lstm_RE.py --drug_gene_dir /export/a12/npeng/bioNLP/n_ary_relations/data_drug_gene/feature_folds_5fld/ --drug_variant_dir /export/a12/npeng/bioNLP/n_ary_relations/data_drug_var/feature_folds_5fld/ --drug_gene_variant_dir /export/a12/npeng/bioNLP/n_ary_relations/data_triple/feature_folds_5fld/  --emb_dir /export/a12/npeng/bioNLP/glove/glove.6B.100d.txt --total_fold 5 --dev_fold $1 --test_fold $1  --circuit $2 --batch_size 8 --lr 0.02 --lstm_type_dim 2 --content_file sentences_2nd --dependent_file graph_arcs --parameters_file  ${PP_DIR}/all_triple_best_params_$2.cv$1.lr0.02.bt8 --drug_gene_prediction_file ${PP_DIR}/drug_gene_$2.cv$1.predictions --drug_var_prediction_file ${PP_DIR}/drug_var_$2.cv$1.predictions --triple_prediction_file ${PP_DIR}/triple_$2.cv$1.predictions --print_prediction True --sample_coef 1.0 > /export/a12/npeng/bioNLP/Nary_multitask_results_20160927/all_triple.accuracy.$2.cv$1.lr0.02.bt8
