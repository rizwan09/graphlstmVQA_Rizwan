# Preprocess the data graphs to get sentences and denpendencies and mismatched entities
for f in {0..4}; do  python theano_src/data_process.py ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/data_graph ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/sentences  ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/dependency_labels 2>  ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/Entity_with_no_dependencies ; done

# Gather the data-processing exceptions
for f in {0..4}; do cat ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/Entity_with_no_dependencies >> ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/Entity_with_no_dependencies; done

# Get stats of the data-processing exceptions
for  f in {0..4}; do grep WARNING ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/Entity_with_no_dependencies | wc -l ; done

# Gather training data  ==> deprecated, will load them on the fly
for f in {0..4}; do for i in $(seq 0 $((f-1))); do cat ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${i}/sentences >> ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/train_cv_${f}; done; for i in $(seq $((f+1)) 4); do  cat ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${i}/sentences >> ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/train_cv_${f}; done; done

# Run linear-chain BiLSTM ==> deprecated, will change to the cv setting
for f in {0..4}; do python theano_src/lstm_RE.py --train_path ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/train_cv_${f} --valid_path ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/sentences --test_path ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/sentences  --emb_dir ../treelstm/data/glove/glove.6B.100d.txt --win_r 2 --wemb1_out_dim 100 > ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/BiLSTM_train_cv_${f}.stdout ; done 

for f in {0..4}; do  python theano_src/data_process.py toy_data/${f}/data_split_0_graph toy_data/${f}/sentences toy_data/${f}/dlabels; done

for f in {0..4}; do python theano_src/lstm_RE.py --train_path ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/train_cv_no_entity_${f} --valid_path ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/sentences_no_entity --test_path ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/sentences_no_entity  --emb_dir ../spinn/glove/glove.6B.100d.txt --win_r 2 --wemb1_out_dim 100 | tee ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/BiLSTM_train_cv_noEntity_${f}.stdout ; done


# The final command for training the BiLSTM:
python theano_src/lstm_RE.py --train_path ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/train_cv_no_entity_${f} --valid_path ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/sentences_no_entity --test_path ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/sentences_no_entity  --emb_dir ../spinn/glove/glove.6B.100d.txt --win_r 2 --wemb1_out_dim 100 | tee ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/BiLSTM_train_cv_noEntity_${f}.stdout

python theano_src/lstm_RE.py --train_path ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/train_cv_no_entity_${f} --valid_path ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/sentences_no_entity --test_path ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/data_folds/0${f}/sentences_no_entity  --emb_dir ../spinn/glove/glove.6B.100d.txt --win_r 2 --wemb1_out_dim 100 --lr 0.005 | tee ~/gcr/scratch/RR1/t-napeng/experiments/BiLSTM/3_deponly_3paths/BiLSTM_train_cv_noEntity_${f}.stdout


python theano_src/data_process.py ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_orig/pos_graph  ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_lstm/all_pos_sentences  ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_lstm/all_pos_dlabels 2>  ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/tree_construction_exceptions

# Generate candidate
python python_src/gen_data.py  ~/gcr/scratch/RR1/hoifung/data/pmc/discourse/candidate_sliding-3/ data/knowledge_base/01-May-2016-ClinicalEvidenceSummaries.tsv data/knowledge_base/Knowledge_database_v12.0.txt testpos testneg

# Split into 5-folds
 python python_src/gen_data.py ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_drug_gene_origin/neg_all  ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_drug_var_origin/neg_all  ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_triple_origin/neg_all

# Split the data to run Chris's document graph generation code
 for dir in data_drug_gene data_drug_var data_triple; do for f in {0..4}; do for pn in pos neg; do split -dl 2000 ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/data_folds/${f}/${pn} ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/data_folds/${f}/${pn}_split_; done; done; done

# Collect the separate graphs into a single file (for drug-gene pair, because it's large)
for f in {0..4}; do for fl in ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_drug_gene/feature_folds/${f}/*_split_*; do cat ${fl} >> ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_drug_gene/feature_folds/${f}/data_graph; done; done

# Collect the separate graphs into a single file (for drug-variant pair and drug-gene-variant triple, relatively small)
for dir in data_drug_var data_triple; do for f in {0..4}; do for pn in pos neg; do cat ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/feature_folds/${f}/${pn}_graph >> ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/feature_folds/${f}/data_graph; done; done; done

# new command for running multi-task learning experiments:
time python theano_src/lstm_RE.py --drug_gene_dir ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_drug_gene/feature_folds/ --drug_variant_dir  ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_drug_var/feature_folds/  --drug_gene_variant_dir ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_triple/feature_folds/ --circuit LSTMRelation_multitask --sample_coef 1 --emb_dir ../spinn/glove/glove.6B.100d.txt --dev_fold 0 | tee ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/multi_task_cv_0

# generate graph file
for dir in data_drug_gene data_drug_var data_triple; do for f in {0..4}; do python theano_src/data_process.py  gen_graph_from_json ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/feature_folds/${f}/data_graph  ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/feature_folds/${f}/graph_arcs 2>>  ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/gen_graph_errors; done; done

# run unweighted graph-lstm on GPU:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math time python theano_src/lstm_RE.py --data_dir ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_triple/feature_folds/  --emb_dir ../spinn/glove/glove.6B.100d.txt --dev_fold 0 --num_entity 3 --circuit GraphLSTMRelation --batch_size 8 --graph True | tee debug.graphLSTM.batched

#######################################################################################################################################################################
#--------------------------------------------------------------- Experimental commands Record Below-------------------------------------------------------------------#
#######################################################################################################################################################################

# Generate candidate
# Note I have several functions in the gen_data.py, need to change the main function accordingly
# This part uses the whole main function except the first two lines 
python python_src/gen_data.py  ~/gcr/scratch/RR1/hoifung/data/pmc/discourse/candidate_sliding-3/ data/knowledge_base/01-May-2016-ClinicalEvidenceSummaries.tsv data/knowledge_base/Knowledge_database_v12.0.txt testpos testneg triple

# A new round: split the data (But I actually used the old splits.):
# Note I have several functions in the gen_data.py, need to change the main function accordingly
# This part only uses the function gen_split_train. 
python python_src/gen_data.py ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_drug_gene/data_orig/neg_all ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_drug_var/data_orig/neg_all ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_triple/data_orig/neg_all 10

# prepare the data for graph generation:
for dir in data_drug_gene data_drug_var data_triple; do for i in {0..9}; do mkdir -p ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/feature_folds/${i} ; done ; done
for dir in data_drug_gene data_drug_var data_triple; do for i in {0..9}; do for pn in pos neg; do split -dl 2000 ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/data_folds/${i}/${pn} ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/data_folds/${i}/${pn}; done; done; done

# Then run on gui chris's script, see C:\Users\t-napeng\Documents\generate_codument_graph.xml
# merge generated graphs and prepare data for LSTM:
    # merge graphs
for dir in data_drug_gene data_drug_var data_triple; do for i in {0..9}; do for pn in pos neg; do cat ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/feature_folds/${i}/${pn}*_graph >> ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/feature_folds/${i}/all_data_graph; done; done; done
    # generate shortest path for LSTM:
for i in {0..9}; do python theano_src/data_process.py gen_chain_shortest_paths ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_drug_var/feature_folds/${i}/all_data_graph ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_drug_var/feature_folds/${i}/shortest_path ; done 
    # generate the whole-sentence graph structure for LSTM:
for dir in data_drug_gene data_drug_var data_triple; do for i in {0..9}; do python theano_src/data_process.py gen_graph_from_json ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/feature_folds/${i}/all_data_graph ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/feature_folds/${i}/sentences  ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/${dir}/feature_folds/${i}/graph_arcs ; done; done

# Run experiments on the linux box:
    # shortest-path baseline
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math time python theano_src/lstm_RE.py --data_dir ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_drug_var/feature_folds/  --emb_dir ../spinn/glove/glove.6B.100d.txt --dev_fold 0 --test_fold 1 --num_entity 2 --circuit WeightedGraphLSTMRelation --batch_size 8 --lr 0.01 --content_file shortest_path 
    # Bi-LSTM baseline
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math time python theano_src/lstm_RE.py --data_dir ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_drug_var/feature_folds/  --emb_dir ../spinn/glove/glove.6B.100d.txt --dev_fold 0 --test_fold 1 --num_entity 2 --circuit WeightedGraphLSTMRelation --batch_size 8 --lr 0.01 --content_file sentences


# Collect results:
tail -n 15 ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_triple/LSTM/*lr0.02_cv*out | grep "valid accuracy"
tail -n 15 ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_triple/LSTM/*lr0.02_drop*_cv*out | grep "valid accuracy"

# run a single job on the linux boxes command:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math time python theano_src/lstm_RE.py --data_dir ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_triple/feature_folds_5fld/  --emb_dir ../spinn/glove/glove.6B.100d.txt --dev_fold 1 --num_entity 3 --circuit WeightedGraphLSTMRelation --batch_size 8 --lr 0.01 | tee debug.weightedGraphLSTM.addweights

# I saved some xml files for running jobs on GCRGPU at 

# Final command for the single-task setting:
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-use_fast_math,exception_verbosity=high time python theano_src/lstm_RE.py --data_dir ~/gcr/scratch/RR1/t-napeng/experiments/n_ary_relations/data_triple/feature_folds_5fld/  --emb_dir ../spinn/glove/glove.6B.100d.txt --total_fold 5 --dev_fold 0 --test_fold 0 --num_entity 3 --circuit (Weighted)(Add)(Graph)LSTMRelation --batch_size 8 --lr 0.02 --lstm_type_dim 2 --content_file sentences_2nd --cost_coef 0.0
# (For the different circuits you can try, please refers to theano_src/neural_architectures.py)

# Final command for multi-task setting:

