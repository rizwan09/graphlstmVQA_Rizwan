#!/bin/bash

for i in {0..4}; do
    echo ${i}
    qsub -l gpu=1 run_lstm.sh ${i} $1  #WeightedAddGraph
    #qsub -l gpu=1 run_drug_var.sh ${i} $1  #WeightedAddGraph
done
