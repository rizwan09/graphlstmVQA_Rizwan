#!/bin/bash

for i in {0..4}; do
    echo ${i}
    qsub -l h_rt=36:00:00,gpu=1 -q gpu.q run_triple_multitask.sh ${i} $1  
done
