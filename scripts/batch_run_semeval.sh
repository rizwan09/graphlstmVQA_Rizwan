#!/bin/bash

for ds in origTextwithNERWNabbr; do
    for lr in 0.03 0.05 0.07; do
        for reg in 2e-4 1e-4; do
            for temb in 5 10; do
                for wdo in 0.0 0.1 0.3 0.5; do
                    for ldo in 0.0 0.1; do
                        qsub -l "mem_free=1g,ram_free=1g" ./scripts/run_semeval.sh cpu $1 ${ds} ${lr} ${reg} ${temb} ${wdo} ${ldo}
                        sleep 30s
                    done
                done
            done
        done
    done
done
