#!/bin/bash
i=0.001
while [ "$i" -lt 0.1 ]
do
        echo "i: $i"
        i=`expr $i + 0.09`
done