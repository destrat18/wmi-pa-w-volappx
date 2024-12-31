#!/bin/bash

SYN_DIR=synthetic_exp
DATA_DIR=$SYN_DIR/data

for dir in $(ls -d $DATA_DIR/*)
do
	res_dir=$(sed "s+data+results+g" <<< $dir)
	mkdir -p $res_dir
	echo Evaluating $dir

    echo "Mode SAE4WMI faza"
    python3 -u evaluateModels.py "$dir" -o "$res_dir" --n-threads "4" --threshold "0.1" -m SAE4WMI faza 
done
