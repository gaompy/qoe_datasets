#!/bin/bash

# Primero se debe clonar el repo y acceder al directorio
#  git clone https://github.com/gaompy/qoe_datasets
#  cd qoe_datasets

CURR_PWD=$(pwd)

# Para cada directorio de conjunto de datos
for i in `seq 1 7`; do
	# RandomForest con 10 corridas de 10-CV
	for CV_SEED in `seq 1 10`; do
		mkdir -p $CURR_PWD/results/$i
		OUTPUT_FILE=$CURR_PWD/results/$i/RandomForest_$CV_SEED.txt
		java weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 -s $CV_SEED -t $CURR_PWD/$i/data_filled_r.arff > $OUTPUT_FILE
	done
done
