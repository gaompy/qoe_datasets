#!/bin/bash

# Primero se debe clonar el repo y acceder al directorio
#  git clone https://github.com/gaompy/qoe_datasets
#  cd qoe_datasets
# Segundo, corroborrar que la variable de entorno CLASSPATH contenga a weka.jar. Ejemplo:
#  export CLASSPATH=~/weka-3-8-3/weka.jar:$CLASSPATH
# Normalmente se lleva a cabo CV (cross-validations) cuando no se especifica un archivo de Training (opción -T)
# y se utiliza el parámetro -s para indicar una semilla para agrupar de forma aleatoria a los conjuntos de Training y Validation.

CURR_PWD=$(pwd)

# Para cada directorio de conjunto de datos hacer 10 corridas de 10-CV
for i in `seq 1 7`; do
	for CV_SEED in `seq 1 10`; do
		for DATASET in data data_filled_r data_filled_weka; do
			mkdir -p $CURR_PWD/results/$i
			OUTPUT_DIR=$CURR_PWD/results/$i

			# Redes Neuronales
			## MultilayerPerceptron
			## RBFNetwork
			## RBFClassifier

			# Basados en modelos de regresión
			## SimpleLogistic
			## Logistic

			# Máquinas de soporte vectorial
			## Sequential Minial Optimization (SMO)

			# Clasificadores Bayesianos
			## RandomForest
			OUTPUT_FILE=$OUTPUT_DIR/RandomForest_$DATASET_$CV_SEED.txt
			java weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			## Naive Bayes
			OUTPUT_FILE=$OUTPUT_DIR/NaiveBayes_$DATASET_$CV_SEED.txt
			java weka.classifiers.bayes.NaiveBayes -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			## A1DE
			## A2DE
			## BayesNet

			# Árboles de decisión
			## RandomTree
			OUTPUT_FILE=$OUTPUT_DIR/RandomTree_$DATASET_$CV_SEED.txt
			java weka.classifiers.trees.RandomTree -K 0 -M 1.0 -V 0.001 -S 1 -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			## REPTree
			## J4S

			# Clasificadores basados en instancias
			## IBk (k=1, k=5, k=10)

			# Tablas de decisiones
			## Decision Table
			## DTNB
		done
	done
done
