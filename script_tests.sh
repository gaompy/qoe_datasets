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
	# Aleatoriedad de los subconjuntos para CV
	for CV_SEED in `seq 1 10`; do
		# Conjunto de datos a analizar
		for DATASET in data data_filled_r data_filled_weka; do
			mkdir -p $CURR_PWD/results/$i
			OUTPUT_DIR=$CURR_PWD/results/$i

			# Redes Neuronales
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo MultilayerPerceptron
			OUTPUT_FILE=$OUTPUT_DIR/MultilayerPerceptron_$DATASET_$CV_SEED.txt
			java weka.classifiers.functions.MultilayerPerceptron -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H a -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo RBFNetwork
			OUTPUT_FILE=$OUTPUT_DIR/RBFNetwork_$DATASET_$CV_SEED.txt
			java weka.classifiers.functions.RBFNetwork -B 2 -S 1 -R 1.0E-8 -M -1 -W 0.1 -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo RBFClassifier
			OUTPUT_FILE=$OUTPUT_DIR/RBFClassifier_$DATASET_$CV_SEED.txt
			java weka.classifiers.functions.RBFClassifier -N 2 -R 0.01 -L 1.0E-6 -C 2 -P 1 -E 1 -S 1 -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE

			# Basados en modelos de regresión
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo SimpleLogistic
			OUTPUT_FILE=$OUTPUT_DIR/SimpleLogistic_$DATASET_$CV_SEED.txt
			java weka.classifiers.functions.SimpleLogistic -I 0 -M 500 -H 50 -W 0.0 -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo Logistic
			OUTPUT_FILE=$OUTPUT_DIR/Logistic_$DATASET_$CV_SEED.txt
			java weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4 -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE

			# Máquinas de soporte vectorial
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo SMO - Sequential Minimal Optimization
			OUTPUT_FILE=$OUTPUT_DIR/SMO_$DATASET_$CV_SEED.txt
			java weka.classifiers.functions.SMO -C 1.0 -L 0.001 -P 1.0E-12 -N 0 -V -1 -W 1 -K "weka.classifiers.functions.supportVector.PolyKernel -E 1.0 -C 250007" -calibrator "weka.classifiers.functions.Logistic -R 1.0E-8 -M -1 -num-decimal-places 4" -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			# Clasificadores Bayesianos
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo RandomForest
			OUTPUT_FILE=$OUTPUT_DIR/RandomForest_$DATASET_$CV_SEED.txt
			java weka.classifiers.trees.RandomForest -P 100 -I 100 -num-slots 1 -K 0 -M 1.0 -V 0.001 -S 1 -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo Naive Bayes
			OUTPUT_FILE=$OUTPUT_DIR/NaiveBayes_$DATASET_$CV_SEED.txt
			java weka.classifiers.bayes.NaiveBayes -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo A1DE
			OUTPUT_FILE=$OUTPUT_DIR/A1DE_$DATASET_$CV_SEED.txt
			java weka.classifiers.bayes.AveragedNDependenceEstimators.A1DE -F 1 -M 1.0 -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo A2DE
			OUTPUT_FILE=$OUTPUT_DIR/A2DE_$DATASET_$CV_SEED.txt
			java weka.classifiers.bayes.AveragedNDependenceEstimators.A2DE -F 1 -M 1.0 -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo BayesNet
			OUTPUT_FILE=$OUTPUT_DIR/BayesNet_$DATASET_$CV_SEED.txt1
			java weka.classifiers.bayes.BayesNet -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff -D -Q weka.classifiers.bayes.net.search.local.K2 -- -P 1 -S BAYES -E weka.classifiers.bayes.net.estimate.SimpleEstimator -- -A 0.5 > $OUTPUT_FILE

			# Árboles de decisión
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo RandomTree
			OUTPUT_FILE=$OUTPUT_DIR/RandomTree_$DATASET_$CV_SEED.txt
			java weka.classifiers.trees.RandomTree -K 0 -M 1.0 -V 0.001 -S 1 -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo REPTree
			OUTPUT_FILE=$OUTPUT_DIR/REPTree_$DATASET_$CV_SEED.txt
			java weka.classifiers.trees.REPTree -M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0 -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo J48
			OUTPUT_FILE=$OUTPUT_DIR/J48_$DATASET_$CV_SEED.txt
			java weka.classifiers.trees.J48 -C 0.25 -M 2 -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE

			# Clasificadores basados en instancias
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo IBk - k=1, k=5, k=10
			OUTPUT_FILE=$OUTPUT_DIR/IB1_$DATASET_$CV_SEED.txt
			java weka.classifiers.lazy.IBk -K 1 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			OUTPUT_FILE=$OUTPUT_DIR/IB5_$DATASET_$CV_SEED.txt
			java weka.classifiers.lazy.IBk -K 5 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			OUTPUT_FILE=$OUTPUT_DIR/IB10_$DATASET_$CV_SEED.txt
			java weka.classifiers.lazy.IBk -K 10 -W 0 -A "weka.core.neighboursearch.LinearNNSearch -A \"weka.core.EuclideanDistance -R first-last\"" -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE

			# Tablas de decisiones
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo Decision Table
			OUTPUT_FILE=$OUTPUT_DIR/DecisionTable_$DATASET_$CV_SEED.txt
			java weka.classifiers.rules.DecisionTable -X 1 -S "weka.attributeSelection.BestFirst -D 1 -N 5" -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
			echo Key Fold $CV_SEED : Dataset $DATASET : Algoritmo DTNB
			OUTPUT_FILE=$OUTPUT_DIR/DTNB_$DATASET_$CV_SEED.txt
			java weka.classifiers.rules.DTNB -X 1 -s $CV_SEED -t $CURR_PWD/$i/$DATASET.arff > $OUTPUT_FILE
		done
	done
done
