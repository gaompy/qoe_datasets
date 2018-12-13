#!/usr/bin/python3

import pandas
import math
import csv

# Directorio de datos desde donde se leerán los csv
DIRDATA = './results/4/'
# Formato de nombres de archivos procesados por R y WEKA
EXT = '.csv'

fileNames = ['data_filled_r_', 'data_filled_weka_']

outputFileNames = ['r_algorithm_performance.csv', 'weka_algorithm_performance.csv']

# 18 algoritmos
algoritmos = [
    'A1DE',
    'IBk1',
    'NaiveBayes',
    'REPTree',
    'A2DE',
    'IBk5',
    'RandomForest',
    'SimpleLogistic',
    'BayesNet',
    'J48',
    'RandomTree',
    'SMO',
    'DecisionTable',
    'Logistic',
    'RBFClassifier',
    'IBk10',
    'MultilayerPerceptron',
    'RBFNetwork'
    ]

allColumns = [
    'areaUnderPRC',
    'areaUnderROC',
    'avgCost',
    'correct',
    'coverageOfTestCasesByPredictedRegions',
    'errorRate',
    'falseNegativeRate',
    'falsePositiveRate',
    'fMeasure',
    'incorrect',
    'kappa',
    'KBInformation',
    'KBMeanInformation',
    'meanAbsoluteError',
    'numFalseNegatives',
    'numFalsePositives',
    'numInstances',
    'numTrueNegatives',
    'numTruePositives',
    'pctCorrect',
    'pctIncorrect',
    'pctUnclassified',
    'precision',
    'recall',
    'relativeAbsoluteError',
    'rootMeanSquaredError',
    'totalCost',
    'trueNegativeRate',
    'truePositiveRate',
    'unclassified',
    'weightedAreaUnderPRC',
    'weightedAreaUnderROC',
    'weightedFalseNegativeRate',
    'weightedFalsePositiveRate',
    'weightedFMeasure',
    'weightedRecall',
    'weightedTrueNegativeRate',
    'weightedTruePositiveRate',
    'key_fold'
    ]

columnsToAnalyze = [
    'pctCorrect',
    'kappa',
    'weightedFMeasure',
    'weightedAreaUnderPRC',
    'weightedAreaUnderROC'
    ]

# Estadísticas para algoritmos en datos procesados por R
df_r_avg = pandas.DataFrame(index=algoritmos, columns=allColumns)
df_r_mad = pandas.DataFrame(index=algoritmos, columns=allColumns)

for algoritmo in algoritmos:
    fileName = DIRDATA + fileNames[0] + algoritmo + EXT
    df = pandas.read_csv(fileName)
    for metrica in allColumns:
        df_r_avg[metrica][algoritmo] = df[metrica].mean(skipna=True)
        df_r_mad[metrica][algoritmo] = df[metrica].mad(skipna=True)

# Estadísticas para algoritmos en datos procesados por WEKA
df_weka_avg = pandas.DataFrame(index=algoritmos, columns=allColumns)
df_weka_mad = pandas.DataFrame(index=algoritmos, columns=allColumns)

for algoritmo in algoritmos:
    fileName = DIRDATA + fileNames[1] + algoritmo + EXT
    df = pandas.read_csv(fileName)
    for metrica in allColumns:
        df_weka_avg[metrica][algoritmo] = df[metrica].mean(skipna=True)
        df_weka_mad[metrica][algoritmo] = df[metrica].mad(skipna=True)

# Sección de impresión de resultados
orden_rendimiento_algoritmos_r = {}
orden_rendimiento_algoritmos_weka = {}
for metrica in columnsToAnalyze:
    orden_rendimiento_algoritmos_r[metrica] = list(df_r_avg.sort_values(by=[metrica],ascending=False)[metrica].index.values)
    orden_rendimiento_algoritmos_weka[metrica] = list(df_weka_avg.sort_values(by=[metrica],ascending=False)[metrica].index.values)

# Resultados para R
with open(DIRDATA + outputFileNames[0], 'w') as csvfile:
    wr = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    wr.writerow(['metrica','algoritmo','media','desviacion_media'])
    for metrica in columnsToAnalyze:
        for algoritmo in orden_rendimiento_algoritmos_r[metrica]:
            avg = df_r_avg[metrica][algoritmo]
            mad = df_r_mad[metrica][algoritmo]
            wr.writerow([metrica, algoritmo, avg, mad])

# Resultados para WEKA
with open(DIRDATA + outputFileNames[1], 'w') as csvfile:
    wr = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    wr.writerow(['metrica','algoritmo','media','desviacion_media'])
    for metrica in columnsToAnalyze:
        for algoritmo in orden_rendimiento_algoritmos_weka[metrica]:
            avg = df_weka_avg[metrica][algoritmo]
            mad = df_weka_mad[metrica][algoritmo]
            wr.writerow([metrica, algoritmo, avg, mad])
