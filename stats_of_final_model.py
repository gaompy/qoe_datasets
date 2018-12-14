#!/usr/bin/python3

import pandas
import math
import csv

# Directorio de datos desde donde se leerán los csv
DIRDATA = './results/4/'
# Formato de nombres de archivos procesados por R y WEKA
EXT = '.csv'

fileNames = ['final_model_stats_data_filled_r', 'final_model_stats_data_filled_weka']

outputFileNames = ['final_model_performance_r.csv', 'final_model_performance_weka.csv']

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
df_r_avg = pandas.DataFrame(index=['final_model'], columns=allColumns)
df_r_mad = pandas.DataFrame(index=['final_model'], columns=allColumns)

fileName = DIRDATA + fileNames[0] + EXT
df = pandas.read_csv(fileName)
for metrica in allColumns:
    df_r_avg[metrica]['final_model'] = df[metrica].mean(skipna=True)
    df_r_mad[metrica]['final_model'] = df[metrica].mad(skipna=True)

# Estadísticas para algoritmos en datos procesados por WEKA
df_weka_avg = pandas.DataFrame(index=['final_model'],columns=allColumns)
df_weka_mad = pandas.DataFrame(index=['final_model'],columns=allColumns)

fileName = DIRDATA + fileNames[1] + EXT
df = pandas.read_csv(fileName)
for metrica in allColumns:
    df_weka_avg[metrica]['final_model'] = df[metrica].mean(skipna=True)
    df_weka_mad[metrica]['final_model'] = df[metrica].mad(skipna=True)

# Resultados para R
with open(DIRDATA + outputFileNames[0], 'w') as csvfile:
    wr = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    wr.writerow(['metrica','media','desviacion_media'])
    for metrica in columnsToAnalyze:
        avg = df_r_avg[metrica]['final_model']
        mad = df_r_mad[metrica]['final_model']
        wr.writerow([metrica, avg, mad])
            
# Resultados para WEKA
with open(DIRDATA + outputFileNames[1], 'w') as csvfile:
    wr = csv.writer(csvfile, delimiter=',',
                            quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
    wr.writerow(['metrica','media','desviacion_media'])
    for metrica in columnsToAnalyze:
        avg = df_weka_avg[metrica]['final_model']
        mad = df_weka_mad[metrica]['final_model']
        wr.writerow([metrica, avg, mad])
