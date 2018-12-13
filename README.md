# Datasets extraídos del Proyecto Final de Grado de Análisis de QoE

#### Los archivos listados en el directorio son:

```sh
$ ls *
file.csv  fill_data_with_r.R  ParseStats.java  ParseStatsMultiThread.java results  README.md  script_test.sh

1:
README.md  data.arff  data.csv  data_filled_r.arff  data_filled_r.csv  data_filled_weka.arff

2:
README.md  data.arff  data.csv  data_filled_r.arff  data_filled_r.csv  data_filled_weka.arff

3:
README.md  data.arff  data.csv  data_filled_r.arff  data_filled_r.csv  data_filled_weka.arff

4:
README.md  data.arff  data.csv  data_filled_r.arff  data_filled_r.csv  data_filled_weka.arff

5:
README.md  data.arff  data.csv  data_filled_r.arff  data_filled_r.csv  data_filled_weka.arff

6:
README.md  data.arff  data.csv  data_filled_r.arff  data_filled_r.csv  data_filled_weka.arff

7:
README.md  data.arff  data.csv  data_filled_r.arff  data_filled_r.csv  data_filled_weka.arff
```
#### Descripción de cada archivo
* El archivo `README.md` de la raiz es este archivo.
* `file.csv` es el archivo original generado por la aplicación.
* `fill_data_with_r.R` es un script en R para llevar a cabo el relleno de datos en cada subconjunto de datos.

En cada directorio (1~7) existen 6 (seis) archivos: `README.md`, `data.arff`, `data.csv`, `data_filled_r.arff`, `data_filled_r.csv` y `data_filled_weka.arff`. Cada archivo se describe a continuación:

* `README.md` es el archivo que describe el contenido de `data.csv`.
* `data.csv` es un subconjunto de datos de `file.csv`.
* `data.arff` es el resultado de procesar el archivo `data.csv` a través de WEKA.
* `data_filled_weka.arff` es el resultado de procesar el filtro de relleno sobre el archivo `data.arff` a través de WEKA.
* `data_filled_r.csv` es el resultado de procesar el relleno de datos sobre el archivo `data.csv` con el paquete MICE de R.
* `data_filled_r.arff` es el resultado de procesar el archivo `data_filled_r.csv` a través de WEKA.

Los archivos ARFF son las fuentes de datos de entrada para los algoritmos de Machine Learning en WEKA.

Los archivos ARFF se generaron mediante el visor de archivos ARFF de WEKA (Tools -> ArffViewer), seleccionando la columna "mos" como atributo de clase y eliminando columnas que no son representativas para el estudio (columnas del CouchDB e identificadores de timestamp de la Aplicación Móvil: "X_id", "X_rev", "ts_from", "ts_to").

El archivo `data_filled_weka.arff` se generó corriendo el filtro `weka.filters.unsupervised.attribute.ReplaceMissingValues` sobre el archivo `data.arff`.

Los archivos `ParseStats.java` y `ParseStatsMultiThread.java` son versiones iterativas y multi-hilos para generar archivos `.csv` de estadísticas corriendo los algoritmos de clasificación.

Dentro del directorio `results` se almacenarán las estadísticas en CSV para ser analizadas y determinar qué grupo de algoritmos tuvo el mejor rendiemiento.

#### Configuración de las pruebas con WEKA

La configuración específica para cada clasificador se encuentra en el archivo script_test.sh.
Se eligió el mismo conjunto de algoritmos que el utilizado en el trabajo de referencia de modo a hacer una comparación más justa de la aplicabilidad del modelo en un ambiente con otro conjunto de métricas.

```sh
java weka.core.WekaPackageManager -install-package RBFNetwork
java weka.core.WekaPackageManager -install-package AnalogicalModeling
java weka.core.WekaPackageManager -install-package IBkLG
java weka.core.WekaPackageManager -install-package multiLayerPerceptrons
java weka.core.WekaPackageManager -install-package CFWNB
java weka.core.WekaPackageManager -install-package bestFirstTree
java weka.core.WekaPackageManager -install-package DTNB
```

Los comandos anteriores instalan los paquetes de Weka en el directorio `~/wekafiles/packages/`. Se deben agregar al CLASSPATH los archivos `.jar` correspondientes.

A continuación los algoritmos separados por grupos:
##### Redes Neuronales:
- MultilayerPerceptron
- RBFNetwork
- RBFClassifier
##### Basados en modelos de regresión
- SimpleLogistic
- Logistic
##### Máquinas de soporte vectorial
- Sequential Minial Optimization (SMO)
##### Clasificadores Bayesianos
- Naive Bayes
- A1DE
- A2DE
- BayesNet
- RandomForest
##### Árboles de decisión
- REPTree
- J4S
- RandomTree
##### Clasificadores basados en instancias
- IBk (k=1, k=5, k=10)
##### Tablas de decisiones
- Decision Table
- DTNB

Una vez que se cuente con los resultados, se sacarán estadísticas de cada grupo, de modo a determinar qué grupo tuvo mejor rendimiento para el conjunto de datos disponible.

Desde el archivo `order_by_mean_mendev.py` se ordenan los resultados de acuerdo a las columnas `kappa`, `weightedFMeasure`, `pctCorrect`, o las que el usuario defina en el parámetro correspondiente.
Debido a la distribución sesgada de las métricas clasificadas, se deberá considerar columnas como `weightedAreaUnderPRC` y `weightedAreaUnderROC`.
