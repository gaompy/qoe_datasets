# Datasets extraídos del Proyecto Final de Grado de Análisis de QoE

Los archivos listados en el directorio son:

```sh
$ ls
file.csv  fill_data_with_r.R  README.md

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
README.md es este archivo.

file.csv es el archivo original generado por la aplicación.

fill_data_with_r.R es un script en R para llevar a cabo el relleno de datos en cada subconjunto de datos.

En cada directorio (1:7) existen 6 (seis) archivos: README.md, data.arff, data.csv, data_filled_r.arff, data_filled_r.csv y data_filled_weka.arff.

README.md es el archivo que describe el contenido de data.csv.
data.csv es un subconjunto de datos de file.csv.
data.arff es el resultado de procesar el archivo data.csv a través de WEKA.
data_filled_weka.arff es el resultado de procesar el filtro de relleno sobre el archivo data.arff a través de WEKA.
data_filled_r.csv es el resultado de procesar el relleno de datos sobre el archivo data.csv con el paquete MICE de R.
data_filled_r.arff es el resultado de procesar el archivo data_filled_r.csv a través de WEKA.

Los archivos ARFF son las fuentes de datos de entrada para los algoritmos de Machine Learning en WEKA.

Los archivos ARFF se generaron mediante el visor de archivos ARFF de WEKA (Tools -> ArffViewer), seleccionando la columna "mos" como atributo de clase y eliminando columnas que no son representativas para el estudio (columnas del CouchDB y identificadores de timestamp de la Aplicación Móvil: "X_id", "X_rev", "ts_from", "ts_to").

El archivo data_filled_weka.arff se generó corriendo el filtro weka.filters.unsupervised.attribute.ReplaceMissingValues sobre el archivo data.arff.
