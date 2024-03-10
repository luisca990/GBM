# GBM

## Descripción

Este repositorio contiene la solución al problema utilizando Python 3.10.0 La estructura de carpetas se organiza de la siguiente manera:

- **Ejercicio_4/**: Carpeta principal.
  - **Datos/**: Carpeta que contiene los archivos de los datos usados para entrenar y probar el modelo.
      - **train/**: Carpeta que contiene los archivos de los datos de entrenamiento separados en las caracteríticas X y la variable objetivo y.
      - **test/**: Carpeta que contiene los archivos de los datos de prueba separados en las caracteríticas  y la variable objetivo y.
      - **data_customer_classification.xlsx**: Archivo que contiene el dataset.
  - **Modelo/**: Carpeta que contiene los archivos generados al exportar el modelo de clasificación.
    - **modelo.json**: Archivo que contiene la estructura del modelo.
    - **pesos_modelo.h5**: Archivo que contiene los pesos del modelo aprendidos  a partir del dataset en la etapa de entrenamiento.
  - **requirements.txt**: Archivo que contiene las librerias y sus versiones usadas para el desarrollo del modelo.
  - **Problema.txt**: Archivo de texto que contiene la descripción del problema.
  - **SolucionE4.ipynb**: Cuaderno de jupyter en el cual está todo el proceso seguido para desarrollar el modelo de clasificación.
  - **SolucionE4.py**: Script de python generado a partir del cuadero de jupyter.

## Instrucciones para ejecutar el ejercicio

Antes de proceder a ejecutar el cuaderno de jupyter es necesario instalar las librerias de python para evitar conflictos con el siguiente comando:
```shell
pip install -r requirements.txt
```
El paso seguido es ejecutar celda a celda del cuaderno de jupyter para replicar el proceso de desarrollo.

