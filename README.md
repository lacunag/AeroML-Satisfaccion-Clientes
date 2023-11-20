# Análisis de Satisfacción AeroML

Este proyecto realiza un análisis de satisfacción de clientes utilizando datos de la industria aeroespacial. Se emplean técnicas de preprocesamiento de datos, construcción de un modelo de clasificación y visualización de resultados.

## Requisitos

- Python 3.x
- Bibliotecas: pandas, numpy, xgboost, lightgbm, catboost, scikit-learn, tensorflow, matplotlib, seaborn

## Instalación

Clona el repositorio o descarga el código fuente.

## Uso

1. Coloca el archivo de datos `satisfaccion_AeroML.xlsx` en el directorio raíz.
2. Ejecuta el script `satisfaccion_aeroml.py`.
   
## Funcionalidades

### Preprocesamiento de Datos
Carga el conjunto de datos y realiza una limpieza básica eliminando filas con valores nulos, codificando variables categóricas y normalizando/escalando características.

### Modelado y Evaluación
Entrena un modelo de clasificación utilizando Random Forest, evalúa su rendimiento con métricas como precisión, recall y F1-score, muestra una matriz de confusión y visualiza la importancia de las características.

### Visualización de Datos
Utiliza PCA para visualizar la distribución de los datos en un espacio de menor dimensión en relación con la variable objetivo de satisfacción.
