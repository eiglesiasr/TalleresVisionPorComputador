# Taller 4: Implementación de un Modelo de Red Neuronal Convolucional

En este taller se implementa un modelo de red neuronal convolucional (CNN) para una tarea de pretexto, específicamente la rotación de imágenes. El objetivo es entrenar un modelo para predecir la rotación de imágenes y luego utilizar las características extraídas para una clasificación downstream.

## Archivos del Proyecto

- **Taller_4_pretexto.ipynb**: Este archivo contiene el código y las explicaciones para el Taller 4. Incluye la carga de imágenes, la implementación de un modelo CNN para la tarea de pretexto (rotación de imágenes), y la evaluación del modelo utilizando un clasificador de regresión logística.

- **README.md**: Este archivo contiene la documentación del proyecto, incluyendo una descripción del taller, instrucciones sobre cómo ejecutar el cuaderno, y detalles sobre las dependencias necesarias.

## Instrucciones para Ejecutar el Cuaderno

1. **Instalar Dependencias**: 

   - TensorFlow
   - NumPy
   - Matplotlib
   - scikit-learn


## Descripción de la Tarea de Pretexto

La tarea de pretexto consiste en rotar imágenes en ángulos de 0, 90, 180 y 270 grados. El modelo se entrena para predecir el ángulo de rotación de las imágenes. Después de entrenar el modelo, se extraen las características de las imágenes y se utilizan para entrenar un clasificador de regresión logística que evalúa el rendimiento del modelo en una tarea de clasificación downstream.

## Resultados

El rendimiento del modelo se evalúa utilizando métricas como la precisión y la matriz de confusión. Los resultados se visualizan a través de gráficos que muestran la precisión y la pérdida durante el entrenamiento.

