# Taller 05: Detección de Objetos con Deep Learning

Este taller guía el desarrollo de un modelo de deep learning para la localización de objetos (aviones) en imágenes, utilizando regresión de bounding boxes y PyTorch.

## Objetivos

- Procesar anotaciones de bounding boxes y normalizarlas.
- Construir un modelo basado en ResNet para predecir las coordenadas de los bounding boxes.
- Entrenar el modelo usando MSELoss.
- Evaluar el desempeño con la métrica IoU (Intersection over Union).
- Visualizar predicciones y comparar con las etiquetas reales.

## Estructura
Taller05-Deteccion/ ├── data/ │ ├── airplanes.zip │ ├── Airplanes.csv │ └── airplanes/ # Imágenes descomprimidas ├── taller_05_deteccion.ipynb └── README.md

## Requisitos

- Python 3.8+
- PyTorch
- torchvision
- pandas
- numpy
- matplotlib
- opencv-python

Instala dependencias con:

```sh
pip install torch torchvision pandas numpy matplotlib opencv-python
```

Flujo del Notebook
Preparación de datos:

Descomprime el dataset y lee las anotaciones de bounding boxes.
Convierte las coordenadas a formato COCO y normalízalas.
Visualización:

Grafica los bounding boxes sobre las imágenes para verificar las etiquetas.
Dataset y DataLoader:

Implementa una clase personalizada para cargar imágenes y etiquetas normalizadas.
Modelo:

Usa ResNet (por defecto ResNet34) como backbone y una cabeza de regresión para predecir las coordenadas.
Entrenamiento:

Entrena el modelo usando MSELoss y Adam.
Evaluación:

Calcula el IoU promedio sobre un subconjunto de imágenes y guarda el mejor modelo.
Visualización de predicciones:

Muestra imágenes con bounding boxes reales (verde) y predichos (rojo).
Notas
El modelo predice coordenadas normalizadas (0-1). Multiplica por el tamaño original para visualizar.
Si hay errores de lectura de imágenes, revisa las rutas y la descompresión del dataset.
Puedes cambiar la arquitectura base modificando la clase del modelo.
Autor:
[Tu Nombre]

Licencia:
MIT