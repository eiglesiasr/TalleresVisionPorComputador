# Taller-01: Clasificación de Enfermedades en Hojas de Mango

Este proyecto implementa un sistema de clasificación para predecir la enfermedad de las hojas de mango utilizando tres enfoques:
- **Extracción de características HOG** para representar las imágenes.
- **Clasificador SVM y MLP** para clasificación tradicional.
- **Transfer Learning con ResNet50** para aprovechar modelos preentrenados.

## Estructura del Notebook

El notebook principal se encuentra en [Taller01-clasificacion/notebooks/taller_01.ipynb](Taller01-clasificacion/notebooks/taller_01.ipynb) y está organizado en los siguientes apartados:

1. **Carga y Preprocesamiento de Datos**  
   Se carga el dataset desde el directorio `mango_leaf_disease`, se extraen las imágenes (usando OpenCV) y sus etiquetas. Además, se valida el balance de clases.

2. **Extracción de Características HOG**  
   Se define la función `extract_hog_features(image)` que extrae el vector de características de cada imagen.

3. **Clasificación con SVM**  
   Se entrena un clasificador SVM y se evalúa el modelo mediante métricas de Accuracy y F1 Score.

4. **Clasificación con MLP (Red Neuronal)**  
   Se crea y entrena un modelo de Red Neuronal Multicapa (MLP), evaluando su rendimiento.

5. **Transfer Learning con ResNet50**  
   Se prepara la imagen con la función `prepare_image(img)`, se utiliza el modelo ResNet50 (sin la capa superior) y se entrena una capa densa para la clasificación. Se evalúa con métricas similares al resto.

6. **Visualización de Resultados**  
   Se genera una gráfica comparativa que muestra el Accuracy y F1 Score de los modelos SVM, MLP y ResNet50.

## Requisitos

- **Python 3.8+**
- **OpenCV** (`opencv-python`)
- **scikit-learn**
- **TensorFlow** (incluye Keras)
- **matplotlib**
- **numpy**

## Instrucciones para Ejecutar el Notebook

1. Instala las dependencias necesarias:
    ```bash
    pip install opencv-python scikit-learn tensorflow matplotlib numpy
    ```
2. Asegúrate de tener el dataset `mango_leaf_disease` en el directorio raíz, con las imágenes organizadas por etiquetas en subcarpetas.
3. Abre el notebook [taller_01.ipynb](Taller01-clasificacion/notebooks/taller_01.ipynb) en Jupyter Notebook o Visual Studio Code.
4. Ejecuta las celdas del notebook en orden para reproducir el flujo de preprocesamiento, entrenamiento y evaluación.

## Descripción de Funciones Clave

- **extract_hog_features(image):**  
  Extrae y devuelve las características HOG de una imagen redimensionada al tamaño predeterminado del descriptor.

- **prepare_image(img):**  
  Prepara la imagen para el modelo ResNet50, convirtiéndola a 3 canales si es necesario, redimensionándola a 224x224 píxeles y aplicando el preprocesamiento requerido.

## Resultados

El notebook imprime y muestra:
- Número total de imágenes y etiquetas cargadas.
- Distribución de clases (balanceo).
- Metrics de evaluación (Accuracy y F1 Score) para SVM, MLP y ResNet50.
- Gráficas comparativas para analizar el desempeño de cada modelo.

## Licencia

Este proyecto se distribuye bajo la licencia MIT.
