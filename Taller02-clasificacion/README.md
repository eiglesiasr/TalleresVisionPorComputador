# Taller 02: Clasificación Multiclase de Imágenes

Este proyecto está diseñado para realizar las siguientes tareas:

1. **Exploración del conjunto de datos**  
   Se inspecciona la distribución de imágenes y clases a partir del dataset LabelMe-12-50k.

2. **Extracción de características**  
   Se utiliza el modelo preentrenado ResNet50 para extraer características de las imágenes.

3. **Entrenamiento de la cabeza de clasificación**  
   Con AutoGluon se entrena un modelo MLP (solo sobre características numéricas) para clasificar las imágenes en múltiples clases.

4. **Evaluación del desempeño del modelo**  
   Se evalúa la precisión (accuracy) de las predicciones obtenidas.

---

## Estructura de Directorios

El dataset debe tener la siguiente organización:

```
data/labelme/
    train/
        class_1/
        class_2/
        ...
    test/
        class_1/
        class_2/
        ...
```

**Nota:** Durante la extracción, el archivo `LabelMe-12-50k.tar.gz` se descomprime en el directorio `data` para generar las carpetas `train` y `test`. Asegúrate de colocar el archivo del dataset en el directorio de trabajo.

---

## Requisitos

- Python 3.10 (o superior)
- PyTorch y torchvision
- matplotlib
- pandas y numpy
- tqdm
- AutoGluon Multimodal

Instala las dependencias (si no las tienes) con:

```
pip install torch torchvision matplotlib pandas numpy tqdm autogluon
```

---

## Uso del Notebook

1. **Setup & Dependencias:**  
   Se importan las librerías necesarias y se selecciona el dispositivo (GPU/MPS/CPU).

2. **Extracción del Dataset:**  
   El Notebook verifica si existen las carpetas `train` y `test` en `data`. Si no existen, se extrae el contenido del archivo `LabelMe-12-50k.tar.gz` en el directorio `data`.

3. **Exploración del Conjunto de Datos:**  
   Se crea un DataFrame con la lista de imágenes, agrupadas por clase, para visualizar la distribución y algunas estadísticas.

4. **Carga de Anotaciones:**  
   La función `load_annotations()` busca y parsea los archivos de anotaciones (por ejemplo, `annotation.txt`) para cada split.  
   Las imágenes se buscan dentro de las subcarpetas (por clase) en cada uno de los directorios `train` y `test`.

5. **Extracción de Características:**  
   Se define un dataset personalizado (`ImageFeatureDataset`) para aplicar una transformación a cada imagen y extraer características con ResNet50 (sin la última capa).  
   Las características se almacenan en DataFrames y se guardan en archivos TSV.

6. **Entrenamiento con AutoGluon:**  
   Se entrena una cabeza de clasificación (MLP) usando AutoGluon Multimodal, utilizando las características extraídas previamente.

7. **Evaluación y Predicciones:**  
   Se evalúa el modelo (por defecto utilizando accuracy) y se obtienen algunas predicciones de ejemplo.

---

## Resultados

Por ejemplo, se obtuvo un accuracy de aproximadamente 0.8554, lo que indica que el 85.54% de las imágenes fueron clasificadas correctamente en el conjunto de test.

---

## Notas Adicionales

- **Multiprocessing en el DataLoader:**  
  Si se presentan problemas con la clase `ImageFeatureDataset` al utilizar `num_workers > 0`, se recomienda usar `num_workers=0` para evitar errores de serialización en el entorno Notebook.

- **Búsqueda de Imágenes:**  
  La función `load_annotations()` busca las imágenes en las subcarpetas de cada split. Asegúrate de que la estructura del dataset coincida con lo descrito para que las rutas se construyan correctamente.

- **Uso de AutoGluon:**  
  AutoGluon utiliza accuracy como métrica predeterminada para clasificación. Si deseas cambiar la métrica, revisa la documentación de AutoGluon Multimodal.
