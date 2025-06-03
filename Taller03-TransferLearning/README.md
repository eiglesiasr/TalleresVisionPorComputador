# Taller 03: Image Retrieval

Este taller sirve de guía para implementar un sistema de recuperación de imágenes basándose en una consulta de texto usando el modelo CLIP de OpenAI y el dataset Caltech 256.

## Objetivos

1. Implementar un sistema de recuperación de texto-imagen (image retrieval).
2. Utilizar imágenes de Caltech 256 (seleccionando un subconjunto aleatorio, por ejemplo, 30% de las imágenes).
3. Generar incrustaciones (embeddings) de las imágenes usando CLIP.
4. Recuperar las imágenes más similares a un query textual mediante similitud coseno.

## Estructura del Proyecto

```
Taller03-TransferLearning/
├── data/
│   └── caltech/           # Directorio donde se descargará y extraerá el dataset Caltech 256
├── taller_03_clasificacion.ipynb   # Notebook principal del taller
└── README.md              # Este archivo
```

## Requisitos e Instalación

- Python 3.8+
- [PyTorch](https://pytorch.org/) (compatible con CUDA / MPS según tu hardware)
- [torchvision](https://pytorch.org/vision/stable/)
- [CLIP](https://github.com/openai/CLIP)  
- [matplotlib](https://matplotlib.org/)
- [tqdm](https://github.com/tqdm/tqdm)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)

Para instalar las dependencias (además de PyTorch y torchvision según tu sistema), ejecuta:

```bash
pip install matplotlib tqdm pandas numpy
pip install git+https://github.com/openai/CLIP.gitx
```

## Uso

1. **Descarga y extracción del dataset Caltech 256**  
   En el notebook se incluyen comandos para:
   - Crear el directorio `data/caltech`.
   - Descargar el archivo TAR desde la URL:  
     `https://data.caltech.edu/records/nyy15-4j048/files/256_ObjectCategories.tar?download=1`
   - Extraer el dataset dentro de `data/caltech`.

2. **Carga del dataset y selección de subconjunto**  
   Se utiliza `ImageFolder` de torchvision para cargar la imagen y se selecciona aleatoriamente un porcentaje (por ejemplo, 30%) de las imágenes.

3. **Precomputación de embeddings**  
   Se utiliza el modelo CLIP en modo evaluación para obtener las representaciones (embeddings) de cada imagen. Las incrustaciones se normalizan y concatenan para su uso en recuperación.

4. **Recuperación de imágenes**  
   Con la función `retrieve_images` se puede pasar una consulta de texto y recuperar los índices de las imágenes más similares (basado en la similitud coseno). Luego, se muestran las imágenes recuperadas junto con su puntuación.

5. **Ejecutar el Notebook**  
   Abre y ejecuta `taller_03_clasificacion.ipynb` en Jupyter o VS Code.  
   Asegúrate de configurar correctamente el dispositivo (CUDA, MPS o CPU) según la disponibilidad antes de comenzar la extracción de embeddings.

## Notas Adicionales

- El modelo CLIP se carga usando el identificador `"ViT-B/32"`.
- La normalización para visualizar las imágenes sigue el preprocesado de CLIP.
- Se recomienda ejecutar el proceso en un entorno con GPU o MPS para acelerar la extracción de features.
- La función `retrieve_images` calcula la similitud coseno entre las incrustaciones del texto y las imágenes para determinar las imágenes más relevantes en función del query.

## Ejemplo de Uso

Dentro del Notebook, se incluye un ejemplo de cómo recuperar imágenes:

```python
query = "bat"
top_indices, scores = retrieve_images(query, top_k=5)

# Visualizar las imágenes recuperadas
import matplotlib.pyplot as plt
plt.figure(figsize=(15,5))
for idx, (img_idx, score) in enumerate(zip(top_indices, scores)):
    img_tensor, label = all_images[img_idx]
    # Convertir el tensor a imagen y revertir la normalización de CLIP
    img = img_tensor.cpu().permute(1,2,0).numpy()
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.subplot(1, 5, idx+1)
    plt.imshow(img)
    plt.title(f"{label}\nScore: {score:.2f}")
    plt.axis("off")
plt.show()
```

Este ejemplo muestra cómo, dado un query (en este caso, "bat"), se recuperan y visualizan las imágenes más relevantes.

---
