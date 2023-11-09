"""
image_utils.py - Utilidades para el manejo y visualización de imágenes histológicas.

Este fichero contiene diversas funciones relacionadas con el procesamiento y visualización
de histologías. Incluye métodos para verficar y modificar las propiedades de una imagen, 
parsear información, construir una WSI a partir de parches, visualizar WSI, generar nuevos parches, ...

Contenido:
  - check_image_properties(image_dir)
  - parse_image_filename(image_filename)
  - display_patches(slide_id, image_dir, patches_per_row=6)
  - construct_wsi(slide_id, image_dir)
  - show_resized_image(image, max_width=1920, max_height=1080)
  - extract_tissue_mask(image, threshold=220)
  - extract_patches(slide_id, image, tissue_mask, wsi_patches_path, patch_size=256, tissue_threshold=0.2)

Uso:
  from image_utils import check_image_properties, parse_image_filename, display_patches, construct_wsi, show_resized_image, extract_tissue_mask, extract_patches
"""

import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

def check_image_properties(image_dir):
  """
  Verifica las propiedades de las imágenes del directorio especificado.

  Parámetros:
    image_dir (str): Ruta del directorio que contiene las imágenes a verificar.

  Retorna:
    None
  """

  print(f"Verificando las propiedades de las imágenes del directorio: {image_dir}")

  image_files = os.listdir(image_dir)

  # Arrays para almacenar los tamaños y formatos de las imágenes
  sizes = []
  formats = []

  for image in image_files:
    # Obtenemos la ruta completa
    image_full_path = os.path.join(image_dir, image)

    # Abrimos la imagen
    with Image.open(image_full_path) as img:
      # Obtenemos el tamaño y el formato
      img_size = img.size
      img_format = img.format
    
    # Añadimos el tamaño y formato a los arrays
    sizes.append(img_size)
    formats.append(img_format)

  # Convertimos los arrays en NumPy arrays
  sizes = np.array(sizes)
  formats = np.array(formats)

  # Comprobamos si hay diferencias en los tamaños y formatos
  different_sizes = (sizes != sizes[0]).any(axis=0)
  different_formats = (formats != formats[0]).any()

  if different_sizes.any():
    print("Al menos una imagen tiene un tamaño diferente")
  else:
    print(f"Todas las imágenes tienen el tamaño {sizes[0]}")

  if different_formats:
    print("Al menos una imagen tiene un formato diferente")
  else:
    print(f"Todas las imágenes tienen el formato {formats[0]}")

def parse_image_filename(image_filename):
  """
  Descompone el nombre de una imagen en atributos específicos.

  Parámetros:
    image_filename (str): Nombre de la imagen a parsear.

  Retorna:
    dict: Un diccionario con los atributos descompuestos.
  """

  # Dividimos el nombre de la imagen en partes usando "_" como separador
  parts = image_filename.split('_')

  # Obtenemos el Slide ID (Parte 0)
  slide_id = parts[0]

  # Obtenemos el nombre del bloque/región (Parte 3)
  block_region = parts[3]

  # Obtenemos las coordenadas x e y (Partes 4 y 5)
  x_coordinate_block = parts[4]
  y_coordinate_block = parts[5]

  # Obtenemos el valor de xini y yini (Partes 4 y 5)
  xini = parts[7]
  yini = parts[9].replace('.jpg', '')

  return {
      'slide_id': slide_id,
      'block_region': int(block_region),
      'x_coordinate_block': int(x_coordinate_block),
      'y_coordinate_block': int(y_coordinate_block),
      'xini': int(xini),
      'yini': int(yini)
  }

def display_patches(slide_id, image_dir, patches_per_row=6):
  """
  Visualización de los parches correspondientes al slide_id.

  Parámetros:
    slide_id (str): Slide ID para el cual se desea mostrar los parches.
    image_dir (str): Ruta del directorio que contiene los parches.
    patches_per_row (int): Número de parches a mostrar por fila.

  Retorna:
      None
  """

  print(f'Visualización de los parches con slide_id: {slide_id}')

  # Buscamos todos los parches con el slide ID dado
  slide_patches = [file for file in os.listdir(image_dir) if file.startswith(slide_id)]

  # Construimos la figura
  num_rows = (len(slide_patches) - 1) // patches_per_row + 1
  fig, axes = plt.subplots(num_rows, patches_per_row, figsize=(15, 2 * num_rows))
  for i, patch_filename in enumerate(slide_patches):
    with Image.open(os.path.join(image_dir, patch_filename)) as patch_image:
      ax = axes[i // patches_per_row, i % patches_per_row]
      ax.imshow(patch_image)
      ax.axis('off')

      # Obtenemos el nombre de la imagen sin el slide_id ni la extensión .jpg
      name_parts = patch_filename.split('_')
      name_without_slide_id = ' '.join(name_parts[1:])
      name_without_extension = name_without_slide_id.split('.')[0]
      ax.set_title(f'{name_without_extension}', fontsize=6)
    
  # Ajustamos el espaciado entre subplots
  plt.subplots_adjust(wspace=0.1, hspace=0.5)

  # Eliminamos los ejes de las figuras sobrantes
  for i in range(len(slide_patches), num_rows * patches_per_row):
    fig.delaxes(axes.flatten()[i])

  # Mostramos la figura
  plt.show()

def construct_wsi(slide_id, image_dir):
  """
  Construye la imagen WSI completa a partir de los parches con el slide ID dado.

  Parámetros:
      slide_id (str): Slide ID para el cual se desea reconstruir la imagen completa.
      image_dir (str): Ruta del directorio que contiene los parches.

  Retorna:
      Image: Imagen PIL que representa la imagen WSI reconstruida.
  """

  slide_patches = [file for file in os.listdir(image_dir) if file.startswith(slide_id)]

  if not slide_patches:
    print("No se encontraron parches para el Slide ID proporcionado.")
    return None

  slide_patches.sort(key=lambda x: (int(parse_image_filename(x)['xini']),
                                    int(parse_image_filename(x)['yini'])))
  
  # Tamaño del parche y superposición
  patch_size = 512
  overlap = patch_size // 2  # 50% de superposición

  # Calculamos el tamaño de la imagen WSI
  max_xini = max(int(parse_image_filename(patch)['xini']) for patch in slide_patches)
  max_yini = max(int(parse_image_filename(patch)['yini']) for patch in slide_patches)

  total_width = max_xini + patch_size
  total_height = max_yini + patch_size

  # Establecemos el fondo de la imagen
  wsi_image = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255

  min_x, min_y, max_x, max_y = float('inf'), float('inf'), 0, 0

  for patch_filename in slide_patches:
    patch_info = parse_image_filename(patch_filename)
    
    # Ajustamos las posiciones considerando la superposición
    xini = int(patch_info['xini']) // 2
    yini = int(patch_info['yini']) // 2
    x_position = xini - (xini // patch_size) * overlap
    y_position = yini - (yini // patch_size) * overlap

    with Image.open(os.path.join(image_dir, patch_filename)) as patch_image:
      patch_array = np.array(patch_image)

    wsi_image[y_position:y_position+patch_size, x_position:x_position+patch_size] = patch_array

    # Actualizamos los límites
    min_x = min(min_x, x_position)
    min_y = min(min_y, y_position)
    max_x = max(max_x, x_position + patch_size)
    max_y = max(max_y, y_position + patch_size)

  # Recortamos la imagen
  cropped_image = wsi_image[min_y:max_y, min_x:max_x]
  wsi_image_PIL = Image.fromarray(cropped_image)

  return wsi_image_PIL
  
def show_resized_image(image, max_width=1920, max_height=1080):
  """
  Muestra una imagen redimensionada y ajustada a un tamaño máximo.

  Parámetros:
    image: Imagen PIL.
    max_width (int): Ancho máximo de la imagen. 
    max_height (int): Altura máxima de la imagen. 
  
  Retorna:
    None
  """
  
  try:
    # Redimensionamos la imagen para ajustarla al tamaño máximo
    image.thumbnail((max_width, max_height), Image.LANCZOS)

    # Visualizamos la imagen
    display(image)
  except Exception as e:
    print(f'Error al mostrar la imagen: {e}')

def extract_tissue_mask(image, threshold=220):
  """
  Extrae la máscara de tejido de una imagen histológica.

  Parámetros:
    image (numpy.array): Imagen histológica en formato RGB.
    threshold (int): Valor umbral

  Retorna:
    tissue_mask (numpy.array): Máscara del tejido.
  """

  # Convertir a escala de grises
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

  # Aplicar umbral
  _, tissue_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

  return tissue_mask
  
def extract_patches(slide_id, image, tissue_mask, wsi_patches_path, patch_size=256, tissue_threshold=0.2):
  """
  Extrae y guarda parches de tamaño fijo de una imagen histológica.

  Parámetros:
    slide_id (str): Identificador de la slide
    image (numpy.array): Imagen histológica en formato RGB.
    tissue_mask (numpy.array): Máscara del tejido.
    wsi_patches_path (str): Ruta para almacenar los parches
    patch_size (int): Tamaño de los parches.
    tissue_threshold (float): Umbral mínimo de tejido
  
  Retorna:
   None
  """

  # Obtenemos las dimensiones de la imagen
  h, w, _ = image.shape

  # Recorremos la imagen en pasos de 'patch_size'
  for i in range(0, h, patch_size):
    for j in range(0, w, patch_size):
      # Extraemos parche y su correspondiente máscara
      patch = image[i:i+patch_size, j:j+patch_size]
      patch_mask = tissue_mask[i:i+patch_size, j:j+patch_size]

      # Comprobamos si el parche tiene el tamaño correcto y una cantidad significativa de tejido
      if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
        if np.sum(patch_mask)/255 >= (patch_size * patch_size * tissue_threshold):
          y, x = i // patch_size, j // patch_size
          patch_filename = f"{slide_id}_Block_Region_0_{y}_{x}_xini_{j}_yini_{i}.jpg"
          patch_save_path = os.path.join(wsi_patches_path, patch_filename)
          cv2.imwrite(patch_save_path, patch)