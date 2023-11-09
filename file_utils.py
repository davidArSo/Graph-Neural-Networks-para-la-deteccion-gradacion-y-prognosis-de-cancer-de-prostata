"""
file_utils.py - Utilidades para manejar archivos del proyecto.

Este fichero contiene diversas funciones relacionadas con la gestión de archivos. Incluye 
métodos para descargar la base de datos del proyecto, verificar la existencia de archivos  y carpetas y descomprimir archivos.

Contenido:
	- download_file(url, destination_path, filename)
	- file_exists(file_path)
	- unzip_file(zip_path, destination_path)
	- folder_exists(folder_path)

Uso:
	from file_utils import download_file, file_exists, unzip_file, folder_exists
"""

import requests
from tqdm import tqdm
import shutil
import os
import zipfile

def download_file(url, destination_path, filename):
  """
    Descarga un archivo desde una URL y lo guarda con un nombre específico.

    Parámetros:
      url (str): URL del archivo que se va a descargar.
      destination_path (str): Ruta de Google Drive donde se almacenará el archivo.
      filename (str): Nombre del archivo que se utilizará para guardar.

    Retorna:
      bool: Verdadero si la descarga fue exitosa, Falso en caso contrario.
    """

  try:
    print('Descargando ...')

    # Realizamos una petición GET para descargar el archivo
    response = requests.get(url, stream=True)

    # Obtenemos el tamaño total del archivo en bytes y creamos una barra de progreso
    total_size = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=filename, leave=True)

    # Comprobamos si la petición se ha realizado correctamente
    if response.status_code == 200:
      # Guardamos el archivo
      with open(filename, 'wb') as local_file:
        for data in response.iter_content(chunk_size=1024):
          local_file.write(data)
          # Actualizamos el progreso de la barra
          progress_bar.update(len(data))
          progress_bar.close()

          # Movemos el archivo descargado 
          shutil.move(filename, destination_path)

          #print(f'Fichero {filename} descargado con éxito')
          return True
    else:
      print('Error en la descarga del fichero')
      return False
  except Exception as e:
    print(f'Error: {e}')
    return False

def file_exists(file_path):
  """
    Comprueba si un archivo existe en la ruta especificada.

    Parámetros:
      file_path (str): Ruta del archivo a comprobar.

    Retorna:
      bool: Verdadero si el archivo existe, Falso en caso contrario.
    """
  
  return os.path.exists(file_path)        

def unzip_file(zip_path, destination_path):
  """
  Descomprime un archivo ZIP.

  Parámetros:
    zip_path (str): Ruta del archivo ZIP que se va a descomprimir.
    destination_path (str): Ruta de la carpeta donde se guardarán los archivos descomprimidos.

  Retorna:
    bool: Verdadero si se ha descomprimido correctamente, Falso en caso contrario.
  """

  try:
    print('Descomprimiendo ...')

    # Descomprimimos el archivo ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
      zip_ref.extractall(destination_path)

    print('Fichero descomprimido con éxito')
    return True
  except Exception as e:
    print(f'Error: {e}')
    return False

def folder_exists(folder_path):
  """
  Comprueba si una carpeta existe en la ruta especificada.

  Parámetros:
    folder_path (str): Ruta de la carpeta a comprobar.

  Retorna:
    bool: Verdadero si la carpeta existe, Falso en caso contrario.
  """
  
  return os.path.exists(folder_path)
