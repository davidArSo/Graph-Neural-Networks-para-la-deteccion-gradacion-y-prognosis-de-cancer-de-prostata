"""
slide_dataset.py - Clase para la carga y transformación de imágenes en un conjunto de datos.

Este fichero contiene la implementación de la clase SlideDataset, la cual se utiliza para cargar
y transformar imágenes en un formato adecuado para su uso. Esta clase facilita la manipulación de 
imágenes para su posterior análisis, trabajo en lotes de datos y otras tareas relacionadas. 

Contenido:
  - SlideDataset(image_dir, transform)

Uso:
  from slide_dataset import SlideDataset
"""

from torch.utils.data import Dataset
import os
from PIL import Image

class SlideDataset(Dataset):
  def __init__(self, images_dir, transform=None):
    self.images_dir = images_dir
    self.image_files = [f for f in os.listdir(images_dir)]
    self.transform = transform

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    img_name = os.path.join(self.images_dir, self.image_files[idx])
    image = Image.open(img_name)

    if self.transform:
      image = self.transform(image)

    return {"name": self.image_files[idx], "image": image}

  def get_image_by_filename(self, filename):
    if filename in self.image_files:
      idx = self.image_files.index(filename)
      img_name = os.path.join(self.images_dir, self.image_files[idx])
      image = Image.open(img_name)
      return image
    else:
      raise ValueError(f'Imagen {filename} no encontrada')
      