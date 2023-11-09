"""
graph_dataset.py - Clase para la generación de grafos a partir de imágenes

Este fichero contiene la implementación de la clase GraphDataset, la cual se utiliza para generar
grafos en un formato adecuado para su posterior uso en las GNN. Esta clase facilita la manipulación
de grafos para su posterior análisis, trabajo en lotes de datos y otras tareas relacionadas. 

Contenido:
  - GraphDataset(wsi_labels_df, data_dir, device, latent_representations, rag, transform=None)

Uso:
  from graph_dataset import GraphDataset
"""

from torch.utils.data import Dataset
from tqdm import tqdm
from graph_utils import construct_graph

class GraphDataset(Dataset):
  def __init__(self, wsi_labels_df, data_dir, device, latent_representations, rag, transform=None):
    self.wsi_labels_df = wsi_labels_df
    self.data_dir = data_dir
    self.device = device
    self.latent_representations = latent_representations
    self.rag = rag
    self.transform = transform
    self.data = self.load_data()

  def load_data(self):
    # Creamos una lista para almacenar los grafos
    graph_list = []

    # Iteramos sobre los slide_id y construimos los grafos
    for slide_id in tqdm(self.wsi_labels_df['slide_id'], desc='Generando grafos ...', unit='slide'):
      graph = construct_graph(slide_id, self.data_dir, self.wsi_labels_df, self.latent_representations, self.device, only_contiguous=self.rag, threshold=363)
      graph_list.append(graph)

    return graph_list

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    data = self.data[idx]
    if self.transform:
      data = self.transform(data)
    return data

  def get(self, idx):
    return self.data[idx]

  def len(self):
    return len(self.data)
