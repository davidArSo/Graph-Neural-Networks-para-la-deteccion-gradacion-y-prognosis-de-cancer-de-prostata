"""
graph_utils.py - Utilidades para la construcción y visualización de grafos.

Este fichero contiene diversas funciones relacionadas con la construcción y visualización
de grafos en el ámbito de la histopatología. 

Contenido:
  - construct_graph(slide_id, image_dir, wsi_labels_df, latent_representations, device, only_contiguous=False, threshold=363)
  - visualize_graph(graph)
  
Uso:
  from graph_utils import construct_graph, visualize_graph
"""

import os
import torch
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import convert
from image_utils import parse_image_filename

def construct_graph(slide_id, image_dir, wsi_labels_df, latent_representations, device, only_contiguous=False, threshold=363):
  """
  Construye un grafo para un slide_id dado utilizando los parches y la escala de Gleason.

  Parámetros:
      slide_id (str): Slide ID para el cual se desea construir el grafo.
      image_dir (str): Ruta del directorio que contiene las imágenes.
      wsi_labels_df (dataframe): Dataframe con información de la WSI.
      latent_representations (dict): Diccionario de representaciones latentes
      device: Dispositivo (cpu o gpu)
      only_contiguous (bool): Booleano que indica si se conectan todos los nodos o solo los contiguos
      threshold (float): Valor de umbral. Solo tendrá efecto en caso de que only_contiguous = True

  Retorna:
      torch_geometric.data.Data: Grafo construido
  """

  # Buscamos todos los parches con el slide ID
  slides = [file for file in os.listdir(image_dir) if file.startswith(slide_id)]

  if not slides:
    print("No se encontraron imágenes para el Slide ID ", slide_id)
    return None

  # Ordenamos las slides por sus coordenadas
  slides.sort(key=lambda x: (int(parse_image_filename(x)['xini']),
                            int(parse_image_filename(x)['yini']),
                            parse_image_filename(x)['block_region']))

  # Lista para almacenar los nodos (atributos y posiciones) del grafo
  x = []
  pos = []

  # Obtenemos el máximo valor de 'yini' de todos los slides
  max_yini = max([int(parse_image_filename(slide)['yini']) for slide in slides])

  # Agregamos cada slide como un nodo al grafo
  for slide in slides:
    slide_info = parse_image_filename(slide)

    # Obtenemos la representación latente del diccionario y la agregamos como característica del nodo
    node_attr = latent_representations[slide]

    # Creamos un tensor con las coordenadas del nodo
    node_pos = torch.tensor([slide_info['xini'], max_yini - slide_info['yini']], dtype=torch.float)

    # Agregamos las características y coordenadas del nodo
    x.append(node_attr)
    pos.append(node_pos)

  # Convertimos las listas a tensores
  x = torch.stack(x)
  pos = torch.stack(pos)

  # Lista para almacenar las aristas (índices y atributos) del grafo
  edge_index = []
  edge_attr = []

  # Establecemos las conexiones entre parches
  for i in range(len(slides)):
    for j in range(i+1, len(slides)):
      pos_i, pos_j = pos[i], pos[j]

      # Calculamos la distancia entre las coordenadas de los parches
      distance_slices = torch.norm(pos_i - pos_j, p=2)

      # Si solo conectamos nodos contiguos y la distancia es mayor que un determinado umbral, continuamos
      if only_contiguous and distance_slices > threshold:
        continue

      # Agregamos las aristas de i a j y de j a i con la distancia como atributo
      edge_index.append((i, j))
      edge_attr.append(distance_slices.item())

      edge_index.append((j, i))
      edge_attr.append(distance_slices.item())

  # Convertimos las listas a tensores
  edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
  edge_attr = torch.tensor(edge_attr, dtype=torch.float)

  # Creamos el objeto Data del grafo
  graph = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

  # Establecemos el slide_id y el valor de Gleason del grafo
  graph.slide_id = slide_id
  graph.gleason_primary = torch.tensor(wsi_labels_df.loc[wsi_labels_df['slide_id'] == slide_id, 'Gleason_primary'].values[0])
  graph.gleason_secondary = torch.tensor(wsi_labels_df.loc[wsi_labels_df['slide_id'] == slide_id, 'Gleason_secondary'].values[0])
  graph.isup = torch.tensor(wsi_labels_df.loc[wsi_labels_df['slide_id'] == slide_id, 'isup'].values[0])

  # Liberamos variables no utilizadas
  del x, pos, edge_index, node_attr

  # Liberamos memoria caché
  if device == 'cuda':
    torch.cuda.empty_cache()

  return graph

def visualize_graph(graph):
    """
    Visualiza el grafo.

    Parámetros:
        graph (torch_geometric.data.Data): Grafo a visualizar.
    
    Retorna:
        None
    """

    # Obtenemos el slide_id y el gleason_score del grafo
    slide_id = graph.slide_id
    gleason_primary = graph.gleason_primary
    gleason_secondary = graph.gleason_secondary

    # Convertimos el grafo de PyTorch Geometric a un objeto NetworkX
    G = convert.to_networkx(graph)

    # Obtenemos las posiciones de los nodos
    pos = {i: (graph.pos[i][0], graph.pos[i][1]) for i in range(graph.num_nodes)}

    # Dibujamos y mostramos el grafo
    plt.figure(figsize=(15, 9))
    nx.draw_networkx(G, pos, with_labels=False, node_size=100, font_size=8, node_color='#E3A7B0')

    plt.title(f'Grafo para la WSI: {slide_id} con Gleason_primary: {gleason_primary} y Gleason_secondary: {gleason_secondary}')
    plt.axis('off')
    plt.show()
