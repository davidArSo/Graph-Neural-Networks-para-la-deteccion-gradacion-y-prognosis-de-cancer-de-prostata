"""
graph_augmentation.py - Técnicas para el aumento de grafos. 

Este fichero contiene diversas funciones relacionadas con técnicas de data augmentation aplicadas 
a grafos.

Contenido:
  - add_gaussian_noise(graph, sigma=0.05)
  - drop_edges(graph, p=0.2)
  - augment_graph_dataset(graph_dataset, num_augmentations=1, sigma=0.05, drop_probability=0.2)
  - adaptive_minority_over_sampling(graph_dataset, num_augmentations=None, sigma=0.05, drop_probability=0.2)

Uso:
  from graph_augmentation import add_gaussian_noise, drop_edges, augment_graph_dataset, adaptive_minority_over_sampling
"""

import torch
import random

def add_gaussian_noise(graph, sigma=0.05):
  """
  Añade ruido gaussiano a las características de los nodos.

  Parámetros:
    graph (Graph): Grafo al cual añadir ruido.
    sigma (float): Desviación estándar del ruido gaussiano.

  Retorna:
    Graph: Grafo con ruido gaussiano añadido.
  """
  noisy_graph = graph.clone()
  noise = torch.randn_like(noisy_graph.x) * sigma
  noisy_graph.x += noise
  return noisy_graph

def drop_edges(graph, p=0.2):
  """
  Elimina una proporción p de aristas.

  Parámetros:
    graph (Graph): El grafo al cual eliminar bordes.
    p (float): Proporción de bordes a eliminar.

  Retorna:
    Graph: Grafo con una fracción de bordes eliminados.
  """
  edge_indices = list(range(graph.edge_index.shape[1]))
  num_edges_to_remove = int(p * len(edge_indices))

  edges_to_remove = random.sample(edge_indices, num_edges_to_remove)
  mask = torch.tensor([i not in edges_to_remove for i in edge_indices])

  dropped_graph = graph.clone()
  dropped_graph.edge_index = graph.edge_index[:, mask]

  return dropped_graph

def augment_graph_dataset(graph_dataset, num_augmentations=1, sigma=0.05, drop_probability=0.2):
  """
  Aumenta el conjunto de grafos añadiendo ruido gaussiano y eliminando aristas.

  Parámetros:
    graph_dataset (GraphDataset): Conjunto de grafos a aumentar.
    num_augmentations (int): Número de grafos a añadir por elemento del conjunto.
    sigma (float): Desviación estándar del ruido gaussiano.
    drop_probability (float): Probabilidad de eliminar un borde.

  Retorna:
    list[Graph]: Lista de grafos aumentados.
  """
  augmented_graphs = []

  for graph in graph_dataset:
    # Agregamos el grafo original
    augmented_graphs.append(graph)

    # Añadimos ruido gaussiano y eliminación de aristas 
    for _ in range(num_augmentations):
        augmented_graph = add_gaussian_noise(graph, sigma)
        augmented_graph = drop_edges(augmented_graph, drop_probability)
        augmented_graphs.append(augmented_graph)

  return augmented_graphs

def adaptive_minority_over_sampling(graph_dataset, num_augmentations=None, sigma=0.05, drop_probability=0.2):
  """
  Sobremuestreo adaptativo de clases minoritarias basado en la puntuación combinada de Gleason.

  Parámetros:
    graph_dataset (GraphDataset): Conjunto de grafos para sobremuestreo.
    num_augmentations (int or None): Número de grafos a añadir por elemeto, o None para auto-cálculo.
    sigma (float): Desviación estándar del ruido gaussiano.
    drop_probability (float): Probabilidad de eliminar un borde.

  Retorna:
    list[Graph]: Lista de grafos sobremuestreados.
  """
  # Calculamos la puntuación combinada de Gleason y contamos ocurrencias
  gleason_counts = {}
  for graph in graph_dataset:
    gleason_score = f"{graph.gleason_primary.item()}+{graph.gleason_secondary.item()}"
    gleason_counts[gleason_score] = gleason_counts.get(gleason_score, 0) + 1

  # Determinamos el objetivo de sobremuestreo
  max_count = max(gleason_counts.values())

  # Si no se especifica num_augmentations, usa el valor necesario para equilibrar las clases
  if num_augmentations is None:
    num_augmentations = {}
    for gleason_score, count in gleason_counts.items():
      num_augmentations[gleason_score] = (max_count - count) // count
  else:
    num_augmentations = {gleason_score: num_augmentations for gleason_score in gleason_counts.keys()}

  # Sobremuestreo adaptativo
  augmented_graphs = list(graph_dataset)
  for graph in graph_dataset:
    gleason_score = f"{graph.gleason_primary.item()}+{graph.gleason_secondary.item()}"
    for _ in range(num_augmentations[gleason_score]):
      augmented_graph = add_gaussian_noise(graph, sigma)
      augmented_graph = drop_edges(augmented_graph, drop_probability)
      augmented_graphs.append(augmented_graph)

    return augmented_graphs
    