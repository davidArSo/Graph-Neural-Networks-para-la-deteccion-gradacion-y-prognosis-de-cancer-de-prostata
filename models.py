"""
models.py - Clase para la definición de modelos basados en Graph Neural Networks

Este fichero contiene la implementación de la clase SegGiniGraphHead, diseñada para operar
sobre grafos de imágenes histopatológicas, específicamente para problemas de clasificación
de grado de Gleason o Isup en patología digital de próstata. La estructura de la red está basada 
en GINConv. 

Contenido:
  - SegGiniGraphHead(num_gin_layers, num_features, hidden_dims, num_classes, gin_dropout_rates, final_dropout_rate=0.5, use_batch_norms=True, mode="gleason",  use_global_avg_pool=True)

Uso:
  from models import SegGiniGraphHead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool

class SegGiniGraphHead(nn.Module):
  def __init__(self, num_gin_layers, num_features, hidden_dims, num_classes, gin_dropout_rates, final_dropout_rate=0.5, use_batch_norms=True, mode="gleason",  use_global_avg_pool=True):
    super(SegGiniGraphHead, self).__init__()

    assert mode in ["gleason", "isup"], "El modo solo puede ser gleason o isup"
    
    # Atributos de clase
    self.num_gin_layers = num_gin_layers
    self.use_global_avg_pool = use_global_avg_pool
    self.mode = mode
    self.gin_layers = torch.nn.ModuleList()
    self.final_dropout_rate = final_dropout_rate
    
    # Configuración de las capas GINConv
    for idx in range(self.num_gin_layers):
      layers = []
      
      input_dim = num_features if idx == 0 else hidden_dims[idx*2-1]
            
      layers.append(nn.Linear(input_dim, hidden_dims[idx*2]))
      if use_batch_norms[idx]:
          layers.append(nn.BatchNorm1d(hidden_dims[idx*2]))
      layers.append(nn.ReLU())
      
      if gin_dropout_rates[idx] > 0:
        layers.append(nn.Dropout(gin_dropout_rates[idx]))
          
      layers.append(nn.Linear(hidden_dims[idx*2], hidden_dims[idx*2+1]))
      if use_batch_norms[idx]:
          layers.append(nn.BatchNorm1d(hidden_dims[idx*2+1]))
      
      self.gin_layers.append(GINConv(nn.Sequential(*layers)))

    self.readout = global_mean_pool

    if final_dropout_rate > 0:
      self.final_dropout = nn.Dropout(final_dropout_rate)

    if self.mode == "gleason":
      # Capas MLP separadas para Gleason Primary y Gleason Secondary
      self.gleason_primary = nn.Linear(hidden_dims[-1], num_classes)
      self.gleason_secondary = nn.Linear(hidden_dims[-1], num_classes)
    elif self.mode == "isup": 
      # Capa MLP única para el grado isup
      self.isup = nn.Linear(hidden_dims[-1], num_classes)

  def forward(self, data):
    x, edge_index = data.x, data.edge_index

    if self.use_global_avg_pool:
      x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
    else:
      x = x.view(x.size(0), -1)

    for gin in self.gin_layers:
      x = F.relu(gin(x, edge_index))

    x = self.readout(x, data.batch)

    if self.final_dropout_rate > 0:
      x = self.final_dropout(x)

    if self.mode == "gleason":
      primary_output = self.gleason_primary(x)
      secondary_output = self.gleason_secondary(x)
      return primary_output, secondary_output
    else:  # mode == "isup"
      isup_output = self.isup(x)
      return isup_output
     