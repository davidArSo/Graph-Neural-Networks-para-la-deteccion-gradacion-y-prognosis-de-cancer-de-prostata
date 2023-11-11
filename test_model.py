"""
test_model.py - Funciones para la evaluación del modelo GNN.

Este fichero proporciona funciones de evaluación del modelo GNN en tareas 
de clasificación de grados gleason (primario, secundario y ambos), isup y detección de cáncer.

Contenido:
  - test_gleason(model, device, test_loader, criterion, gleason_mapping, test_primary=True, test_secondary=True)
  - test_sum_gleason(model, device, test_loader, criterion, gleason_mapping)
  - test_isup(model, device, test_loader, criterion)
  - test_cancer_present(model, device, test_loader, criterion)

Uso:
  from test_model import test_gleason, test_sum_gleason, test_isup, test_cancer_present
"""

import torch
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score

def test_gleason(model, device, test_loader, criterion, gleason_mapping, test_primary=True, test_secondary=True):
  model.to(device)
  model.eval()

  running_loss = 0.0
  primary_correct = 0
  secondary_correct = 0
  total = 0

  predicted_primary_labels = []
  predicted_secondary_labels = []
  primary_true_labels = []
  secondary_true_labels = []

  with torch.no_grad():
    for batch_idx, graph in enumerate(test_loader):
      graph = graph.to(device)

      gleason_primary_output, gleason_secondary_output = model(graph)

      loss = 0.0

      if test_primary:
        gleason_primary_target = torch.tensor([gleason_mapping[label.item()] for label in graph.gleason_primary], device=device)
        loss_primary = criterion(gleason_primary_output, gleason_primary_target)
        loss += loss_primary

        _, primary_predicted = torch.max(gleason_primary_output.data, 1)
        primary_correct += (primary_predicted == gleason_primary_target).sum().item()

        predicted_primary_labels.extend(primary_predicted.cpu().numpy())
        primary_true_labels.extend(gleason_primary_target.cpu().numpy())

      if test_secondary:
        gleason_secondary_target = torch.tensor([gleason_mapping[label.item()] for label in graph.gleason_secondary], device=device)
        loss_secondary = criterion(gleason_secondary_output, gleason_secondary_target)
        loss += loss_secondary

        _, secondary_predicted = torch.max(gleason_secondary_output.data, 1)
        secondary_correct += (secondary_predicted == gleason_secondary_target).sum().item()

        predicted_secondary_labels.extend(secondary_predicted.cpu().numpy())
        secondary_true_labels.extend(gleason_secondary_target.cpu().numpy())

      running_loss += loss.item()
      total += graph.gleason_primary.size(0)

    primary_accuracy = (primary_correct / total) if test_primary else None
    secondary_accuracy = (secondary_correct / total) if test_secondary else None

    avg_loss = running_loss / len(test_loader)

    if test_primary and test_secondary:

      combined_labels = [f"{p}+{s}" for p, s in zip(primary_true_labels, secondary_true_labels)]
      combined_predicted = [f"{p}+{s}" for p, s in zip(predicted_primary_labels, predicted_secondary_labels)]

      correct_combined = sum(1 for true, pred in zip(combined_labels, combined_predicted) if true == pred)
      combined_accuracy = correct_combined / total

      precision, recall, f1, _ = precision_recall_fscore_support(combined_labels, combined_predicted, average='weighted', zero_division=1)
      kappa = cohen_kappa_score(combined_labels, combined_predicted)

      return avg_loss, combined_accuracy, precision, recall, f1, kappa

    else:
      if test_primary:
          true_labels = primary_true_labels
          predicted_labels = predicted_primary_labels
      else:
          true_labels = secondary_true_labels
          predicted_labels = predicted_secondary_labels

      precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted', zero_division=1)
      kappa = cohen_kappa_score(true_labels, predicted_labels)

      return avg_loss, primary_accuracy or secondary_accuracy, precision, recall, f1, kappa

def test_sum_gleason(model, device, test_loader, criterion, gleason_mapping):
  model.to(device)
  model.eval()

  running_loss = 0.0
  correct = 0
  total = 0
  predicted_labels = []
  true_labels = []

  with torch.no_grad():
      for batch_idx, graph in enumerate(test_loader):
        graph = graph.to(device)

        gleason_output = model(graph)

        gleason_labels = [primary.item() + secondary.item() for primary, secondary in zip(graph.gleason_primary, graph.gleason_secondary)]
        gleason_target = torch.tensor([gleason_mapping[label] for label in gleason_labels], device=device)

        loss = criterion(gleason_output, gleason_target)
        running_loss += loss.item()

        _, predicted = torch.max(gleason_output.data, 1)
        correct += (predicted == gleason_target).sum().item()
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(gleason_target.cpu().numpy())

        total += gleason_target.size(0)

  avg_loss = running_loss / len(test_loader)
  accuracy = correct / total
  precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted', zero_division=1)
  kappa = cohen_kappa_score(true_labels, predicted_labels)

  return avg_loss, accuracy, precision, recall, f1, kappa
  
def test_isup(model, device, test_loader, criterion):
  model.to(device)
  model.eval()

  running_loss = 0.0
  correct = 0
  total = 0

  predicted_labels = []
  true_labels = []

  with torch.no_grad():
    for batch_idx, graph in enumerate(test_loader):
      graph = graph.to(device)

      isup_output = model(graph)
      isup_target = graph.isup

      _, predicted = torch.max(isup_output.data, 1)

      loss = criterion(isup_output, isup_target)
      correct += (predicted == isup_target).sum().item()

      predicted_labels.extend(predicted.cpu().numpy())
      true_labels.extend(isup_target.cpu().numpy())

      running_loss += loss.item()
      total += isup_target.size(0)

    accuracy = correct / total
    avg_loss = running_loss / len(test_loader)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='weighted', zero_division=1)
    kappa = cohen_kappa_score(true_labels, predicted_labels)

    return avg_loss, accuracy, precision, recall, f1, kappa
    
def test_cancer_present(model, device, test_loader, criterion):
  model.to(device)
  model.eval()

  running_loss = 0.0
  correct = 0
  total = 0

  predicted_labels = []
  true_labels = []

  with torch.no_grad():
    for batch_idx, graph in enumerate(test_loader):
      graph = graph.to(device)

      output = model(graph)

      cancer_target = (graph.isup > 0).long().to(device)
      true_labels.extend(cancer_target.cpu().numpy())

      loss = criterion(output, cancer_target)
      running_loss += loss.item()

      _, predicted = torch.max(output.data, 1)
      predicted_labels.extend(predicted.cpu().numpy())

      correct += (predicted == cancer_target).sum().item()
      total += cancer_target.size(0)

    accuracy = correct / total
    avg_loss = running_loss / len(test_loader)

    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='binary', zero_division=1)
    kappa = cohen_kappa_score(true_labels, predicted_labels)

    return avg_loss, accuracy, precision, recall, f1, kappa
