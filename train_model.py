"""
train_model.py - Funciones para el entrenamiento y validación del modelo GNN.

Este fichero proporciona funciones de entrenamiento y validación del modelo GNN en tareas 
de clasificación de grados gleason (primario, secundario y ambos), isup y detección de cáncer.

Contenido:
  - train_gleason(model, device, train_loader, optimizer, criterion, gleason_mapping, train_primary=True, train_secondary=True)
  - validate_gleason(model, device, val_loader, criterion, gleason_mapping, validate_primary=True, validate_secondary=True)
  - train_validate_gleason(model, device, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, gleason_mapping, train_validate_primary=True, train_validate_secondary=True)
  - train_sum_gleason(model, device, train_loader, optimizer, criterion, gleason_mapping)
  - validate_sum_gleason(model, device, val_loader, criterion, gleason_mapping)
  - train_validate_sum_gleason(model, device, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, gleason_mapping)
  - train_isup(model, device, train_loader, optimizer, criterion)
  - validate_isup(model, device, validate_loader, criterion):
  - train_validate_isup(model, device, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs)
  - train_cancer_present(model, device, train_loader, optimizer, criterion)
  - validate_cancer_present(model, device, validate_loader, criterion)
  - train_validate_cancer_present(model, device, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs)

Uso:
  from train_model import train_validate_gleason, train_validate_sum_gleason, train_validate_isup, train_validate_cancer_present
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

def train_gleason(model, device, train_loader, optimizer, criterion, gleason_mapping, train_primary=True, train_secondary=True):
  model.to(device)
  model.train()

  running_loss = 0.0
  primary_correct = 0
  secondary_correct = 0
  correct_pairs = 0
  total = 0

  for batch_idx, graph in enumerate(train_loader):
    graph = graph.to(device)
    optimizer.zero_grad()

    gleason_primary_output, gleason_secondary_output = model(graph)

    gleason_primary_target = torch.tensor([gleason_mapping[label.item()] for label in graph.gleason_primary], device=device)
    gleason_secondary_target = torch.tensor([gleason_mapping[label.item()] for label in graph.gleason_secondary], device=device)

    if train_primary:
      loss_primary = criterion(gleason_primary_output, gleason_primary_target)
      running_loss += loss_primary.item()
      loss_primary.backward(retain_graph=train_secondary)

      _, primary_predicted = torch.max(gleason_primary_output.data, 1)
      primary_correct += (primary_predicted == gleason_primary_target).sum().item()

    if train_secondary:
      loss_secondary = criterion(gleason_secondary_output, gleason_secondary_target)
      running_loss += loss_secondary.item()
      loss_secondary.backward()

      _, secondary_predicted = torch.max(gleason_secondary_output.data, 1)
      secondary_correct += (secondary_predicted == gleason_secondary_target).sum().item()

    if train_primary and train_secondary:
      correct_pairs += ((primary_predicted == gleason_primary_target) & (secondary_predicted == gleason_secondary_target)).sum().item()
      
    optimizer.step()

    total += gleason_primary_target.size(0)

  primary_accuracy = primary_correct / total if train_primary else None
  secondary_accuracy = secondary_correct / total if train_secondary else None
  combined_accuracy = correct_pairs / total if train_primary and train_secondary else None

  avg_loss = running_loss / len(train_loader)

  return avg_loss, combined_accuracy, primary_accuracy, secondary_accuracy
  
def validate_gleason(model, device, val_loader, criterion, gleason_mapping, validate_primary=True, validate_secondary=True):
  model.to(device)
  model.eval()

  running_loss = 0.0
  primary_correct = 0
  secondary_correct = 0
  correct_pairs = 0
  total = 0

  with torch.no_grad():
    for batch_idx, graph in enumerate(val_loader):
      graph = graph.to(device)

      gleason_primary_output, gleason_secondary_output = model(graph)

      gleason_primary_target = torch.tensor([gleason_mapping[label.item()] for label in graph.gleason_primary], device=device)
      gleason_secondary_target = torch.tensor([gleason_mapping[label.item()] for label in graph.gleason_secondary], device=device)

      if validate_primary:
        loss_primary = criterion(gleason_primary_output, gleason_primary_target)
        running_loss += loss_primary.item()

        _, primary_predicted = torch.max(gleason_primary_output.data, 1)
        primary_correct += (primary_predicted == gleason_primary_target).sum().item()

      if validate_secondary:
        loss_secondary = criterion(gleason_secondary_output, gleason_secondary_target)
        running_loss += loss_secondary.item()

        _, secondary_predicted = torch.max(gleason_secondary_output.data, 1)
        secondary_correct += (secondary_predicted == gleason_secondary_target).sum().item()

      if validate_primary and validate_secondary:
        correct_pairs += ((primary_predicted == gleason_primary_target) & (secondary_predicted == gleason_secondary_target)).sum().item()

      total += gleason_primary_target.size(0)

  primary_accuracy = primary_correct / total if validate_primary else None
  secondary_accuracy = secondary_correct / total if validate_secondary else None
  combined_accuracy = correct_pairs / total if validate_primary and validate_secondary else None

  avg_loss = running_loss / len(val_loader)

  return avg_loss, combined_accuracy, primary_accuracy, secondary_accuracy
  
def train_validate_gleason(model, device, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, gleason_mapping, train_validate_primary=True, train_validate_secondary=True):

  train_losses = []
  train_primary_accuracies = []
  train_secondary_accuracies = []
  train_combined_accuracies = []
  val_losses = []
  val_primary_accuracies = []
  val_secondary_accuracies = []
  val_combined_accuracies = []

  for epoch in range(num_epochs):
    train_loss, train_combined_acc, train_primary_acc, train_secondary_acc = train_gleason(model, device, train_loader, optimizer, criterion, gleason_mapping, train_validate_primary, train_validate_secondary)
    val_loss, val_combined_acc, val_primary_acc, val_secondary_acc = validate_gleason(model, device, val_loader, criterion, gleason_mapping, train_validate_primary, train_validate_secondary)

    train_losses.append(train_loss)
    train_primary_accuracies.append(train_primary_acc)
    train_secondary_accuracies.append(train_secondary_acc)
    train_combined_accuracies.append(train_combined_acc)
    val_losses.append(val_loss)
    val_primary_accuracies.append(val_primary_acc)
    val_secondary_accuracies.append(val_secondary_acc)
    val_combined_accuracies.append(val_combined_acc)

    print_string = f'Epoch [{epoch + 1}/{num_epochs}] - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}'

    if train_validate_primary and not train_validate_secondary:
      print_string += f' - train_primary_acc: {train_primary_acc:.4f} - val_primary_acc: {val_primary_acc:.4f}'
    elif not train_validate_primary and train_validate_secondary:
      print_string += f' - train_secondary_acc: {train_secondary_acc:.4f} - val_secondary_acc: {val_secondary_acc:.4f}'
    else:
      print_string += f' - train_combined_acc: {train_combined_acc:.4f} - val_combined_acc: {val_combined_acc:.4f}'

    print(print_string)

    scheduler.step() 

  # Creamos el gráfico
  plt.style.use("ggplot")
  plt.figure()

  plt.plot(np.arange(1, num_epochs + 1), train_losses, label='Train Loss')
  plt.plot(np.arange(1, num_epochs + 1), val_losses, label='Val Loss')

  if train_validate_primary and not train_validate_secondary:
    plt.plot(np.arange(1, num_epochs + 1), train_primary_accuracies, label='Train Primary Accuracy')
    plt.plot(np.arange(1, num_epochs + 1), val_primary_accuracies, label='Val Primary Accuracy')
  elif not train_validate_primary and train_validate_secondary:
    plt.plot(np.arange(1, num_epochs + 1), train_secondary_accuracies, label='Train Secondary Accuracy')
    plt.plot(np.arange(1, num_epochs + 1), val_secondary_accuracies, label='Val Secondary Accuracy')
  else:
    plt.plot(np.arange(1, num_epochs + 1), train_combined_accuracies, label='Train Combined Accuracy')
    plt.plot(np.arange(1, num_epochs + 1), val_combined_accuracies, label='Val Combined Accuracy')

  plt.title("Métricas de entrenamiento y validación en la predicción de grados Gleason")
  plt.xlabel("Épocas")
  plt.ylabel("Pérdida (Loss) / Precisión (Accuracy)")
  plt.legend()

  plt.show()
  
def train_sum_gleason(model, device, train_loader, optimizer, criterion, gleason_mapping):
  model.to(device)
  model.train()

  running_loss = 0.0
  correct = 0
  total = 0

  for batch_idx, graph in enumerate(train_loader):
    graph = graph.to(device)
    optimizer.zero_grad()

    gleason_output = model(graph)

    gleason_labels = [primary.item() + secondary.item() for primary, secondary in zip(graph.gleason_primary, graph.gleason_secondary)]
    gleason_target = torch.tensor([gleason_mapping[gleason_label] for gleason_label in gleason_labels], device=device)

    loss = criterion(gleason_output, gleason_target)
    running_loss += loss.item()
    loss.backward()

    _, predicted = torch.max((gleason_output).data, 1)
    correct += (predicted == gleason_target).sum().item()
   
    optimizer.step()
    total += gleason_target.size(0)

  avg_loss = running_loss / len(train_loader)
  accuracy = correct / total

  return avg_loss, accuracy
  
def validate_sum_gleason(model, device, val_loader, criterion, gleason_mapping):
  model.to(device)
  model.eval()

  running_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
    for batch_idx, graph in enumerate(val_loader):
      graph = graph.to(device)

      gleason_output = model(graph)

      gleason_labels = [primary.item() + secondary.item() for primary, secondary in zip(graph.gleason_primary, graph.gleason_secondary)]
      gleason_target = torch.tensor([gleason_mapping[label] for label in gleason_labels], device=device)

      loss = criterion(gleason_output, gleason_target)
      running_loss += loss.item()

      _, predicted = torch.max(gleason_output.data, 1)
      correct += (predicted == gleason_target).sum().item()

      total += gleason_target.size(0)

  avg_loss = running_loss / len(val_loader)
  accuracy = correct / total

  return avg_loss, accuracy

def train_validate_sum_gleason(model, device, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, gleason_mapping):

  train_losses = []
  val_losses = []
  train_accuracies = []
  val_accuracies = []

  for epoch in range(num_epochs):
    train_loss, train_acc = train_sum_gleason(model, device, train_loader, optimizer, criterion, gleason_mapping)
    val_loss, val_acc = validate_sum_gleason(model, device, val_loader, criterion, gleason_mapping)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print_string = (f'Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f} - '
                    f'Train Accuracy: {train_acc:.4f} - Val Loss: {val_loss:.4f} - '
                    f'Val Accuracy: {val_acc:.4f}')
    print(print_string)

    scheduler.step() 

  # Creación del gráfico
  plt.style.use("ggplot")
  plt.figure()
  plt.plot(np.arange(1, num_epochs + 1), train_losses, label='Train Loss')
  plt.plot(np.arange(1, num_epochs + 1), val_losses, label='Val Loss')
  plt.plot(np.arange(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
  plt.plot(np.arange(1, num_epochs + 1), val_accuracies, label='Val Accuracy')

  plt.title("Training and Validation Metrics for Gleason Grade Prediction")
  plt.xlabel("Epochs")
  plt.ylabel("Loss / Accuracy")
  plt.legend()
  plt.show()

def train_isup(model, device, train_loader, optimizer, criterion):
  model.to(device)
  model.train()

  running_loss = 0.0
  correct = 0
  total = 0

  for batch_idx, graph in enumerate(train_loader):
    graph = graph.to(device)
    optimizer.zero_grad()

    isup_output = model(graph)

    isup_target = torch.tensor(graph.isup, device=device)

    loss = criterion(isup_output, isup_target)
    running_loss += loss.item()
    loss.backward()

    _, predicted = torch.max(isup_output.data, 1)
    correct += (predicted == isup_target).sum().item()

    optimizer.step()

    total += isup_target.size(0)

  accuracy = correct / total
  avg_loss = running_loss / len(train_loader)

  return avg_loss, accuracy
  
def validate_isup(model, device, validate_loader, criterion):
  model.to(device)
  model.eval()  

  running_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad(): 
    for batch_idx, graph in enumerate(validate_loader):
      graph = graph.to(device)

      isup_output = model(graph)

      isup_target = torch.tensor(graph.isup, device=device)

      loss = criterion(isup_output, isup_target)
      running_loss += loss.item()

      _, predicted = torch.max(isup_output.data, 1)
      correct += (predicted == isup_target).sum().item()

      total += isup_target.size(0)

  accuracy = correct / total
  avg_loss = running_loss / len(validate_loader)

  return avg_loss, accuracy
  
def train_validate_isup(model, device, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs):

  train_losses = []
  train_accuracies = []
  val_losses = []
  val_accuracies = []

  for epoch in range(num_epochs):
    train_loss, train_acc = train_isup(model, device, train_loader, optimizer, criterion)
    val_loss, val_acc = validate_isup(model, device, val_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print_string = f'Epoch [{epoch + 1}/{num_epochs}] - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}'
    print(print_string)

    scheduler.step()

  plt.style.use("ggplot")
  plt.figure()

  plt.plot(np.arange(1, num_epochs + 1), train_losses, label='Train Loss')
  plt.plot(np.arange(1, num_epochs + 1), val_losses, label='Val Loss')
  plt.plot(np.arange(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
  plt.plot(np.arange(1, num_epochs + 1), val_accuracies, label='Val Accuracy')

  plt.title("Métricas de entrenamiento y validación en la predicción del ISUP")
  plt.xlabel("Épocas")
  plt.ylabel("Pérdida (Loss)/ Precisión (Accuracy)")
  plt.legend()

  plt.show()
  
def train_cancer_present(model, device, train_loader, optimizer, criterion):
  model.to(device)
  model.train()

  running_loss = 0.0
  correct = 0
  total = 0

  for batch_idx, graph in enumerate(train_loader):
    graph = graph.to(device)
    optimizer.zero_grad()

    output = model(graph)

    is_cancer_present = (graph.isup > 0).long()  
    is_cancer_present = is_cancer_present.to(device)

    loss = criterion(output, is_cancer_present)
    running_loss += loss.item()
    loss.backward()

    _, predicted = torch.max(output.data, 1)
    correct += (predicted == is_cancer_present).sum().item()

    optimizer.step()

    total += is_cancer_present.size(0)

  accuracy = correct / total
  avg_loss = running_loss / len(train_loader)

  return avg_loss, accuracy
  
def validate_cancer_present(model, device, validate_loader, criterion):
  model.to(device)
  model.eval()  

  running_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():  
    for batch_idx, graph in enumerate(validate_loader):
      graph = graph.to(device)

      output = model(graph)

      # Mapeamos la característica isup para determinar la presencia de cáncer
      is_cancer_present = (graph.isup > 0).long()  
      is_cancer_present = is_cancer_present.to(device)

      loss = criterion(output, is_cancer_present)
      running_loss += loss.item()

      _, predicted = torch.max(output.data, 1)
      correct += (predicted == is_cancer_present).sum().item()

      total += is_cancer_present.size(0)

  accuracy = correct / total
  avg_loss = running_loss / len(validate_loader)

  return avg_loss, accuracy
  
def train_validate_cancer_present(model, device, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs):

  train_losses = []
  train_accuracies = []
  val_losses = []
  val_accuracies = []

  for epoch in range(num_epochs):
    train_loss, train_acc = train_cancer_present(model, device, train_loader, optimizer, criterion)
    val_loss, val_acc = validate_cancer_present(model, device, val_loader, criterion)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print_string = f'Epoch [{epoch + 1}/{num_epochs}] - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - train_acc: {train_acc:.4f} - val_acc: {val_acc:.4f}'
    print(print_string)

    scheduler.step()

  plt.style.use("ggplot")
  plt.figure()

  plt.plot(np.arange(1, num_epochs + 1), train_losses, label='Train Loss')
  plt.plot(np.arange(1, num_epochs + 1), val_losses, label='Val Loss')
  plt.plot(np.arange(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
  plt.plot(np.arange(1, num_epochs + 1), val_accuracies, label='Val Accuracy')

  plt.title("Métricas de entrenamiento y validación en detección de cancer")
  plt.xlabel("Épocas")
  plt.ylabel("Pérdida (Loss) / Precisión (Accuracy)")
  plt.legend()

  plt.show()
  