"""
feature_extractor.py - Incluye clases para la extracción de características (representaciones latentes) de parches

Este fichero contiene la implementación de las clases PatchEmbeddingPretrained y PatchEmbeddingUNet. Estas clases
permiten la extracción de representaciones latentes de parches mediante redes preentrenadas o arquitectura U-Net.

Contenido:
  - PatchEmbeddingPretrained (model_name, device)
  - PatchEmbeddingUNet (dimensions, device)

Uso:
  from feature_extractor import PatchEmbeddingPretrained, PatchEmbeddingUNet 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from transformers import AutoModel
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

class PatchEmbeddingPretrained(nn.Module):
  def __init__(self, model_name, device='cuda'):
    super(PatchEmbeddingPretrained, self).__init__()

    # Establecemos el 'device' y 'model_name' como atributos 
    self.device = device
    self.model_name = model_name

    # Movemos el modelo al dispositivo
    self.to(device)

    # Cargamos el modelo preentrenado
    if model_name == 'vgg19':
      self.pretrained_model = models.vgg19(pretrained=True).to(self.device)
    elif model_name == 'resnet50':
      self.pretrained_model = models.resnet50(pretrained=True).to(self.device)
    elif model_name == 'mobilenet_v2':
      self.pretrained_model = models.mobilenet_v2(pretrained=True).to(self.device)
    elif model_name == 'PANDA_ConvNeXT':
      self.pretrained_model = AutoModel.from_pretrained('smc/PANDA_ConvNeXT_K').to(self.device)
    else:
      raise ValueError(f"No se reconoce el nombre del modelo '{model_name}'. Elige entre ['vgg19', 'resnet50', 'mobilenet_v2', 'PANDA_ConvNeXT'].")

    # Eliminamos las últimas capas
    if model_name != 'PANDA_ConvNeXT':
      self.pretrained_model = nn.Sequential(*list(self.pretrained_model.children())[:-1])

  def forward(self, x):
    x = x.to(self.device)

    if self.model_name == 'PANDA_ConvNeXT':
      x = self.pretrained_model(x).last_hidden_state
    else:
      x = self.pretrained_model(x)

    return x

  def get_latent_representation(self, slide):
    """
    Obtiene la representación latente de un parche.

    Parámetros:
        slide : Parche de entrada en formato (C, H, W), donde C es el número de canales,
                H es la altura y W es el ancho.

    Retorna:
        torch.Tensor: Representación latente del parche
    """

    # Convertimos a tensor y movemos al mismo dispositivo
    slide_array = np.array(slide)
    slide_tensor = torch.from_numpy(slide_array).permute(2, 0, 1).unsqueeze(0).float()
    slide_tensor = slide_tensor.to(self.device)

    # Cambiamos al modo de evaluación
    self.eval()

    # Desactivamos el cálculo de gradientes durante la propagación
    with torch.no_grad():
      latent_representation = self.forward(slide_tensor).squeeze()

    # Volvemos al modo de entrenamiento
    self.train()

    del slide_array, slide_tensor

    return latent_representation.cpu().squeeze()

  def get_all_latent_representation(self, dataset, batch_size=32, num_workers=4):
    """
    Obtiene las representaciones latentes de todos parche.

    Parámetros:
        dataset : Conjunto de datos con todas las imágenes

    Retorna:
        dict: Un diccionario que mapea nombres de parches a sus representaciones latentes como torch.Tensor.
    """

    # Uso de un DataLoader con batch_size y num_workers
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Lista de representaciones latentes
    latent_representations = {}

    # Cambiamos al modo de evaluación
    self.eval()

    # Desactivamos el cálculo de gradientes durante la propagación
    with torch.no_grad():
      # Nos desplazamos por batch de muestras
      for batch in tqdm(dataloader, desc="Procesando lotes ...", unit="batch"):
        batch_names = batch["name"]
        batch_images = batch["image"]
        batch_representation = self.forward(batch_images)

        for name, representation in zip(batch_names, batch_representation):
          latent_representations[name] = representation.cpu().squeeze()

        if self.device == 'cuda':
          torch.cuda.empty_cache()

        del batch, batch_representation

    # Volvemos al modo de entrenamiento
    self.train()

    return latent_representations
    

class PatchEmbeddingUNet(nn.Module):
  def __init__(self, dimensions=[64,128,256,512,1024], device='cuda'):
    super(PatchEmbeddingUNet, self).__init__()

    assert len(dimensions) == 5, "La lista de dimensiones debe tener exactamente 5 elementos."

    # Establecemos el 'device' como un atributo de la clase
    self.device = device

    # Movemos el modelo al dispositivo
    self.to(device)

     # Contracting path
    self.enc1 = self.conv_block(3, dimensions[0])
    self.enc2 = self.conv_block(dimensions[0], dimensions[1])
    self.enc3 = self.conv_block(dimensions[1], dimensions[2])
    self.enc4 = self.conv_block(dimensions[2], dimensions[3])
    self.center = self.conv_block(dimensions[3], dimensions[4])

    # Expansive path
    self.upconv4 = nn.ConvTranspose2d(dimensions[4], dimensions[3], kernel_size=2, stride=2)
    self.dec4 = self.conv_block(dimensions[4], dimensions[3])
    self.upconv3 = nn.ConvTranspose2d(dimensions[3], dimensions[2], kernel_size=2, stride=2)
    self.dec3 = self.conv_block(dimensions[3], dimensions[2])
    self.upconv2 = nn.ConvTranspose2d(dimensions[2], dimensions[1], kernel_size=2, stride=2)
    self.dec2 = self.conv_block(dimensions[2], dimensions[1])
    self.upconv1 = nn.ConvTranspose2d(dimensions[1], dimensions[0], kernel_size=2, stride=2)
    self.dec1 = self.conv_block(dimensions[1], dimensions[0])

    self.final = nn.Conv2d(dimensions[0], 3, kernel_size=1)

  def conv_block(self, in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        #nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        #nn.BatchNorm2d(out_channels)
    )

  def contractive_path(self, x):
    x = x.to(self.device)

    enc1_out = self.enc1(x)
    enc2_out = self.enc2(F.max_pool2d(enc1_out, kernel_size=2))
    enc3_out = self.enc3(F.max_pool2d(enc2_out, kernel_size=2))
    enc4_out = self.enc4(F.max_pool2d(enc3_out, kernel_size=2))
    center_out = self.center(F.max_pool2d(enc4_out, kernel_size=2))

    return enc1_out, enc2_out, enc3_out, enc4_out, center_out

  def forward(self, x):
    x = x.to(self.device)

    # Contracting path
    enc1_out, enc2_out, enc3_out, enc4_out, center_out = self.contractive_path(x)

    # Expansive path
    up4 = self.upconv4(center_out)
    merge4 = torch.cat([enc4_out, up4], 1)
    dec4_out = self.dec4(merge4)

    up3 = self.upconv3(dec4_out)
    merge3 = torch.cat([enc3_out, up3], 1)
    dec3_out = self.dec3(merge3)

    up2 = self.upconv2(dec3_out)
    merge2 = torch.cat([enc2_out, up2], 1)
    dec2_out = self.dec2(merge2)

    up1 = self.upconv1(dec2_out)
    merge1 = torch.cat([enc1_out, up1], 1)
    dec1_out = self.dec1(merge1)

    # Salida final
    out = self.final(dec1_out)

    return torch.sigmoid(out)

  def train_model(self, train_loader, val_loader, optimizer, criterion, num_epochs=20):

    # Valores de loss y métricas para entrenamiento y validación
    train_losses = []
    val_losses = []

    train_psnr = []
    val_psnr = []

    train_ssim = []
    val_ssim = []

    # Por cada época ...
    for epoch in range(num_epochs):

      # Cambiamos al modo de entrenamiento
      self.train()

      # Inicializamos valores
      total_loss = 0.0
      total_psnr = 0.0
      total_ssim = 0.0

      # Usamos batch de muestras
      for batch in train_loader:
        images = batch["image"].to(self.device)

        optimizer.zero_grad()
        outputs = self.forward(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        psnr = peak_signal_noise_ratio(images.detach().cpu().numpy(), outputs.detach().cpu().numpy(), data_range=1.0)
        ssim = structural_similarity(images.detach().cpu().numpy(), outputs.detach().cpu().numpy(), channel_axis=0, win_size=3)

        total_psnr += psnr
        total_ssim += ssim

        if self.device == 'cuda':
          torch.cuda.empty_cache()

        del images, outputs, loss

      avg_loss = total_loss / len(train_loader)
      avg_psnr = total_psnr / len(train_loader)
      avg_ssim = total_ssim / len(train_loader)

      val_loss, val_avg_psnr, val_avg_ssim = self.eval_model(val_loader, criterion)

      train_losses.append(avg_loss)
      val_losses.append(val_loss)
      train_psnr.append(avg_psnr)
      val_psnr.append(val_avg_psnr)
      train_ssim.append(avg_ssim)
      val_ssim.append(val_avg_ssim)

      print(f'Epoch [{epoch+1}/{num_epochs}] - '
            f'train_loss: {avg_loss:.4f} - train_PSNR: {avg_psnr:.4f} - train_SSIM: {avg_ssim:.4f} - '
            f'val_loss: {val_loss:.4f} - val_PSNR: {val_avg_psnr:.4f} - val_SSIM: {val_avg_ssim:.4f}')

    plt.style.use("ggplot")

    # Creamos una figura con dos subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Subplot para pérdida (Loss)
    axs[0].plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    axs[0].plot(range(1, num_epochs+1), val_losses, label='Val Loss')
    axs[0].set_title("Pérdida (Loss)")
    axs[0].set_xlabel("Épocas")
    axs[0].set_ylabel("Pérdida (Loss)")
    axs[0].legend()

    # Subplot para métricas PSNR y SSIM
    axs[1].plot(range(1, num_epochs+1), train_psnr, label='Train PSNR', color='orange')
    axs[1].plot(range(1, num_epochs+1), val_psnr, label='Val PSNR', color='lime')
    axs[1].plot(range(1, num_epochs+1), train_ssim, label='Train SSIM', color='purple')
    axs[1].plot(range(1, num_epochs+1), val_ssim, label='Val SSIM', color='gray')
    axs[1].set_title("PSNR y SSIM")
    axs[1].set_xlabel("Épocas")
    axs[1].set_ylabel("PSNR / SSIM")
    axs[1].legend()

    # Añadimos un título general
    plt.suptitle("Métricas de entrenamiento y validación de U-Net")

    plt.tight_layout()
    plt.show()


  def eval_model(self, data_loader, criterion):

    # Cambiamos al modo de evaluación
    self.eval()

    # Inicializamos valores
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    # Desactivamos el cálculo de gradientes
    with torch.no_grad():

      # Usamos batch de muestras
      for batch in data_loader:
        images = batch["image"].to(self.device)
        outputs = self.forward(images)

        loss = criterion(outputs, images)
        total_loss += loss.item()

        psnr = peak_signal_noise_ratio(images.detach().cpu().numpy(), outputs.detach().cpu().numpy(), data_range=1.0)
        ssim = structural_similarity(images.detach().cpu().numpy(), outputs.detach().cpu().numpy(), channel_axis=0, win_size=3)
        total_psnr += psnr
        total_ssim += ssim

        if self.device == 'cuda':
          torch.cuda.empty_cache()

        del images, outputs

    avg_loss = total_loss / len(data_loader)
    avg_psnr = total_psnr / len(data_loader)
    avg_ssim = total_ssim / len(data_loader)

    return avg_loss, avg_psnr, avg_ssim

  def get_latent_representation(self, slide):
    """
    Obtiene la representación latente de un parche.

    Parámetros:
        slide : Parche de entrada en formato (C, H, W), donde C es el número de canales,
                H es la altura y W es el ancho.

    Retorna:
        torch.Tensor: Representación latente del parche
    """

    # Convertimos a tensor
    slide_array = np.array(slide)
    slide_tensor = torch.from_numpy(slide_array).permute(2, 0, 1).unsqueeze(0).float()

    # Movemos el tensor al mismo dispositivo
    slide_tensor = slide_tensor.to(self.device)

    # Cambiamos al modo de evaluación
    self.eval()

    # Desactivamos el cálculo de gradientes durante la propagación
    with torch.no_grad():
      _, _, _, _, latent_representation = self.contractive_path(slide_tensor)

    # Volvemos al modo de entrenamiento
    self.train()

    del slide_array, slide_tensor

    return latent_representation.cpu().squeeze()

  def get_all_latent_representation(self, dataset, batch_size=32, num_workers=4):
    """
    Obtiene las representaciones latentes de todos parche.

    Parámetros:
        dataset : Conjunto de datos con todas las imágenes

    Retorna:
        dict: Un diccionario que mapea nombres de parches a sus representaciones latentes como torch.Tensor.
    """

    # Uso de un DataLoader con batch_size y num_workers
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    # Lista de representaciones latentes
    latent_representations = {}

    # Cambiamos al modo de evaluación
    self.eval()

    # Desactivamos el cálculo de gradientes durante la propagación
    with torch.no_grad():
      # Nos desplazamos por batch de muestras
      for batch in tqdm(dataloader, desc="Procesando lotes ...", unit="batch"):
        batch_names = batch["name"]
        batch_images = batch["image"].to(self.device)
        _, _, _, _, batch_representation = self.contractive_path(batch_images)

        for name, representation in zip(batch_names, batch_representation):
          latent_representations[name] = representation.cpu().squeeze()

        if self.device == 'cuda':
          torch.cuda.empty_cache()

        del batch, batch_representation, batch_names, batch_images

    # Volvemos al modo de entrenamiento
    self.train()

    return latent_representations
    