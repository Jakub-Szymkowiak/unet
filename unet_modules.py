import torch
import torch.nn as nn

from torchvision import transforms

class RepeatedConv(nn.Module):
  def __init__(self, in_size, out_size, upwards=False):
    super().__init__()

    self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3)
    self.bnor1 = nn.BatchNorm2d(out_size)
    self.relu1 = nn.ReLU()

    self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3)
    self.bnor2 = nn.BatchNorm2d(out_size)
    self.relu2 = nn.ReLU()

  def forward(self, x):
    x = self.conv1(x)
    x = self.bnor1(x)
    x = self.relu1(x)

    x = self.conv2(x)
    x = self.bnor2(x)
    x = self.relu2(x)
    
    return x

class Input(nn.Module):
  def __init__(self, in_size):
    super().__init__()

    self.repconv = RepeatedConv(in_size, 64)

  def forward(self, x):
    x = self.repconv(x)
    
    return x

class Contracting(nn.Module):
  def __init__(self, in_size, out_size):
    super().__init__()

    self.maxpool = nn.MaxPool2d(2)
    self.repconv = RepeatedConv(in_size, out_size)

  def forward(self, x):
    x = self.maxpool(x)
    x = self.repconv(x)

    return x

class Expansive(nn.Module):
  def __init__(self, in_size, out_size):
    super().__init__()

    self.upconv   = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
    self.repconv  = RepeatedConv(in_size, out_size)

  def forward(self, x_up, x_copy):
    x_up = self.upconv(x_up)

    h, w = x_up.shape[2], x_up.shape[3]
    x_copy = transforms.CenterCrop([h,w])(x_copy)

    x = torch.cat([x_up, x_copy], dim=1)

    x = self.repconv(x)
    
    return x

class Output(nn.Module):
  def __init__(self, out_size):
    super().__init__()

    self.conv = nn.Conv2d(64, out_size, kernel_size=1)

  def forward(self, x):
    x = self.conv(x)

    return x