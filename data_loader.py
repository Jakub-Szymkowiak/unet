import torch

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import os

from pathlib import Path
from PIL import Image

from numpy.random import randint

max_seed = int(1e5)

preprocess_img = transforms.Compose([
  transforms.Grayscale(3),
  transforms.Resize([572, 572]),
  transforms.ToTensor()
])

preprocess_seg = transforms.Compose([
  # transforms.Grayscale(1),
  transforms.Resize([388, 388]),
  transforms.ToTensor()
])

augmentation = transforms.RandomApply([
  # transforms.RandomRotation(180),
  transforms.RandomVerticalFlip(0.1),
  transforms.RandomHorizontalFlip(0.1)
])

class SegmentationDataset(Dataset):
  def __init__(self, img_loc, seg_loc):
    super().__init__()
    self.img_loc = img_loc
    self.seg_loc = seg_loc

    self.img_paths = sorted([path for path in os.listdir(img_loc)])
    self.seg_paths = sorted([path for path in os.listdir(seg_loc)])
    assert len(self.img_paths) == len(self.seg_paths)

    self.preprocess_img = preprocess_img
    self.preprocess_seg = preprocess_seg

    self.augment = augmentation

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, id):
    img_path = self.img_paths[id]
    img_raw = Image.open(self.img_loc / img_path)
    img = preprocess_img(img_raw)

    seg_path = self.seg_paths[id]
    seg_raw = Image.open(self.seg_loc / seg_path)
    seg = preprocess_seg(seg_raw)

    #apply same transform
    torch_seed = randint(max_seed)

    torch.random.manual_seed(torch_seed)
    img = self.augment(img)

    torch.random.manual_seed(torch_seed)
    seg = self.augment(seg)

    return img, seg