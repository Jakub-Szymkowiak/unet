import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import random_split, DataLoader

from pathlib import Path

from data_loader import SegmentationDataset
from unet import U_Net

img_loc = Path("/content/drive/MyDrive/projects/unet/data/img/")
seg_loc = Path("/content/drive/MyDrive/projects/unet/data/seg/")

dataset = SegmentationDataset(img_loc, seg_loc)
train_set, test_set = random_split(dataset, [25, 10])

train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=1, shuffle=True)

model = U_Net(3,1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99)

def testLoss():
  model.eval()
  acc = 0

  with torch.no_grad():
    for data in test_loader:
      img, seg = data
      img, seg = img.to(device), seg.to(device)
      
      outputs = model(img)
      acc = acc + loss_fn(outputs, seg)
    
  return acc / len(test_loader)

def saveModel():
  save_path = "/content/drive/MyDrive/projects/unet/unet_trained.pth"
  torch.save(model.state_dict(), save_path)

def train(no_epochs):
  model.train()

  best_loss = 0

  train_hist = {
      "running_train": [],
      "test": []
  }

  for epoch in range(no_epochs):
    run_loss = 0

    for i, (img, seg) in enumerate(train_loader, 0):
      img, seg = img.to(device), seg.to(device)
      
      outputs = model(img)
      loss = loss_fn(outputs, seg)  
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      run_loss += loss.item()
      
    test_loss = testLoss()

    print(f"Epoch {epoch+1}. Running training loss: {run_loss / len(train_loader)}; test loss: {test_loss}.")  

    if test_loss > best_loss:
      best_loss = test_loss
      saveModel()

    train_hist["running_train"].append(run_loss)
    train_hist["test"].append(test_loss)

    run_loss = 0
  
  return train_hist

no_epochs = 25
train_hist = train(no_epochs)