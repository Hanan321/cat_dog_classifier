import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#1. Device
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")
print("Using device: ", device)

#2.Image transform
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

#3. Load dataset
train_dataset = datasets.ImageFolder(root="data/train",transform=transform)
val_dataset =datasets.ImageFolder(root="data/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
 
print("Clasess: ", train_dataset.classes)

#4. Define model
model = nn.Sequential(
  nn.Flatten(),
  nn.Linear(128 * 128 * 3, 128),
  nn.ReLU(),
  nn.Linear(128,2)
  ).to(device)

