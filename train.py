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

