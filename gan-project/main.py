from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import generator
import torch

transform =  transforms.Compose([
    transforms.ToTensor(), # PyTorch tensörüne çevir.
    transforms.Normalize((0.5,), (0.5,)) # Normalize et.
])

dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)

# verileri küçük gruplar halinde hazırla (Batch)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)  
device = "cuda" if torch.cuda.is_available() else "cpu"

# Generator-Discriminator

import discriminator
from torch import nn, optim

G = generator.Generator().to(device)
D = discriminator.Discriminator().to(device)

# Tensorflowdan farklı olarak: loss fonk. ve optimizerları kendimiz tanımlarız.

# Binary CrossEntropy
loss_fn = nn.BCELoss()

optim_G = optim.Adam(G.parameters, lr=0.0002)
optim_D = optim.Adam(D.parameters, lr=0.0002)

#

#16 farklı hayali gürültü vektörü oluştur.
noise = torch.randn(16, 100).to(device)

