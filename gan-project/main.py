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



G = generator.Generator().to(device)

G.eval() # Modeli inference moduna al. -> Üretim moduna.

#16 farklı hayali gürültü vektörü oluştur.
noise = torch.randn(16, 100).to(device)


# Discriminator -> Generatorün oluşturduğu yapıları eleştirmek.
