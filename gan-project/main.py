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

# Generator-Discriminator

device = "cuda" if torch.cuda.is_available() else "cpu"

G = generator.Generator().to(device)

G.eval() # Modeli inference moduna al. -> Üretim moduna.

#16 farklı hayali gürültü vektörü oluştur.
noise = torch.randn(16, 100).to(device)

with torch.no_grad():
    fake_images = G(noise)

from torchvision import utils

grid = utils.make_grid(fake_images, nrow=4, normalize=True)
npimg = grid.cpu().numpy().transpose((1,2,0))

import os
import matplotlib.pyplot as plt

os.makedirs("generated_fake_imgs", exist_ok=True)
plt.imshow(npimg)
plt.title("Generatorden rastgele görseller.")
plt.axis("off")
plt.savefig("generated_fake_imgs/fake_images.png")
plt.show()

#Discrimantoru yazalım.
#20.30