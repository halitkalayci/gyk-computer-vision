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

# Binary CrossEntropy Loss
loss_fn = nn.BCELoss()
#
optim_G = optim.Adam(G.parameters(), lr=0.0002) # learning-rate
optim_D = optim.Adam(D.parameters(), lr=0.0002)
#

#16 farklı hayali gürültü vektörü oluştur.
fixed_noise = torch.randn(16, 100).to(device)

epochs = 50

import torch
from torchvision import utils
import os
import matplotlib.pyplot as plt

def save_images(images, epoch):
    os.makedirs("generated_fake_imgs", exist_ok=True)
    grid = utils.make_grid(images, nrow=4, normalize=True)
    npimg = grid.cpu().numpy().transpose((1,2,0))
    plt.imshow(npimg)
    plt.axis("off")
    plt.savefig(f"generated_fake_imgs/epoch_{epoch+1}.png")
    plt.close()

for epoch in range(epochs):
    for real, _ in dataloader:
        real = real.to(device)
        batch_size = real.size(0)

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # 1- Discriminator Eğitimi
        noise = torch.randn(batch_size, 100).to(device) # rastgele gürültü üret.
        fake_images = G(noise)

        # tekil bir eğitim adımı
        D_real = D(real)
        D_fake = D(fake_images.detach())
        D_loss = loss_fn(D_real, real_labels) + loss_fn(D_fake, fake_labels)

        optim_D.zero_grad() #Gradleri sıfırlayalım.
        D_loss.backward()
        optim_D.step()
        #

        # 2- Generator eğitimi
        output = D(fake_images)
        G_loss = loss_fn(output, real_labels)

        optim_G.zero_grad()
        G_loss.backward()
        optim_G.step()
    
    # Her epoch sonunda örnek çizimleri kaydet.
    with torch.no_grad():
        fake = G(fixed_noise)
        save_images(fake, epoch)
    print(f"Epoch {epoch+1} D Loss: {D_loss.item()} G Loss: {G_loss.item()}")