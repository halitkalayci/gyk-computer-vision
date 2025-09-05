from torch import nn

# Çıktı -> 0-1 arası bir class (gerçek-sahte)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Kapasite (Modelin Öğrenme Gücü)
        self.net = nn.Sequential(
            nn.Flatten(), #Görseli düzleştir.
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2), # Gradient Vanishing
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() # 0-1 classification
        )
    def forward(self, x):
        return self.net(x)

        # 784 -> 512 -> 256 -> 1

        # CNN + bir çok katman 2048+