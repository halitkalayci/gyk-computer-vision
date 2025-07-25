from torch import nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__() #Pytorch base NN'ü kendi kodlarını çalıştır.
        self.net = nn.Sequential(
            nn.Linear(100,256),
            nn.ReLU(True),
            nn.Linear(256,512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh() # Çıktıyı -1 ile 1 aralığına getir.
        )
    
    #İleri besleme
    def forward(self, x):
        return self.net(x).view(-1, 1, 28, 28) # 28x28 lik görsel üretiyoruz. Katmandan geçtikten sonra.