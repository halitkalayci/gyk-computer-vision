# Transfer Learning => Eğitilmiş bir modeli kullanarak yeni bir model oluşturmak.

# imageları process
# label dosyalarını oku.
# imagelardan ve labellardan veri seti oluştur.

import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")

print(X[1])
print(y.shape)

# X,y train_test_split
# Tensorflow sequential modeli oluştur.
# modeli compile et.

# Activation fonksiyonları nelerdir, ne için kullanılır?
# Loss function nedir, ne için kullanılır?
# Optimizer nedir, ne için kullanılır?

# Yazı hazırlamak.

# Sonraki derse tam kurulumlarla gelmek. Python 3.8-3.11 
#Tensorflow kurulu.