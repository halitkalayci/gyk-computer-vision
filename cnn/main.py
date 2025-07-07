import tensorflow as tf
import matplotlib.pyplot as plt

# MNIST 0-9 arası el yazısı rakamlar.

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizasyon => RGB kanallarının 0-255 aralığındansa 0-1 aralığına çekilmesi.
X_train = X_train / 255
X_test = X_test / 255 # 0-1
#

# CNN'in input formatı => (örnek sayısı, genişlik, yükseklik, kanal sayısı)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)  # Sebebini CNN'e geçtiğimizde konuşacağız.
#

plt.figure(figsize=(10,10))

for i in range(10):
    plt.subplot(10, 10, i+1)
    plt.imshow(X_train[i].reshape(28,28), cmap="gray")
    plt.axis("off")
plt.suptitle("İLK 10 Görüntü")
plt.show()
