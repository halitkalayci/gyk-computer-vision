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

# CNN
# Sequential => Katmanları sırayla eklediğimiz bir yapı.
model = tf.keras.models.Sequential(
    [
        # Kernel_Size => 3x3 (Görüntü üzerinde 3x3 filtreler ile dolaş.)
        tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)), # Görüntü üzerindeki ilk tarama.
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)), # 2x2 boyutundaki alanların sadece en yüksek değerini alarak küçültür. -> İlk tarafa sonrası bilgiyi özetle.
        tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu"), # Daha karmaşık özellikleri bulmak için.
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)), # Özetleme
        tf.keras.layers.Flatten(), # ND'den 1D'ye çevirir.
        tf.keras.layers.Dense(128, activation="relu"), # 128 nöronlu bir katman. Nöron => Karar mekanizması
        tf.keras.layers.Dense(10, activation="softmax") # 10 nöronlu bir katman. => 10 rakam var.
        # Softmax => 10 rakam arasında tahmin güven skoru en yüksek olanı seçer.
        # 10 tahmin yapar, en yüksek değer sahibi seçer.
    ]
)

model.summary()

# Modelin genel parametlerini belirleyip eğitmeye hazır hale getirir.
# Optimizer => Modelin eğitimi sırasında kullanılacak optimizasyon algoritması.
# Loss => Modelin eğitimi sırasında kullanılacak loss fonksiyonu.
# Metrics => Modelin eğitimi sırasında kullanılacak metrikler.
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Epoch => Veriyi baştan sona kaç kere göreceğiz?
model.fit(X_train, y_train, epochs=5, validation_split=0.2)

model.save("model.keras")