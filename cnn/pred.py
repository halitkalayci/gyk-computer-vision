import tensorflow as tf

model = tf.keras.models.load_model("model.h5")

model.summary()

from tensorflow.keras.utils import plot_model

plot_model(model, to_file="model.png", show_shapes=True, show_layer_names=True)


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalizasyon => RGB kanallarının 0-255 aralığındansa 0-1 aralığına çekilmesi.
X_train = X_train / 255
X_test = X_test / 255 # 0-1
#

# CNN'in input formatı => (örnek sayısı, genişlik, yükseklik, kanal sayısı)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)  # Sebebini CNN'e geçtiğimizde konuşacağız.


test_loss, test_acc = model.evaluate(X_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")


# Kendi çizdiğiniz bir rakamı tahmin ettirelim.
# Bu kodları yazalım.