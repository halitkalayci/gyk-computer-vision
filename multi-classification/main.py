import tensorflow as tf


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# MobileNetV2 + Kendi Model
# MobileNetV2 -> 224x224x3
IMG_SIZE = 224
# 50k veriyi aynı anda işlemek RAM dostu değil.
# batching ile işlemek daha iyi.
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE # Tensorflowun kendi CPU optimizasyonu

def preprocess_image(img,label):
    image = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    image = tf.cast(image, tf.float32) / 255
    return image,label

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) # Bu kısımda verisetini TensorDataset'e çeviriyoruz.
train_ds = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE) # Bu kısımda her bir veri için preprocess_image fonksiyonunu çağırıyoruz.
train_ds = train_ds.shuffle(10000) # Burada veri setini rastgele karıştırıyoruz.
train_ds = train_ds.batch(BATCH_SIZE) # Burada batchlere böl. 32/32/32
train_ds = train_ds.prefetch(AUTOTUNE) # Pre-Fetch ile X adet veriyi önden çekiyoruz.


test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
# YANLIŞ!
#X_train = preprocess_image(x_train)
#X_test = preprocess_image(x_test)

#y_train = y_train.flatten()
#y_test = y_test.flatten()

# BaseModel tanımla
# Kendi modelimizi tanımla
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet',
)
base_model.trainable = False # BaseModel'in trainable özelliğini False yapıyoruz. -> Freeze

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(), #Çıktıyı flatten edicem.
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_ds, epochs=5, validation_data=test_ds)

model.save("multi_classification_model.h5")