# Transfer Learning => Eğitilmiş bir modeli kullanarak yeni bir model oluşturmak.

# imageları process
# label dosyalarını oku.
# imagelardan ve labellardan veri seti oluştur.

import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")


# Eğer tek sınıf çalışıyorsa
y_coords = y[:,1:5]
#

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y_coords,test_size=0.2,random_state=42)

# CNN Modeli kurmak
#
import tensorflow as tf
model = tf.keras.models.Sequential([
    # Konvolüsyon katmanı
    # ReLU => 0-1 arasında değerler alır. Negatif değerleri 0 yapar.
    tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(640,640,3)), # RGB => 3 renk kanalı
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)), # 2x2 boyutundaki alanların sadece en yüksek değerini alarak küçültür. -> İlk tarafa sonrası bilgiyi özetle.

    tf.keras.layers.Conv2D(64,kernel_size=(3,3),activation="relu",input_shape=(640,640,3)), # RGB => 3 renk kanalı
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)), # 2x2 boyutundaki alanların sadece en yüksek değerini alarak küçültür. -> İlk tarafa sonrası bilgiyi özetle.

    tf.keras.layers.Conv2D(128,kernel_size=(3,3),activation="relu",input_shape=(640,640,3)), # RGB => 3 renk kanalı
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)), # 2x2 boyutundaki alanların sadece en yüksek değerini alarak küçültür. -> İlk tarafa sonrası bilgiyi özetle.

    tf.keras.layers.Conv2D(256,kernel_size=(3,3),activation="relu",input_shape=(640,640,3)), # RGB => 3 renk kanalı
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)), # 2x2 boyutundaki alanların sadece en yüksek değerini alarak küçültür. -> İlk tarafa sonrası bilgiyi özetle.

    tf.keras.layers.Flatten(), 

    # 4 conv katmanı sonrası özellik haritasını düzleştirdik.

    tf.keras.layers.Dense(1024, activation="relu"), # 1024 nöronlu bir katman.
    tf.keras.layers.Dropout(0.5), # Overfitting'i önlemek için. # %50 oranında nöronların kapatılması.

    tf.keras.layers.Dense(512, activation="relu"), # 512 nöronlu bir katman.
    tf.keras.layers.Dropout(0.5), # Overfitting'i önlemek için. # %50 oranında nöronların kapatılması.

    tf.keras.layers.Dense(256, activation="relu"), # 256 nöronlu bir katman.
    tf.keras.layers.Dropout(0.5), # Overfitting'i önlemek için. # %50 oranında nöronların kapatılması.

    tf.keras.layers.Dense(4, activation="linear") # 4 nöronlu bir katman. Linear aktivasyon çünkü => Regresyon için.
])
# 19:45 => modeli compile edip eğitelim.

# 0-9 arası rakam
# 50.000 kişini farklı çizimi var. => 
# 1000 kişinin farklı çizimi var => 3 