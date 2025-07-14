import tensorflow as tf

model = tf.keras.models.load_model("plate_detection_cnn.keras")

#model.summary()

# Sayılarla çalışır -> Float32 -> Float16

#model = tf.keras.models.clone_model(model)
#model.set_weights([w.astype("float16") for w in model.get_weights()])

# 

#model.save_weights("plate_detection_cnn2.weights.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("plate_model.tflite", "wb") as f:
    f.write(tflite_model)