import tensorflow as tf
import numpy as np
from PIL import Image

# Boilerplate Code -> Basmakalıp
interpreter = tf.lite.Interpreter(model_path="plate_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = Image.open("data/images/1.jpg").convert("RGB").resize((640,640))
img_array = np.array(image, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=0)

interpreter.set_tensor(input_details[0]['index'], img_array)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print(output_data)
