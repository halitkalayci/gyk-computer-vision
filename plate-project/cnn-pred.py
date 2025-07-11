import tensorflow as tf

model = tf.keras.models.load_model("plate_detection_cnn.keras")

import numpy as np

X = np.load("X.npy")
y = np.load("y.npy")

y_coords = y[:,1:5]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y_coords,test_size=0.2,random_state=42)


test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.6f}")
print(f"Test MAE: {test_mae:.6f}")

predictions = model.predict(X_test[:5])

print(predictions)

for i in range(5):
    print(f"Örnek {i+1}:")
    print(f"Gerçek koordinatlar: {y_test[i]}")
    print(f"Tahmin koordinatlar: {predictions[i]}")
    print(f"Fark: {np.abs(y_test[i] - predictions[i])}")

from matplotlib import pyplot as plt

plt.figure(figsize=(15,10))

for i in range(5):
    plt.subplot(2,3,i+1)

    img = X_test[i]
    plt.imshow(img)

    real_coords = y_test[i]
    x_center, y_center, width, height = real_coords

    x1 = (x_center - width/2)* 640
    y1 = (y_center - height/2)* 640
    x2 = (x_center + width/2)* 640
    y2 = (y_center + height/2)* 640

    plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],color="red",linewidth=2)

    predicted_coords = predictions[i]
    x_center, y_center, width, height = predicted_coords

    x1 = (x_center - width/2)* 640
    y1 = (y_center - height/2)* 640
    x2 = (x_center + width/2)* 640
    y2 = (y_center + height/2)* 640

    plt.plot([x1,x2,x2,x1,x1],[y1,y1,y2,y2,y1],color="blue",linewidth=2)


plt.tight_layout()
plt.show()
