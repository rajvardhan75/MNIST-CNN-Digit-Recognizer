# predict_sample.py

import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_loader import load_mnist_data

# Load model and data
model = load_model("saved_model/mnist_cnn_model.h5")
x_train, y_train, x_test, y_test = load_mnist_data()

# Pick random samples to test
num_samples = 10
indices = random.sample(range(len(x_test)), num_samples)

for i in indices:
    image = x_test[i]
    label = np.argmax(y_test[i])

    # Predict
    prediction = model.predict(np.expand_dims(image, axis=0))
    predicted_label = np.argmax(prediction)

    # Show result
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Actual: {label}, Predicted: {predicted_label}")
    plt.axis('off')
    plt.show()
