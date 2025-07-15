# predict_custom.py

import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = load_model("saved_model/mnist_cnn_model.h5")

# Load and preprocess image
img_path = "test_digit.png"  # <-- replace with your file name
img = Image.open(img_path).convert('L')  # convert to grayscale
img = img.resize((28, 28))               # resize to 28x28
img = np.array(img)

# Invert if background is white and digit is dark (optional)
if np.mean(img) > 127:
    img = 255 - img

img = img.astype('float32') / 255.0
img = img.reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

# Show image and prediction
plt.imshow(img.reshape(28, 28), cmap='gray')
plt.title(f"Predicted Digit: {predicted_class}")
plt.axis('off')
plt.show()
