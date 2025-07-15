# test_data_loader.py

from data_loader import load_mnist_data
import matplotlib.pyplot as plt

x_train, y_train, x_test, y_test = load_mnist_data()

print("Training data shape:", x_train.shape)
print("Test data shape:", x_test.shape)
print("Example label (one-hot):", y_train[0])

# Plot 1 sample image to check
plt.imshow(x_train[0].reshape(28, 28), cmap='gray')
plt.title("Sample Digit")
plt.axis('off')
plt.show()
