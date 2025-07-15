# evaluate.py

from tensorflow.keras.models import load_model
from data_loader import load_mnist_data
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate_model():
    # Load saved model
    model = load_model("saved_model/mnist_cnn_model.h5")

    # Load test data
    _, _, x_test, y_test = load_mnist_data()

    # Predict
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification report
    print("Classification Report:\n")
    print(classification_report(y_true, y_pred_classes))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Save confusion matrix plot
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png")
    plt.show()

    print("Evaluation complete. Confusion matrix saved in 'results/'.")

if __name__ == "__main__":
    evaluate_model()
