# train.py

from data_loader import load_mnist_data
from cnn_model import build_cnn_model
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os

def train_model():
    # Load preprocessed data
    x_train, y_train, x_test, y_test = load_mnist_data()

    # Build the CNN
    model = build_cnn_model()

    # Create a directory for saved models
    os.makedirs("saved_model", exist_ok=True)

    # Save best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        filepath="saved_model/mnist_cnn_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )

    # Train the model
    history = model.fit(
        x_train, y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=128,
        callbacks=[checkpoint],
        verbose=1
    )

    # Plot training & validation accuracy/loss
    os.makedirs("results", exist_ok=True)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/accuracy_loss_plot.png")
    plt.show()

    print("Training complete. Model saved to 'saved_model/mnist_cnn_model.h5'")

if __name__ == "__main__":
    train_model()
