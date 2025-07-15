# 🧠 MNIST CNN Digit Recognizer

A deep learning project that uses a Convolutional Neural Network (CNN) to classify handwritten digits from the MNIST dataset. Includes an interactive interface built with Gradio to test the model in real-time by drawing digits on a canvas.

---

## 📌 Project Overview

This project demonstrates a complete workflow for building, training, evaluating, and deploying a CNN for digit classification. It uses the MNIST dataset and showcases:

- Data preprocessing
- CNN model architecture using TensorFlow/Keras
- Evaluation using accuracy and confusion matrix
- Real-time testing with a Gradio web interface

---

## 📁 Project Structure

CNN/
├── data_loader.py # Loads and preprocesses MNIST data
├── model.py # Defines the CNN model architecture
├── train.py # Trains the model and saves it
├── evaluate.py # Evaluates the model using test data
├── requirements.txt # All Python dependencies
└── saved_model/ # Trained model weights (HDF5 or SavedModel format)

## 🧪 Model Architecture

- Conv2D → ReLU → MaxPooling
- Conv2D → ReLU → MaxPooling
- Flatten → Dense(128) → Dropout(0.5)
- Output Dense(10) with Softmax

Compiled with:
- Optimizer: Adam  
- Loss: Categorical Crossentropy  
- Metric: Accuracy  

---

## 📊 Evaluation

- **Accuracy:** >99% on test data  
- **Confusion Matrix:** Low misclassification rates  
- **Classification Report:** High precision and recall across all digits  

---
