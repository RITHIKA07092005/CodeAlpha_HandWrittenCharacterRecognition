# CodeAlpha_HandWrittenCharacterRecognition
Handwritten character recognition using CNN on EMNIST Letters dataset — CodeAlpha ML Internship Task 3
# ✍️ Handwritten Character Recognition | CodeAlpha ML Internship — Task 3

A deep learning project that recognizes handwritten English alphabets (A–Z) 
using a Convolutional Neural Network (CNN) trained on the EMNIST Letters dataset.

---

## 📌 Project Overview

This project is built as part of the **CodeAlpha Machine Learning Internship**.  
The model takes a 28×28 grayscale image of a handwritten letter and predicts 
which alphabet (A to Z) it belongs to — achieving ~90% test accuracy.

---

## 🗂️ Dataset

- **Name:** EMNIST Letters  
- **Source:** [Kaggle — crawford/emnist](https://www.kaggle.com/datasets/crawford/emnist)  
- **Size:** 145,600 images (28×28 pixels, grayscale)  
- **Classes:** 26 (A–Z uppercase letters)

---

## 🧠 Model Architecture

- 3 Convolutional Blocks (Conv2D + BatchNormalization + MaxPooling + Dropout)
- Fully Connected Dense layers
- Output: Softmax over 26 classes
- Optimizer: Adam | Loss: Categorical Crossentropy

---

## 📊 Results

| Metric        | Score     |
|---------------|-----------|
| Test Accuracy | ~90%      |
| Loss Function | Cat. CE   |
| Parameters    | ~1.2M     |

---

## 🛠️ Tech Stack

- Python 3.x
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- Scikit-learn
- Google Colab (GPU)

---

## 🚀 How to Run

1. Open `handwritten_character_recognition.py` in **Google Colab**
2. Enable GPU: Runtime → Change Runtime Type → GPU
3. Upload your `kaggle.json` API key when prompted
4. Run all cells from top to bottom

---

## 📁 Project Structure

CodeAlpha_HandwrittenCharacterRecognition/
│
├── handwritten_character_recognition.py   # Main code (all cells)
├── samples.png                            # Sample dataset images
├── training_curves.png                    # Accuracy & Loss plots
├── confusion_matrix.png                   # Confusion matrix heatmap
├── emnist_cnn_final.keras                 # Saved trained model
└── README.md                              # Project documentation

---

## 🏢 Internship

**Organization:** CodeAlpha  
**Domain:** Machine Learning  
**Task:** Task 3 — Handwritten Character Recognition
