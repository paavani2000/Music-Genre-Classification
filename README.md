# 🎵 Music Genre Classification using Deep Learning

This project aims to classify music tracks into genres using Convolutional Neural Networks (CNNs). Audio features are extracted using Librosa and the models are trained and evaluated in Python using TensorFlow/Keras.

---

## 📌 Project Overview

Music genre classification is a crucial task in music recommendation, categorization, and music therapy. This project uses deep learning techniques to automatically detect the genre of a given audio clip.

---

## 🎯 Objectives

- Automatically classify songs into 10 music genres.
- Extract meaningful audio features using signal processing techniques.
- Train a CNN model on spectrogram representations of songs.
- Evaluate and improve classification accuracy.

---

## 🛠️ Tech Stack

| Area              | Tools / Libraries                     |
|-------------------|----------------------------------------|
| Language          | Python                                 |
| Audio Processing  | Librosa                                |
| Deep Learning     | TensorFlow, Keras                      |
| Visualization     | Matplotlib, Seaborn                    |
| IDE               | Google Colab, Jupyter Notebook         |
| Dataset           | GTZAN Dataset                          |

---

## 🎧 Genres Covered

1. Pop  
2. Rock  
3. Indie Rock  
4. EDM  
5. Jazz  
6. Country  
7. Hip Hop & Rap  
8. Classical Music  
9. Latin Music  
10. K-Pop

---

## 📁 Dataset

- **GTZAN Dataset**
  - 1000 audio tracks
  - 30 seconds each
  - 10 balanced genre classes

> Download link: [GTZAN Dataset on Marsyas](http://marsyas.info/downloads/datasets.html)

---

## 📊 Features Extracted

- **MFCC (Mel-Frequency Cepstral Coefficients)**
- **Mel Spectrograms**

Extracted using the [Librosa](https://librosa.org/) Python library.

---

## 🧠 Model Architecture

- Input: 2D Mel Spectrogram or MFCC image
- Convolutional layers with ReLU activation
- MaxPooling layers
- Dense layers for classification
- Softmax output layer (10 classes)

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.7+
- TensorFlow 2.x
- Librosa
- Matplotlib
- Numpy
- Scikit-learn
- Google Colab or Jupyter Notebook

### 🐍 Installation

```bash
pip install tensorflow librosa matplotlib numpy scikit-learn
