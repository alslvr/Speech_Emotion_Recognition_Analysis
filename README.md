# Speech Emotion Recognition & Perception Pipeline

An end-to-end machine learning perception pipeline designed to classify human emotional states from raw acoustic audio signals.

---

## Overview

This project implements a complete audio processing and classification workflow in Python[cite: 3]. It extracts high-dimensional spectral and acoustic features from speech recordings, handles rigorous preprocessing to prevent data leakage and imbalance, and benchmarks multiple supervised learning architectures to evaluate decision boundary complexity and emotion separability[cite: 3].

---

## Features

- **Acoustic Feature Extraction:** Uses `librosa` to compute Short-Time Fourier Transforms (STFT), extracting 2,370+ dimensional spectral feature representations (MFCCs, chroma, and spectral contrast) from raw audio streams[cite: 3].
- **Robust Preprocessing Pipeline:** Integrates `StandardScaler` feature normalization, missing-value imputation, and stratified dataset partitioning across 50,000+ samples to eliminate scale bias and class imbalance skew[cite: 3].
- **Model Benchmarking:** Implements and compares Multi-Layer Perceptrons (MLP with 256–128 hidden nodes), Random Forest ensembles, and linear baseline classifiers[cite: 3].
- **Performance Diagnostics:** Evaluates model generalization using confusion matrices, precision-recall breakdowns, and macro F1-score metrics[cite: 3].

---

## Technical Highlights

- **Acoustic Dimension:** 2,370+ extracted spectral feature components[cite: 3].
- **Dataset Scale:** 50,000+ processed samples across multi-speaker audio datasets[cite: 3].
- **Benchmark Results:** Achieved **~42% top-1 test accuracy** and a **0.63 F1-score** on target emotion classes using ensemble models[cite: 3].

---

## Tech Stack & Libraries

- **Language:** Python[cite: 3]
- **Audio & Signal Processing:** Librosa, NumPy[cite: 3]
- **Machine Learning & Modeling:** Scikit-Learn[cite: 3]
- **Data Analysis & Visualization:** Pandas, Matplotlib, Seaborn

---

## Repository Structure

```text
├── Speech_Emotion_Recognition_Analysis.ipynb   # Main Jupyter/Colab notebook containing feature extraction & modeling[cite: 3]
└── README.md                                    # Project documentation
