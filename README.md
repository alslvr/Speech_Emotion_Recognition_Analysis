# Speech Emotion Recognition & Perception Pipeline

An end-to-end machine learning perception pipeline designed to classify human emotional states from raw acoustic audio signals.

---

## Overview

This project implements a complete audio processing and classification workflow in Python[cite: 3]. It extracts high-dimensional spectral and acoustic features from speech recordings, handles rigorous preprocessing to prevent data leakage and imbalance, and benchmarks multiple supervised learning architectures to evaluate decision boundary complexity and emotion separability.

---

## Features

- **Acoustic Feature Extraction:** Uses `librosa` to compute Short-Time Fourier Transforms (STFT), extracting 2,370+ dimensional spectral feature representations (MFCCs, chroma, and spectral contrast) from raw audio streams.
- **Robust Preprocessing Pipeline:** Integrates `StandardScaler` feature normalization, missing-value imputation, and stratified dataset partitioning across 50,000+ samples to eliminate scale bias and class imbalance skew.
- **Model Benchmarking:** Implements and compares Multi-Layer Perceptrons (MLP with 256–128 hidden nodes), Random Forest ensembles, and linear baseline classifiers.
- **Performance Diagnostics:** Evaluates model generalization using confusion matrices, precision-recall breakdowns, and macro F1-score metrics.

---

## Technical Highlights

- **Acoustic Dimension:** 2,370+ extracted spectral feature components.
- **Dataset Scale:** 50,000+ processed samples across multi-speaker audio datasets.
- **Benchmark Results:** Achieved **~42% top-1 test accuracy** and a **0.63 F1-score** on target emotion classes using ensemble models.

---

## Tech Stack & Libraries

- **Language:** Python
- **Audio & Signal Processing:** Librosa, NumPy
- **Machine Learning & Modeling:** Scikit-Learn
- **Data Analysis & Visualization:** Pandas, Matplotlib, Seaborn

---

## Repository Structure

```text
├── Speech_Emotion_Recognition_Analysis.ipynb   # Main Jupyter/Colab notebook containing feature extraction & modeling[cite: 3]
└── README.md                                    # Project documentation
