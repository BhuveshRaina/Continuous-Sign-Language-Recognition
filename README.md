# Continuous Sign Language (CSL) Sentence Recognition Model

## Overview
Continuous Sign Language (CSL) recognition is a complex challenge bridging Computer Vision and Natural Language Processing (NLP). Unlike isolated sign recognition, continuous signing involves fluid transitions, co-articulation, and varying speeds. 

This project implements an end-to-end deep learning pipeline capable of transcribing continuous sign language video sequences directly into grammatically coherent text sentences across 108 canonical target classes.

## Tech Stack
* **Language:** Python
* **Deep Learning:** PyTorch
* **Computer Vision:** MediaPipe Holistic
* **Architecture:** Multi-Layer Perceptron (MLP), Bidirectional LSTM (Bi-LSTM), Temporal Attention

## Methodology
### 1. Data Preparation & Feature Extraction
* Avoided memory-intensive raw video processing by utilizing **MediaPipe Holistic** to extract skeletal landmarks.
* Captured 33 Pose, 21 Left Hand, and 21 Right Hand landmarks, flattening them into a **225-dimensional feature vector** per frame.
* Standardized sequences to a maximum of 60 frames via padding and truncation.

### 2. Label Normalization (NLP)
* Applied a robust text preprocessing pipeline to raw labels, merging casing inconsistencies, varying punctuation, and contractions.
* Consolidated 197 noisy label variations into **108 canonical classes**, achieving a 45.18% reduction in redundancy.

### 3. Model Architecture
* **Feature Extractor:** An MLP (with LayerNorm and GELU) projecting raw 225-dimensional keypoints into a 256-dimensional space.
* **Sequence Modeling:** A **Bi-LSTM** captures the contextual flow of time-series gesture data, outputting a 512-dimensional hidden state.
* **Temporal Attention:** Assigns learned scalar weights to each frame, enabling the model to focus on active gestures while ignoring static, zero-padded frames.

## Results
The model was trained for 100 epochs using the AdamW optimizer with a Cosine Annealing Learning Rate Scheduler and Automatic Mixed Precision (AMP). 

Evaluated on a completely unseen test set (17.92% of the dataset):
* **Training Accuracy:** 99.64%
* **Validation Accuracy:** 68.54%
* **Test Set Accuracy:** 62.04%
* **Average BLEU Score:** 55.73
* **Average CHRF Score:** 67.55
