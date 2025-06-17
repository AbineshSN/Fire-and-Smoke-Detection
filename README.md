# Fire-and-Smoke-Detection

# Overview

Fire Detection is a deep learning project developed to identify the presence of fire in images using TensorFlow and computer vision techniques. Implemented through Python, this project constructs Convolutional Neural Networks (CNNs) using both Sequential and Functional APIs to classify images as containing fire or not. The system automates fire detection for applications in safety monitoring and emergency response, achieving high accuracy through robust image preprocessing, model training, and evaluation. The workflow is designed for reproducibility and experimentation, leveraging standard deep learning practices.

# Objectives

Build an accurate image classification model to detect fire in images.
Implement and compare CNN models using TensorFlow’s Sequential and Functional APIs.
Process and augment image datasets to enhance model performance.
Evaluate models with metrics like accuracy, precision, and recall, and visualize results for interpretability.


# Key Components

Data Preprocessing: Cleans dataset by removing corrupted images, resizes images to 256x256 pixels, normalizes pixel values, and applies data augmentation.
Dataset Splitting: Divides data into 70% training, 20% validation, and 10% test sets for balanced model training and evaluation.
Model Architecture: Constructs CNNs with Conv2D, MaxPooling2D, Flatten, and Dense layers, using ReLU and sigmoid activations for binary classification.
Training: Trains models over 20 epochs using the Adam optimizer and binary cross-entropy loss, monitoring accuracy, precision, and recall.
Evaluation & Visualization: Assesses model performance on test sets, achieving up to 99% accuracy, with plots of loss and accuracy to demonstrate learning stability.


# Technologies Used

Programming Language: Python 3.8+
Deep Learning Framework: TensorFlow
Computer Vision: OpenCV (implied for image processing)
Data Processing: NumPy (assumed for numerical operations)
Visualization: Matplotlib (implied for plotting)
Models: Custom CNNs with Sequential and Functional APIs
Environment: Jupyter Notebook (assumed), Conda or Pip
Version Control: Git


# Dataset

The dataset comprises images labeled as fire or non-fire, stored in separate directories. Images are in JPEG, JPG, BMP, or PNG formats, with preprocessing to remove faulty files. The dataset is not included due to licensing but can be sourced from public repositories or custom collections compatible with TensorFlow’s image processing utilities.


# Project Structure

notebooks/: Jupyter Notebooks for preprocessing, training, evaluation, and prediction
data/: Placeholder for dataset (not included)
models/: Stores trained model weights
outputs/: Contains evaluation metrics, visualizations, and prediction results
README.md: Project documentation


# Results

The models achieve exceptional performance:
Sequential Model: ~100% training accuracy, 98% validation accuracy, 98% test accuracy.
Functional API Model: 99% test accuracy, 1.0 precision, 0.95 recall. Visualizations show stable loss reduction and accuracy improvement, with predictions accurately identifying fire in unseen images.
