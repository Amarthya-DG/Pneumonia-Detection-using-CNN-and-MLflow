## Pneumonia Detection using CNN with PyTorch and MLflow
This repository contains a Convolutional Neural Network (CNN) model to detect pneumonia from chest X-ray images. The model is built using PyTorch, and MLflow is utilized to track the training process, hyperparameters, and metrics.

## Project Overview
The objective of this project is to classify X-ray images into two categories:

Normal
Pneumonia
The dataset used for this task is sourced from the Pneumonia X-ray Images dataset, which is available on Kaggle.

## Features
CNN Model: A custom convolutional neural network built for binary classification.
Data Augmentation: Includes random horizontal flipping during training for better generalization.
Image Preprocessing: Images are resized to 500x500 pixels, converted to grayscale, and normalized.
Model Tracking with MLflow: Logs hyperparameters, metrics, and the model for easy experiment tracking.
Evaluation Metrics: The confusion matrix and classification report provide insights into the model’s performance.

## Dataset
The dataset contains chest X-ray images classified into two categories: Normal and Pneumonia. You can download the dataset from Kaggle using the following command:

!kaggle datasets download -d pcbreviglieri/pneumonia-xray-images
Extract the dataset into the following structure:

dataset/
│
└───cnn/
    └───pneumonia_revamped/
        ├───train/  # Training images
        ├───val/    # Validation images
        └───test/   # Test images


## Install the dependencies:

pip install -r requirements.txt
Download the dataset from Kaggle and place it in the data/ directory.

## Run the training:

python train.py

Model Architecture
CNN: Consists of multiple convolutional layers followed by ReLU activations and max-pooling layers.
Fully Connected Layers: These layers map the features to the output with softmax activation for classification.

## MLflow Integration
All model parameters, metrics, and the trained model will be logged in MLflow. Start the MLflow UI with:

mlflow ui
Navigate to http://localhost:5000 in your browser to explore the training metrics.

## Conclusion
This project implements a CNN for pneumonia detection using chest X-rays, tracking performance with MLflow. The model achieved 78% accuracy after 2 epochs, and further training can improve performance.
