This project presents a privacy-preserving stress recognition system using EEG (Electroencephalogram) signals combined with Federated Learning and Transfer Learning techniques. The goal of this project is to build a machine learning model that can detect human stress levels from brain signals without sharing raw EEG data, thereby maintaining user data privacy.

Traditional stress detection systems require centralized data collection, which raises privacy concerns and data security risks. To overcome this issue, this project uses Federated Learning, where multiple clients (devices or datasets) train a model locally and only share model updates instead of raw data. Additionally, Transfer Learning is applied to improve model performance and reduce training time by using pre-trained models.

The system processes EEG signals, extracts important features, and classifies stress levels into different categories such as stressed and not stressed using deep learning models.
# Key Features
-EEG signal preprocessing and feature extraction
-Stress classification using Deep Learning
-Federated Learning for privacy-preserving training
-Transfer Learning for improved accuracy and faster training
-Decentralized model training without sharing raw EEG data
-Performance evaluation using Accuracy, Precision, Recall, and F1-score
