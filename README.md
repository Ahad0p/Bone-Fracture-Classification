# 🦴 Bone Fracture Classification using CNN (VGG16 + Flask Web App)

This project uses a **Convolutional Neural Network (CNN)** based on **VGG16** for classifying X-ray bone fracture images into two categories:
- **Oblique Fracture**
- **Spiral Fracture**

A simple **Flask web application** is built to upload an image and display predictions in real time.

---

## 🚀 Project Overview

This deep learning model is designed to help in **automated bone fracture detection** from medical X-ray images.  
It leverages **transfer learning** using the **VGG16** architecture (pre-trained on ImageNet), fine-tuned for this specific 2-class classification task.

---

## 🧠 Features

- 🧩 **Transfer Learning (VGG16)** for feature extraction  
- 📈 **High accuracy** on test data  
- 🖼️ **Image preprocessing and augmentation**  
- 💻 **Flask Web App** for real-time image classification  
- ⚙️ **Deployed-ready model** (`.keras` format)  

---

## 🏗️ Project Structure

📦 Bone-Fracture-Classification
├── model/
│ └── model_vgg16.keras # Saved CNN model
├── static/
│ └── uploads/ # Uploaded images
├── templates/
│ └── index.html # Frontend template
├── app.py # Flask backend
├── main.py # (Optional) Debug or verification script
├── Untitled5.ipynb # Training notebook
└── README.md # Project documentation


---

## ⚙️ Requirements

Install dependencies inside your virtual environment:

```bash
pip install tensorflow keras flask pillow numpy opencv-python

🧬 Model Architecture

Base Model: VGG16 (pre-trained on ImageNet)

Layers Added:

Flatten()

Dense(256, activation='relu')

Dense(2, activation='softmax')

Loss Function: Categorical Crossentropy
Optimizer: Adam
Metrics: Accuracy

🧾 Training Summary

Image Input Size: 224 × 224 × 3

Epochs: 20 (adjustable)

Batch Size: 32

Augmentation: Rotation, flipping, zoom

Accuracy: ~95% (after fine-tuning)