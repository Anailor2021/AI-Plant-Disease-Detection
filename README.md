# CropScan AI — AI Plant Disease Detection

CropScan AI is a deep learning web application that detects plant diseases from leaf images using a MobileNetV2 model trained on the PlantVillage dataset. Upload a photo of a plant leaf and the app identifies the disease, displays a Grad-CAM heatmap showing what the model focused on, and provides treatment recommendations specific to the detected disease.

Presentation: [https://youtu.be/rEvtURbUq00]

---

## Team Members

- Daunte Williams  
- Austin Nailor  
- Katiana Newbold  
- Walther Aragon  
- Justin Cabrera

---

## Objectives

Plant diseases are responsible for significant crop losses worldwide. Early detection is critical but requires expert knowledge that many farmers do not have access to. This project addresses that by building an AI system that provides instant diagnosis and actionable treatment advice.

---

## Model

Built using transfer learning with MobileNetV2 pretrained on ImageNet. The base model was frozen and two dense layers were added on top to classify 38 disease categories. Training was done on Kaggle using two Tesla T4 GPUs.

- Dataset: PlantVillage (54,000+ leaf images, 38 classes, 14 crop species)
- Training accuracy: 98.96%
- Validation accuracy: 95.25%
- Epochs: 10

---

## Features

- Classifies leaf images into 38 plant disease categories
- Shows top 3 predictions with confidence scores
- Grad-CAM heatmap to visualize what regions the model focused on
- Disease-specific treatment recommendations for each of the 38 classes
- Camera support on mobile devices for live photo capture
- Responsible AI disclaimer with low confidence warnings

---

## Project structure

```
CropScan-AI/
├── app.py                  Flask backend with Grad-CAM and prediction logic
├── requirements.txt        Python dependencies
├── class_names.json        List of 38 disease class labels
├── model_weights.pkl       Trained MobileNetV2 weights
├── Dockerfile              Docker configuration for deployment
├── templates/
│   └── index.html          Frontend web interface
└── assets/
    ├── accuracy_loss_curves.png
    ├── class_distribution.png
    ├── confusion_matrix.png
    └── sample_predictions.png
```

---

## How to run locally

Clone the repo and install dependencies:

```
pip install -r requirements.txt
```

Run the app:

```
python app.py
```

Open your browser at the local URL that appears in the terminal once the app starts.

---

## Tech stack

Python, Flask, TensorFlow, MobileNetV2, OpenCV, Grad-CAM, Gunicorn

---

## Dataset

PlantVillage Dataset on Kaggle
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

---
