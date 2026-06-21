# AI-Based Traffic Violation Detection System

## Overview

The AI-Based Traffic Violation Detection System is a computer vision application developed to automatically identify traffic rule violations from vehicle images. The system uses the YOLOv8 object detection model to detect multiple traffic violations and EasyOCR to extract vehicle registration numbers from detected license plates.

The project provides an end-to-end pipeline that detects violations, extracts vehicle numbers, and generates a downloadable violation report through an interactive Streamlit web application.

---

## Features

* Helmet Detection
* No-Helmet Detection
* Triple Riding Detection
* Vehicle Number Plate Detection
* License Plate Text Extraction using OCR
* PDF Report Generation
* Streamlit-Based User Interface
* Real-Time Image Analysis

---

## System Workflow

Traffic Image Upload

↓

YOLOv8 Object Detection

↓

Violation Identification

* With Helmet
* Without Helmet
* Triple Riding
* Number Plate

↓

License Plate Localization

↓

Plate Cropping

↓

EasyOCR Text Extraction

↓

Violation Report Generation

↓

Results Display in Streamlit Dashboard

---

## Dataset

Traffic Violation Dataset used for training:

Dataset Link:
https://www.kaggle.com/datasets/devgurucodes/trafffic-violations-triple-riding-no-helmet-plate

Classes:

* Plate
* WithHelmet
* WithoutHelmet
* TripleRiding

---

## Model Training

### Model

* YOLOv8s

### Training Configuration

| Parameter   | Value              |
| ----------- | ------------------ |
| Epochs      | 15                 |
| Batch Size  | 16                 |
| Image Size  | 800 × 800          |
| Framework   | Ultralytics YOLOv8 |
| Environment | Google Colab       |
| GPU         | Tesla T4           |

### Dataset Split

| Split      | Images |
| ---------- | -----: |
| Train      |  ~5000 |
| Validation |    383 |
| Test       |    194 |

---

## Technologies Used

### Frontend

* Streamlit

### Backend

* Python

### Deep Learning

* YOLOv8
* PyTorch

### OCR

* EasyOCR

### Data Processing

* OpenCV
* NumPy
* Pandas

---

## Project Structure

```text
grid_prototype/
│
├── app.py
├── train_model.py
├── small.yaml
├── requirements.txt
├── README.md
└── Grid_yolo_project_2026.ipynb
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yaswitha525/Traffic_Violation_Detection_System_prototype.git

cd Traffic_Violation_Detection_System_prototype
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## Demo Video

Project Demonstration:

https://drive.google.com/file/d/1LU3kwLJsQ0efW6qxLMdf_ihB_1tiStvU/view?usp=drivesdk

---

## Output

The system can:

* Detect traffic violations from uploaded images.
* Identify riders with and without helmets.
* Detect triple riding violations.
* Detect vehicle number plates.
* Extract vehicle registration numbers using OCR.
* Generate downloadable PDF reports.

---

## Future Enhancements

* Real-Time Video Processing
* Live CCTV Integration
* Automatic Challan Generation
* Vehicle Database Integration
* Traffic Analytics Dashboard
* Cloud Deployment

---

## Authors

Dandamudi Sai Yaswitha

B.Tech – Artificial Intelligence & Machine Learning

Shri Vishnu Engineering College for Women

---

## License

This project is developed for educational, research, and hackathon purposes.
