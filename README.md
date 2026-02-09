<p align="center">
  <img src="assets/banner.png" width="100%">
</p>

<h1 align="center">🤖 AI-Powered Face Recognition Attendance System</h1>

<p align="center">
  <b>Real-Time | Secure | Intelligent | Web-Based</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue">
  <img src="https://img.shields.io/badge/AI-CNN-green">
  <img src="https://img.shields.io/badge/Framework-Streamlit-red">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-orange">
</p>

---

## 🚀 Project Overview

The **AI-Powered Face Recognition Attendance System** is a smart web-based application that automatically marks attendance using **facial recognition technology**.

It replaces traditional manual attendance with an **accurate, secure, and real-time AI solution** powered by **CNN, OpenCV, and Streamlit**.

---

## 🎯 Problem Statement

Traditional attendance systems suffer from:

❌ Proxy attendance  
❌ Manual errors  
❌ Time wastage  
❌ Poor record management  

📌 **Solution:**  
An automated attendance system using **Artificial Intelligence** and **Computer Vision**.

---

## ⭐ Key Features

✅ Secure Login & Registration  
✅ Real-Time Face Detection  
✅ CNN-Based Face Recognition  
✅ Attendance Only Once Per Day  
✅ “Already Attendance Marked” Alert  
✅ Add New Person from Web UI  
✅ Start / Stop Camera  
✅ Logout (Exit to Login Page)  
✅ Attendance Dashboard  
✅ Download Attendance Report (CSV)  
✅ Clean, Colorful & Responsive UI  

---

## 🛠️ Technology Stack

| Category | Tools |
|--------|------|
| Programming | Python |
| AI / ML | CNN, TensorFlow, Keras |
| Computer Vision | OpenCV |
| Web App | Streamlit |
| Data Handling | Pandas, NumPy |
| Storage | CSV Files |

---

## 🧩 System Architecture



1️⃣ Camera captures face  
2️⃣ OpenCV detects face  
3️⃣ CNN recognizes identity  
4️⃣ Attendance logic validates  
5️⃣ Record saved with date & time  
6️⃣ Dashboard updates live  

---

## 📸 Application Screenshots

### 🔐 Login & Registration


### 📷 Real-Time Face Recognition


### 📊 Attendance Dashboard


---

## 📁 Project Structure

```
MachineLearning-Project-Final/
│
├── 📂 dataset/                  # Face images of registered users
│
├── 📂 model/
│   └── 🧠 face_model.h5         # Trained CNN face recognition model
│
├── 📄 attendance.csv            # Attendance records (Name, Date, Time)
├── 📄 users.csv                 # Login & registration credentials
│
├── 🐍 app.py                    # Main Streamlit web application
│
├── 🎨 assets/                   # UI images & banners
│   ├── 🖼️ banner.png            # Project banner image
│   ├── 🔐 login.png             # Login page screenshot
│   ├── 📷 camera.png            # Face recognition screen
│   └── 📊 dashboard.png         # Attendance dashboard
│
└── 📘 README.md                 # Project documentation
```

---

## 🧠 How the System Works

The system captures real-time video from the webcam and detects faces using **OpenCV Haar Cascade**.  
Each detected face is preprocessed and passed to a **CNN model** trained on registered user images.  
If the face is recognized with sufficient confidence, attendance is marked automatically.

---

## 🔄 Application Workflow

1. User logs in or registers
2. Camera is started from the web interface
3. Face is detected in real time
4. CNN model predicts the identity
5. Attendance is checked for duplication
6. Attendance is stored with date & time
7. Dashboard updates instantly

---

## 📸 Dataset Handling

- Each registered person has a separate folder inside `dataset/`
- 20 face images are captured per person
- Images are:
  - Converted to grayscale
  - Resized to 100×100 pixels
- Dataset is used to train the CNN model

---

## 🧠 CNN Model Description

The Convolutional Neural Network (CNN) consists of:
- Convolution layers for feature extraction
- MaxPooling layers for dimensionality reduction
- Fully connected layers for classification
- Softmax activation for final prediction

The trained model is saved as:

---

---

## 👨‍💻 Developer

## **MIT UMARETIYA**

## 📫 Contact Information

- **GitHub:** https://github.com/Mit-Gitprofile
- **LinkedIn:** https://www.linkedin.com/in/mit-umaretiya-562048348/  
- **Email:** mitumaretiya29@gmail.com 
---
### 🛠 Technologies Used
- Python  
- Machine Learning  
- CNN (Deep Learning)  
- OpenCV  
- TensorFlow / Keras  
- Streamlit  
- Pandas & NumPy  

---

⭐ If you like this project, please consider giving it a star!


