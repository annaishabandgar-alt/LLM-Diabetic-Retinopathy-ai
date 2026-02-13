# 🩺 LLM-Diabetic-Retinopathy-AI

An AI-powered diagnostic system for detecting **Diabetic Retinopathy (DR)** and **Diabetic Macular Edema (DME)** from retinal scans using Deep Learning and LLM-based clinical explanations.

---

## 👁️ Project Overview

This project combines:

- 🧠 Deep Learning (ResNet50) for medical image classification  
- 👁️ Vision validation (GPT-4o Vision) to filter non-medical images  
- 📄 LLM-based clinical summaries (GPT-4) for diagnosis explanation  
- 🌐 Streamlit Web App for interactive predictions  

The model classifies retinal scans into:

- ✅ Healthy  
- ⚠️ At Risk  

---

## 🏗️ Model Details

- Architecture: ResNet50 (Transfer Learning)
- Task: Binary Classification (Healthy vs At Risk)
- Dataset: OLIVES Dataset (Fundus & OCT images)
- Output: Diagnosis + AI-generated explanation

---

## 📂 Project Structure

LLM-Diabetic-Retinopathy-ai/
│
├── train.py        # Model training script
├── app.py          # Streamlit web application
├── model.pth       # Saved trained model weights
├── README.md       # Project documentation

---

## 📊 Dataset

OLIVES Dataset  
Contains paired Fundus and OCT retinal images with clinical labels.

This implementation uses the disease_classification subset and maps labels into:

- Healthy → 0  
- At Risk → 1  

Dataset Source:  
https://huggingface.co/datasets/gOLIVES/OLIVES_Dataset

---

## ⚙️ Installation

### 1️⃣ Clone Repository

git clone https://github.com/your-username/LLM-Diabetic-Retinopathy-ai.git
cd LLM-Diabetic-Retinopathy-ai

---

### 2️⃣ Install Dependencies

pip install torch torchvision datasets openai streamlit

---

## 🚀 Usage

### 🔹 Train Model

python train.py

Demo mode uses 100 samples (~5 minutes).

---

### 🔹 Run Web App

streamlit run app.py

Open in browser:
http://localhost:8501

---

## 🧠 System Workflow

1. User uploads retinal scan  
2. GPT-4o Vision validates medical image  
3. ResNet50 predicts Healthy / At Risk  
4. GPT-4 generates clinical-style explanation  
5. Results displayed in Streamlit interface  

---

## 📈 Future Improvements

- Multi-class DR severity grading  
- Model deployment (AWS / GCP / Azure)  
- Database integration for patient records  
- Real-time OCT processing  
- Performance optimization with GPU  

---

## ⚠️ Disclaimer

This system is for educational and research purposes only.  
It is not intended for clinical or medical diagnosis.

---

## 👨‍💻 Author

Developed as an AI-powered medical imaging research project.
