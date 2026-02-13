import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from openai import OpenAI
import base64
from io import BytesIO
import os
from dotenv import load_dotenv

# Config
load_dotenv()
ENV_KEY = os.getenv("OPENAI_API_KEY", "").strip()
MODEL_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(page_title="Retinal AI", page_icon="👁️", initial_sidebar_state="collapsed")

# Sidebar Logic
user_key = ENV_KEY # Default to Env

if not ENV_KEY:
    with st.sidebar:
        st.header("Settings")
        user_key = st.text_input("OpenAI API Key", type="password")
        if not user_key:
            st.error("⚠️ Enter API Key to proceed.")
            st.stop()

from tenacity import retry, stop_after_attempt, wait_fixed

client = OpenAI(api_key=user_key, timeout=20.0, max_retries=3)

# Retry Decorator for Robustness
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_openai_vision(b64):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [{"type": "text", "text": "Is this a retinal Fundus/OCT scan? YES/NO"}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
        max_tokens=5
    )

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH): return None
    m = models.resnet50(weights='DEFAULT')
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return m.to(DEVICE).eval()

def validate_image(img):
    # Ensure RGB for JPEG compatibility
    if img.mode != 'RGB': img = img.convert('RGB')
    buf = BytesIO()
    img.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    try:
        res = call_openai_vision(b64)
        return "YES" in res.choices[0].message.content.upper()
    except Exception as e:
        # Log error but allow continuation to prevent blocker
        print(f"Validation Warning: {e}")
        return True

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def call_openai_text(prompt):
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

def predict(img, model):
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        probs = torch.softmax(model(tf(img.convert('RGB')).unsqueeze(0).to(DEVICE)), 1)
        conf, pred = probs.max(1)
    return pred.item(), conf.item()

# UI
st.title("👁️ Retinal Disease AI")
model = load_model()

import csv
from datetime import datetime

# ... (Previous imports remain, ensure csv/datetime are available)

# Logging Function
def log_prediction(img_name, features, pred_class, confidence):
    file = "predictions.csv"
    headers = ["Timestamp", "Image_Name", "Age", "HbA1c", "BP", "Diabetes_Duration", "Lipids", "Biomarker_Data", "Predicted_Class", "Confidence"]
    
    # Prepare Row
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        img_name,
        features.get("Age", ""),
        features.get("HbA1c", ""),
        features.get("BP", ""),
        features.get("Diabetes_Duration", ""),
        features.get("Lipids", ""),
        features.get("Biomarker_Data", ""),
        pred_class,
        f"{confidence:.2%}"
    ]
    
    # Append to CSV
    try:
        write_header = not os.path.exists(file)
        with open(file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header: writer.writerow(headers)
            writer.writerow(row)
    except Exception as e: print(f"Logging Error: {e}")

# ... (Load Model/Validate functions remain same)

if not model: st.error(f"Model {MODEL_PATH} not found. Run train.py first.")
else:
    # 🔹 1. Clinical Features Form
    features = {}
    with st.expander("🩺 Clinical Features (Optional)"):
        c1, c2 = st.columns(2)
        features["Age"] = c1.number_input("Age", min_value=0, max_value=120, value=None)
        features["HbA1c"] = c2.number_input("HbA1c (%)", min_value=0.0, max_value=20.0, value=None)
        features["BP"] = c1.text_input("Blood Pressure (e.g. 120/80)")
        features["Diabetes_Duration"] = c2.number_input("Diabetes Duration (Years)", min_value=0, value=None)
        features["Lipids"] = c1.text_input("Lipids / Cholesterol")
        
        # CSV Upload for Biomarkers
        bio_file = c2.file_uploader("Fetch Biomarkers from CSV", type=["csv"])
        if bio_file:
            try:
                import pandas as pd
                df = pd.read_csv(bio_file)
                if not df.empty:
                    # Take first row and convert to string representation
                    bio_data = df.iloc[0].to_dict()
                    features["Biomarker_Data"] = str(bio_data)
                    st.caption(f"✅ Loaded: {features['Biomarker_Data']}")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    # Construct Clinical Context String
    clinical_context = ", ".join([f"{k}: {v}" for k,v in features.items() if v])
    
    # 🔹 2. Image Upload
    up = st.file_uploader("Upload Retinal Scan", type=["jpg", "png"])
    
    if up:
        img = Image.open(up)
        st.image(img, width=300)
        
        if st.button("Analyze & Log"):
            if not validate_image(img):
                st.error("Invalid Image: Not a retinal scan.")
            else:
                lbls = {0: "Healthy", 1: "At Risk (DR/DME)"}
                pred, conf = predict(img, model)
                pred_label = lbls[pred]
                
                # 🔹 3. Display Results
                col1, col2 = st.columns(2)
                col1.metric("Result", pred_label, delta_color="inverse" if pred else "normal")
                col2.metric("Confidence", f"{conf:.1%}")
                
                # 🔹 4. Log to CSV
                log_prediction(up.name, features, pred_label, conf)
                st.toast("✅ Data Saved to CSV")

                # 🔹 5. Smart AI Analysis
                with st.spinner("Analyzing Clinical Data..."):
                    try:
                        prompt = f"""
                        Patient Analysis.
                        Image Prediction: {pred_label} (Confidence: {conf:.1%}).
                        Clinical Data: {clinical_context if clinical_context else 'None provided'}.
                        
                        Task: Provide a medical assessment.
                        1. If Clinical Data exists, integrate it. Example: If Image is Healthy but HbA1c is high, warn about potential future risk.
                        2. If Image is At Risk, explain the severity.
                        3. Keep it brief and professional for a patient.
                        """
                        res = call_openai_text(prompt)
                        st.info(res.choices[0].message.content)
                    except Exception as e:
                        # Fallback
                        st.warning("⚠️ AI Analysis offline. Showing standard result.")
                        if pred == 1: st.info("Result: At Risk. Please consult a doctor.")
                        else: st.success("Result: Healthy. maintain regular checkups.")
