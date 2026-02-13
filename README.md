# Diabetic Retinopathy Prediction System 👁️

A professional AI system for classifying Diabetic Retinopathy (DR) and Diabetic Macular Edema (DME) from retinal scans.

## 🚀 Overview
-   **Model**: ResNet50 (Fine-tuned for binary classification).
-   **Dataset**: [OLIVES Dataset](https://huggingface.co/datasets/gOLIVES/OLIVES_Dataset) (Fundus/OCT images).
-   **Features**:
    -   **Validation**: GPT-4o Vision filters out non-medical images.
    -   **Diagnosis**: Detects "Healthy" vs "At Risk".
    -   **Explanation**: GPT-4 generates clinical summaries.

## 📂 Project Structure
-   `train.py`: Downloads OLIVES data, trains model, saves to `model.pth`.
-   `app.py`: Streamlit frontend for interactive testing.
-   `model.pth`: Trained model weights (generated after training).

## 🛠️ Usage

1.  **Install Dependencies**
    ```bash
    pip install torch torchvision datasets openai streamlit
    ```

2.  **Train Model**
    ```bash
    python train.py
    ```
    *Training takes ~5 mins on demo mode (100 samples).*

3.  **Run Application**
    ```bash
    streamlit run app.py
    ```
    *Open http://localhost:8501 in your browser.*

## 📊 Dataset Info
The OLIVES dataset contains paired Fundus and OCT images with clinical labels. This project uses the `disease_classification` subset, mapping labels to a binary Healthy/At-Risk target for demonstration.
