import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from datetime import datetime
import seaborn as sns
from PIL import Image
import os
import requests
import zipfile

ZIP_URL = "https://drive.google.com/uc?export=download&id=14axsqY4xGIqIAPHygeVkx0nEi9c_m9k9"
ZIP_FILE = "soil_fertility_app.zip"
EXTRACT_DIR = "data"

def download_and_extract_zip():
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        if not os.path.exists(ZIP_FILE):
            print("Downloading zip...")
            with open(ZIP_FILE, "wb") as f:
                f.write(requests.get(ZIP_URL).content)
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

download_and_extract_zip()


# Page Configuration
st.set_page_config(
    page_title="Soil Fertility Prediction", 
    layout="wide",
    page_icon="🌱" 
)

# CSS
st.markdown("""
    <style>
        .main {
            background-color: #F5F5F5;
        }
        .metric-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border-left: 4px solid #2E7D32;
        }
        .section-header {
            color: #2E7D32;
            border-bottom: 2px solid #2E7D32;
            padding-bottom: 5px;
            margin-top: 20px;
        }
        .sidebar .sidebar-content {
            background-color: #F5F5F5;
        }
        .prediction-card {
            background-color: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .stProgress > div > div > div {
            background-color: #2E7D32;
        }
        .st-b7 {
            background-color: #2E7D32 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Load model ---
try:
    model = pickle.load(open("small_model.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ Model file not found. Make sure 'small_model.pkl' is in the folder.")
    st.stop()

# --- App title ---
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.title("🌾 Soil Fertility Prediction")
    st.markdown("**Harumanis Mango Vegetative Stage**")
with col2:
    # You can add a logo here if available
    st.image("https://via.placeholder.com/150x50?text=AgriTech", width=150)

# --- Sidebar navigation ---
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("", ["🔍 Predict Fertility", "📊 Performance Dashboard"])
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
        This app predicts soil fertility levels for Harumanis mango cultivation 
        using machine learning. Input soil properties to get predictions.
    """)

# --------------------------------------------
# PAGE 1: PREDICTION FORM
# --------------------------------------------
if page == "🔍 Predict Fertility":
    st.markdown('<h2 class="section-header">Soil Properties Input</h2>', unsafe_allow_html=True)
    
    # Input form in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🌱 Macronutrients")
        n = st.number_input("Nitrogen (N) ppm", min_value=0, max_value=300, help="Range: 0-300 ppm")
        p = st.number_input("Phosphorus (P) ppm", min_value=0, max_value=300, help="Range: 0-300 ppm")
        k = st.number_input("Potassium (K) ppm", min_value=0, max_value=300, help="Range: 0-300 ppm")
    
    with col2:
        st.markdown("#### 🌡️ Environmental Factors")
        temp = st.slider("Temperature (°C)", 10, 50, 25, help="Optimal range: 25-35°C")
        humidity = st.slider("Humidity (%)", 0, 100, 60, help="Optimal range: 60-80%")
    
    if st.button("Predict Fertility Level", type="primary"):
        with st.spinner("Analyzing soil properties..."):
            features = np.array([[n, p, k, temp, humidity]])
            prediction = model.predict(features)[0]
            
            # Prediction card
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([0.2, 0.8])
            with col1:
                if prediction == "Low":
                    st.image("https://via.placeholder.com/100x100/FFEEEE?text=⚠️", width=100)
                elif prediction == "Medium":
                    st.image("https://via.placeholder.com/100x100/FFF6E6?text=🔍", width=100)
                else:
                    st.image("https://via.placeholder.com/100x100/E8F5E9?text=✓", width=100)
            
            with col2:
                st.markdown(f"### Fertility Level: **{prediction}**")
                
                # Detailed recommendations
                if prediction == "Low":
                    st.error("""
                    **Recommendations:**
                    - Apply organic compost (5-10 kg per tree)
                    - Supplement with NPK fertilizer (15:15:15)
                    - Consider soil pH adjustment if needed
                    """)
                elif prediction == "Medium":
                    st.warning("""
                    **Recommendations:**
                    - Monitor soil monthly
                    - Apply balanced fertilizer (10:10:10)
                    - Maintain organic matter content
                    """)
                else:
                    st.success("""
                    **Recommendations:**
                    - Continue current practices
                    - Regular soil testing (every 3 months)
                    - Maintain organic inputs
                    """)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Save to prediction log
            log_entry = pd.DataFrame([{
                "N (ppm)": n,
                "P (ppm)": p,
                "K (ppm)": k,
                "Temp (°C)": temp,
                "Humidity (%)": humidity,
                "Prediction": prediction,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
            }])
            
            try:
                logs = pd.read_csv("prediction_log.csv")
                updated = pd.concat([logs, log_entry], ignore_index=True)
            except FileNotFoundError:
                updated = log_entry
                
            updated.to_csv("prediction_log.csv", index=False)

# --------------------------------------------
# PAGE 2: PERFORMANCE DASHBOARD
# --------------------------------------------
elif page == "📊 Performance Dashboard":
    st.markdown('<h2 class="section-header">Model Performance Dashboard</h2>', unsafe_allow_html=True)
    
    try:
        results = pd.read_csv("ml_results.csv")
        
        # Key Metrics
        st.markdown("### 📊 Key Performance Indicators")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Accuracy", f"{accuracy_score(results['Actual'], results['Predicted']):.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            report = classification_report(results['Actual'], results['Predicted'], output_dict=True)
            st.metric("Weighted F1-Score", f"{report['weighted avg']['f1-score']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Predictions", len(results))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Confusion Matrix
        st.markdown("### 🧮 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8,6))
        cm = confusion_matrix(results["Actual"], results["Predicted"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", 
                   xticklabels=np.unique(results["Actual"]), 
                   yticklabels=np.unique(results["Actual"]))
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix", fontweight='bold')
        st.pyplot(fig)
        
        # Feature Importance
        st.markdown("### 🔍 Feature Importance")
        importance_df = pd.read_csv("feature_importance.csv").sort_values("Importance", ascending=False)
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(data=importance_df, x="Importance", y="Feature", palette="Greens_d", ax=ax)
        ax.set_title("Relative Feature Importance", fontweight='bold')
        st.pyplot(fig)
        
        # Classification Report
        st.markdown("### 📋 Detailed Classification Report")
        st.dataframe(
            pd.DataFrame(classification_report(results["Actual"], results["Predicted"], output_dict=True)))
        
        # Prediction History
        st.markdown("### 📝 Recent User Predictions")
        try:
            logs = pd.read_csv("prediction_log.csv")
            st.dataframe(
                logs.tail(10).sort_values("Timestamp", ascending=False).style.background_gradient(
                    subset=["N (ppm)", "P (ppm)", "K (ppm)"], cmap="Greens"))
        except FileNotFoundError:
            st.info("No prediction history available yet.")
            
    except FileNotFoundError:
        st.error("Performance data not found. Please run model training first.")
