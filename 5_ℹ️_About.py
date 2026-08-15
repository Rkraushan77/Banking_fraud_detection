import streamlit as st

# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="About",
    page_icon="ℹ️",
    layout="wide"
)

# ==========================================
# Title
# ==========================================

st.title("ℹ️ About This Project")

st.markdown("""
This project demonstrates the application of **Machine Learning**
to detect fraudulent banking transactions and assist financial
institutions in preventing financial fraud.
""")

st.divider()

# ==========================================
# Project Overview
# ==========================================

st.header("📌 Project Overview")

st.write("""
The Banking Fraud Detection System is an end-to-end Machine Learning
application that predicts whether a banking transaction is fraudulent
or legitimate.

The project includes:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Feature Scaling
- Model Comparison
- Hyperparameter Tuning
- Model Evaluation
- Interactive Streamlit Deployment
""")

st.divider()

# ==========================================
# Dataset
# ==========================================

st.header("📂 Dataset Information")

col1, col2 = st.columns(2)

with col1:
    st.info("""
**Dataset Type**

• Banking Transactions

• Binary Classification

• Fraud Detection
""")

with col2:
    st.info("""
**Target Variable**

fraud_flag

0 → Legitimate

1 → Fraud
""")

st.divider()

# ==========================================
# Technologies
# ==========================================

st.header("🛠 Technologies Used")

st.markdown("""
### Programming

- Python

### Data Analysis

- Pandas
- NumPy

### Visualization

- Matplotlib
- Streamlit

### Machine Learning

- Scikit-learn
- Random Forest Classifier
- GridSearchCV

### Deployment

- Streamlit Community Cloud
""")

st.divider()

# ==========================================
# ML Workflow
# ==========================================

st.header("⚙ Machine Learning Workflow")

st.markdown("""
1. Data Collection

2. Data Cleaning

3. Feature Engineering

4. Data Preprocessing

5. Train-Test Split

6. Feature Scaling

7. Model Training

8. Hyperparameter Tuning

9. Model Evaluation

10. Streamlit Deployment
""")

st.divider()

# ==========================================
# Model Summary
# ==========================================

st.header("📈 Final Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", "95.15%")
col2.metric("F1 Score", "80.08%")
col3.metric("ROC-AUC", "87.80%")

st.divider()

# ==========================================
# Key Features
# ==========================================

st.header("⭐ Key Features")

st.success("""
✔ Real-Time Fraud Prediction

✔ Interactive Dashboard

✔ Model Performance Analysis

✔ Feature Importance

✔ User-Friendly Interface

✔ Resume-Ready Machine Learning Project
""")

st.divider()

# ==========================================
# Future Improvements
# ==========================================

st.header("🚀 Future Improvements")

st.markdown("""
- Real-time API Integration

- Live Banking Transaction Monitoring

- Explainable AI (SHAP)

- Deep Learning Models

- Cloud Deployment

- Database Integration
""")

st.divider()

# ==========================================
# Developer
# ==========================================

st.header("👨‍💻 Developer")

st.write("""
**Project:** Banking Fraud Detection System

Developed using Python, Scikit-learn, and Streamlit.

This project demonstrates a complete Machine Learning pipeline
from data preprocessing to deployment.
""")

st.divider()

st.success("🎉 Thank you for exploring this project!")