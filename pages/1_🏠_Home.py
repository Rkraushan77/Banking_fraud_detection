import streamlit as st

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Home",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------
# Hero Section
# ---------------------------------------------------

st.title("🏦 Banking Fraud Detection System")

st.markdown("""
### Intelligent Fraud Detection Using Machine Learning

This application predicts whether a banking transaction is **Fraudulent** or **Legitimate**
using a **Random Forest Classifier** optimized through **GridSearchCV**.

The goal is to help financial institutions identify suspicious transactions
and reduce fraudulent activities using data-driven insights.
""")

st.divider()

# ---------------------------------------------------
# Model Performance
# ---------------------------------------------------

st.subheader("📊 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", "95.15%")
col2.metric("Precision", "82.28%")
col3.metric("Recall", "78.00%")
col4.metric("F1 Score", "80.08%")
col5.metric("ROC AUC", "87.80%")

st.divider()

# ---------------------------------------------------
# Project Overview
# ---------------------------------------------------

st.subheader("📌 Project Overview")

st.write("""
This project focuses on detecting fraudulent banking transactions using
Machine Learning techniques. Multiple classification algorithms were trained
and compared, and the best-performing model was selected after
hyperparameter tuning using GridSearchCV.

The final model is deployed as an interactive Streamlit web application
for real-time fraud prediction.
""")

st.divider()

# ---------------------------------------------------
# Workflow
# ---------------------------------------------------

st.subheader("⚙️ Machine Learning Workflow")

st.markdown("""
✅ Data Collection

⬇️

✅ Data Cleaning

⬇️

✅ Exploratory Data Analysis (EDA)

⬇️

✅ Feature Engineering

⬇️

✅ Feature Scaling

⬇️

✅ Model Comparison

⬇️

✅ Hyperparameter Tuning

⬇️

✅ Model Evaluation

⬇️

✅ Fraud Prediction

⬇️

✅ Streamlit Deployment
""")

st.divider()

# ---------------------------------------------------
# Technologies Used
# ---------------------------------------------------

st.subheader("🛠️ Technologies Used")

tech1, tech2, tech3 = st.columns(3)

with tech1:
    st.info("""
**Programming**

- Python
- Pandas
- NumPy
""")

with tech2:
    st.info("""
**Machine Learning**

- Scikit-learn
- Random Forest
- GridSearchCV
""")

with tech3:
    st.info("""
**Visualization**

- Matplotlib
- Seaborn
- Streamlit
""")

st.divider()

# ---------------------------------------------------
# Dataset Information
# ---------------------------------------------------

st.subheader("📁 Dataset Summary")

st.success("""
Dataset: Banking Transactions Dataset

Target Variable:
• fraud_flag

Total Features:
• 18 Input Features

Prediction:
• Fraud
• Legitimate Transaction
""")

st.divider()

# ---------------------------------------------------
# Business Objective
# ---------------------------------------------------

st.subheader("🎯 Business Objective")

st.write("""
Financial fraud causes significant losses to banks every year.
This system helps detect suspicious transactions before they are approved,
allowing banks to reduce fraud risk and improve customer security.
""")

st.divider()

# ---------------------------------------------------
# Project Highlights
# ---------------------------------------------------

st.subheader("⭐ Project Highlights")

left, right = st.columns(2)

with left:
    st.success("""
✔ Data Cleaning

✔ Feature Engineering

✔ Model Comparison

✔ Hyperparameter Tuning

✔ Feature Importance Analysis
""")

with right:
    st.success("""
✔ Random Forest Classifier

✔ Fraud Prediction

✔ Streamlit Deployment

✔ Interactive Dashboard

✔ Resume Ready Project
""")

st.divider()

st.caption("Developed using Python, Scikit-learn and Streamlit")
