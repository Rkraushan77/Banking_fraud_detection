# 🏦 Banking Fraud Detection System

An end-to-end Machine Learning project that detects potentially fraudulent banking transactions using classification algorithms and provides an interactive **Streamlit web application** for real-time prediction and model analysis.

---

## 📌 Project Overview

Financial fraud is a major challenge for banking and financial institutions. The objective of this project is to develop a machine learning system that can identify suspicious banking transactions and classify them as:

* **0 → Legitimate Transaction**
* **1 → Fraudulent Transaction**

The project covers the complete machine learning workflow:

**Data → Preprocessing → Model Training → Model Comparison → Hyperparameter Tuning → Evaluation → Feature Importance → Model Serialization → Streamlit Deployment**

---

## 🎯 Project Objectives

* Detect potentially fraudulent banking transactions.
* Compare multiple machine learning classification algorithms.
* Select the best-performing model.
* Optimize the selected model using hyperparameter tuning.
* Evaluate the model using fraud-appropriate metrics.
* Understand important transaction features.
* Save the trained model for deployment.
* Build an interactive Streamlit application.
* Provide a user-friendly fraud prediction interface.

---

## 🧠 Machine Learning Workflow

```text
                    Banking Transaction Data
                              │
                              ▼
                     Data Understanding
                              │
                              ▼
                         Data Cleaning
                              │
                              ▼
                    Feature Preparation
                              │
                              ▼
                    Train / Test Split
                              │
                              ▼
                    Model Comparison
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
     Logistic Regression     KNN          Naive Bayes
            │
            ├──────── Decision Tree
            ├──────── Random Forest
            └──────── Gradient Boosting
                              │
                              ▼
                    Best Model Selection
                              │
                              ▼
                     GridSearchCV
                              │
                              ▼
                     5-Fold CV Tuning
                              │
                              ▼
                    Final Random Forest
                              │
                              ▼
                    Test Set Evaluation
                              │
                              ▼
                    Feature Importance
                              │
                              ▼
                       Joblib Model
                              │
                              ▼
                    Streamlit Application
```

---

# 📊 Dataset & Features

The project uses transaction-level banking information containing behavioral, transactional, device, account, geographic and authentication-related attributes.

### Key Features

| Feature                          | Description                                         |
| -------------------------------- | --------------------------------------------------- |
| `transaction_amount`             | Monetary value of the transaction                   |
| `login_attempts`                 | Number of login attempts                            |
| `device_risk_score`              | Risk score associated with the device               |
| `transfer_frequency`             | Frequency of transfers                              |
| `anomaly_score`                  | Measure of unusual transaction behavior             |
| `account_age_days`               | Age of the account                                  |
| `transaction_time_hour`          | Hour when transaction occurred                      |
| `failed_transactions_last_30d`   | Failed transactions during previous 30 days         |
| `avg_monthly_balance`            | Average monthly account balance                     |
| `daily_transaction_count`        | Number of daily transactions                        |
| `geo_distance_km`                | Geographic distance associated with the transaction |
| `session_duration_minutes`       | Duration of the user session                        |
| `transaction_velocity_score`     | Transaction velocity/risk measure                   |
| `payment_channel`                | Transaction channel                                 |
| `authentication_type`            | Authentication mechanism                            |
| `card_present_flag`              | Indicates whether card was present                  |
| `international_transaction_flag` | Indicates international transaction                 |
| `suspicious_ip_flag`             | Indicates suspicious IP activity                    |
| `fraud_flag`                     | Target variable                                     |

Categorical variables such as payment channel and authentication type were converted into numerical features using encoding.

---

# 🤖 Models Compared

The following classification algorithms were evaluated:

1. Logistic Regression
2. K-Nearest Neighbors
3. Gaussian Naive Bayes
4. Decision Tree
5. Random Forest
6. Gradient Boosting

The models were compared using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

# 🌲 Final Model

The final model selected for the project is:

## Random Forest Classifier

Random Forest was selected because it can effectively model nonlinear relationships and interactions in tabular transaction data while providing feature-importance information.

---

# ⚙️ Hyperparameter Tuning

The Random Forest model was optimized using:

### GridSearchCV

The search included:

```python
{
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}
```

### Best Parameters

```python
{
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 2
}
```

### Cross-Validation

A **5-fold cross-validation** strategy was used during hyperparameter tuning.

The training dataset was divided into five folds, with each fold used as a validation set once while the remaining folds were used for training.

---

# 📈 Final Model Performance

The final Random Forest model achieved the following results on the held-out test dataset:

| Metric    |      Score |
| --------- | ---------: |
| Accuracy  | **95.15%** |
| Precision | **82.28%** |
| Recall    | **78.00%** |
| F1 Score  | **80.08%** |
| ROC-AUC   | **87.80%** |

### Interpretation

**Accuracy — 95.15%**

The model correctly classified approximately 95% of the test transactions overall.

**Precision — 82.28%**

When the model predicts a transaction as fraudulent, approximately 82% of those predictions are actually fraud.

**Recall — 78.00%**

The model identifies approximately 78% of the actual fraudulent transactions.

**F1 Score — 80.08%**

The F1 score provides a balance between precision and recall.

**ROC-AUC — 87.80%**

The model demonstrates good ability to distinguish fraudulent transactions from legitimate transactions across classification thresholds.

> ⚠️ Accuracy alone is not sufficient for evaluating fraud detection because fraud datasets can be imbalanced. Precision, Recall, F1 and ROC-AUC are therefore also considered.

---

# ⭐ Feature Importance

Random Forest feature importance was used to understand which variables contributed most to the model's predictions.

The project generates:

```text
feature_importance.csv
```

This file is used by the Streamlit Model Performance page to visualize important features.

Feature importance should be interpreted as **predictive contribution**, not causal proof.

---

# 🖥️ Streamlit Application

The trained model was integrated into an interactive Streamlit application.

### Application Pages

```text
🏠 Home
📊 Dashboard
🤖 Prediction
📈 Model Performance
ℹ️ About
```

### Home

Provides an overview of the fraud detection system and its purpose.

### Dashboard

Provides a visual overview of the transaction and model-related information.

### Prediction

Allows users to enter transaction information and receive a fraud prediction.

### Model Performance

Displays:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Model information
* Hyperparameters
* Feature importance

### About

Provides information about the project, methodology and technology stack.

---

# 💾 Model Deployment Architecture

The trained machine learning artifacts are stored separately so the application does not need to retrain the model every time it starts.

Example artifacts:

```text
fraud_detection_model.pkl
feature_columns.pkl
scaler.pkl
feature_importance.csv
```

### Why save `feature_columns.pkl`?

The model expects the same feature structure used during training. The saved feature schema helps ensure that prediction inputs are aligned correctly.

### Why use Joblib?

Joblib provides a convenient way to serialize and reload scikit-learn models and preprocessing objects.

---

# 🗂️ Project Structure

```text
Fraud_Detection_System/
│
├── app.py
│
├── pages/
│   ├── 1_🏠_Home.py
│   ├── 2_📊_Dashboard.py
│   ├── 3_🤖_Predict.py
│   ├── 4_📈_Model_Performance.py
│   └── 5_ℹ️_About.py
│
├── models/
│   └── fraud_detection_model.pkl
│
├── data/
│   └── banking_transactions.csv
│
├── feature_columns.pkl
├── scaler.pkl
├── feature_importance.csv
├── requirements.txt
└── README.md
```

> Adjust the folder structure above to match the exact structure of your GitHub repository.

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Random Forest
* Logistic Regression
* KNN
* Naive Bayes
* Decision Tree
* Gradient Boosting

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib

### Model Persistence

* Joblib

### Deployment

* Streamlit

### Development Environment

* VS Code
* Python Virtual Environment
* Git & GitHub

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Fraud_Detection_System.git
```

Move into the project directory:

```bash
cd Fraud_Detection_System
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 🚀 Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🔮 Prediction Workflow

The prediction process works approximately as follows:

```text
User Input
    ↓
Input Validation
    ↓
Feature Construction
    ↓
Feature Alignment
    ↓
Preprocessing
    ↓
Random Forest Model
    ↓
Prediction Probability
    ↓
Fraud / Legitimate Result
```

---

# 🔍 Example Prediction

The application produces a prediction such as:

```text
Prediction: FRAUDULENT TRANSACTION

Risk Level: High

Model Confidence: XX%
```

or:

```text
Prediction: LEGITIMATE TRANSACTION

Risk Level: Low

Model Confidence: XX%
```

---

# ⚠️ Important Model Limitations

This project is an ML prototype and should not be considered a production banking fraud system without additional validation and security controls.

Important limitations include:

### 1. Fraud patterns change

Attackers continuously change their behavior, resulting in concept drift.

### 2. Accuracy can be misleading

A high accuracy score does not necessarily mean that the model catches most fraud.

### 3. False negatives matter

A fraudulent transaction classified as legitimate can cause financial loss.

### 4. False positives matter

A legitimate transaction classified as fraud can inconvenience customers and increase investigation costs.

### 5. Deployment features must match training

The production system should obtain all required model features from reliable transaction systems rather than using artificial default values.

---

# 🚀 Future Improvements

Several improvements can make the system more production-ready.

### Machine Learning

* Time-based validation
* Class imbalance handling
* Precision-Recall AUC
* Probability threshold optimization
* Calibration
* Ensemble model comparison
* SHAP explainability
* Permutation importance
* Automated retraining

### Production

* Real-time transaction scoring
* API-based prediction service
* Model monitoring
* Data drift detection
* Concept drift detection
* Model performance monitoring
* Automated retraining pipeline

### Security

* User authentication
* Role-based access
* Secure model storage
* Input validation
* Secure logging
* Protection of sensitive banking information

---

# 💡 Key Business Insight

A fraud detection model should not simply maximize accuracy.

The real objective is to find the right balance between:

```text
          Catch Fraud
              ↕
        High Recall
              ↕
      Reduce False Alarms
              ↕
        High Precision
```

The optimal operating threshold should ultimately be selected according to the **business cost of false positives and false negatives**.

---

# 🎓 What This Project Demonstrates

This project demonstrates practical knowledge of:

* Data preprocessing
* Exploratory data analysis
* Feature engineering
* Categorical encoding
* Binary classification
* Model comparison
* Random Forest
* Hyperparameter tuning
* GridSearchCV
* Cross-validation
* Classification metrics
* Feature importance
* Model serialization
* Machine learning deployment
* Streamlit development
* GitHub project management

---

# 📌 Resume Description

**Banking Fraud Detection System | Python, Scikit-learn, Random Forest, Streamlit**

> Developed an end-to-end banking fraud detection system using multiple classification algorithms and selected a tuned Random Forest model through GridSearchCV with 5-fold cross-validation. Achieved **95.15% accuracy, 82.28% precision, 78.00% recall, 80.08% F1-score and 87.80% ROC-AUC** on the held-out test set. Built an interactive Streamlit application for transaction prediction, model-performance analysis and feature-importance visualization.

---

# 👨‍💻 Author

**Raushan Kumar**

Data Analytics & Machine Learning Enthusiast

---

# ⭐ If You Found This Project Useful

If you find this project interesting, consider giving the repository a ⭐ on GitHub.

---

## Disclaimer

This project is developed for educational and portfolio purposes. It is not intended to make autonomous financial decisions or replace a production banking fraud-management system.
