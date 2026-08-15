import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Model Performance",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("📈 Model Performance")

st.markdown("""
This page summarizes the performance of the trained
Random Forest model after hyperparameter tuning.
""")

st.divider()


# ============================================================
# PERFORMANCE METRICS
# ============================================================

st.subheader("📊 Final Evaluation Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", "95.15%")
col2.metric("Precision", "82.28%")
col3.metric("Recall", "78.00%")
col4.metric("F1 Score", "80.08%")
col5.metric("ROC-AUC", "87.80%")

st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("🤖 Best Model")

st.success("""
Algorithm : Random Forest Classifier

Hyperparameter Tuning : GridSearchCV

Cross Validation : 5-Fold

Final Model : Best Estimator from GridSearchCV
""")

st.divider()


# ============================================================
# BEST PARAMETERS
# ============================================================

st.subheader("⚙️ Best Hyperparameters")

st.code("""
{
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 2
}
""")

st.divider()


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("⭐ Top Important Features")

# Get the Fraud_Detection_System folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Location of feature importance file
FEATURE_FILE = BASE_DIR / "feature_importance.csv"


# Check whether file exists
if FEATURE_FILE.exists():

    importance = pd.read_csv(FEATURE_FILE)

    # Sort features
    importance = importance.sort_values(
        by="Importance",
        ascending=True
    )

    # Show only top 20
    top_features = importance.tail(20)

    # Create chart
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 20 Important Features")

    plt.tight_layout()

    st.pyplot(fig)

else:

    st.error(
        f"Feature importance file not found:\n{FEATURE_FILE}"
    )


st.divider()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.subheader("📋 Classification Report")

report = pd.DataFrame({
    "Metric": [
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Fraud": [
        0.82,
        0.78,
        0.80
    ],

    "Legitimate": [
        0.97,
        0.98,
        0.97
    ]
})

st.dataframe(
    report,
    use_container_width=True,
    hide_index=True
)

st.divider()


# ============================================================
# MODEL SUMMARY
# ============================================================

st.subheader("🎯 Model Summary")

col1, col2 = st.columns(2)

with col1:

    st.info("""
    **Model**

    Random Forest Classifier

    **Validation**

    5-Fold Cross Validation

    **Optimization**

    GridSearchCV
    """)


with col2:

    st.info("""
    **Accuracy**

    95.15%

    **F1 Score**

    80.08%

    **ROC-AUC**

    87.80%
    """)


st.divider()


# ============================================================
# CONCLUSION
# ============================================================

st.subheader("📝 Conclusion")

st.success("""
The Random Forest model achieved 95.15% accuracy and an
ROC-AUC score of 87.80% on the test dataset.

The model was optimized using GridSearchCV with 5-fold
cross-validation and is integrated into the Streamlit
fraud detection application.
""")