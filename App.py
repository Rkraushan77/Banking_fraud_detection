import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Banking Fraud Detection",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Banking Fraud Detection System")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("banking_transactions.csv")

try:
    df = load_data()
except Exception as e:
    st.error(f"Dataset Error: {e}")
    st.stop()

# --------------------------------------------------
# DATA PREVIEW
# --------------------------------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Fraud Cases", int(df["fraud_flag"].sum()))

# --------------------------------------------------
# FRAUD DISTRIBUTION
# --------------------------------------------------
st.subheader("Fraud Distribution")

fig, ax = plt.subplots(figsize=(5, 3))

df["fraud_flag"].value_counts().plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Fraud Flag")
ax.set_ylabel("Count")

st.pyplot(fig)

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------
st.subheader("Correlation Heatmap")

corr = df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    ax=ax
)

st.pyplot(fig)

# --------------------------------------------------
# SELECT IMPORTANT FEATURES
# --------------------------------------------------
selected_features = [
    "transaction_amount",
    "anomaly_score",
    "device_risk_score",
    "transaction_velocity_score",
    "suspicious_ip_flag",
    "international_transaction_flag"
]

X = df[selected_features]
y = df["fraud_flag"]

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

# --------------------------------------------------
# MODEL ACCURACY
# --------------------------------------------------
st.subheader("Model Performance")

st.success(
    f"Random Forest Accuracy: {accuracy:.2%}"
)

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------
st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": selected_features,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(8, 4))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
    ax=ax
)

st.pyplot(fig)

# --------------------------------------------------
# FRAUD PREDICTION FORM
# --------------------------------------------------
st.header("🔍 Check Transaction")

col1, col2 = st.columns(2)

with col1:

    transaction_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=1000.0
    )

    anomaly_score = st.number_input(
        "Anomaly Score",
        min_value=0.0,
        value=50.0
    )

    device_risk_score = st.number_input(
        "Device Risk Score",
        min_value=0.0,
        value=50.0
    )

with col2:

    transaction_velocity_score = st.number_input(
        "Transaction Velocity Score",
        min_value=0.0,
        value=50.0
    )

    suspicious_ip_flag = st.selectbox(
        "Suspicious IP Flag",
        [0, 1]
    )

    international_transaction_flag = st.selectbox(
        "International Transaction Flag",
        [0, 1]
    )

# --------------------------------------------------
# PREDICT
# --------------------------------------------------
if st.button("Predict Fraud"):

    input_data = pd.DataFrame([{
        "transaction_amount":
            transaction_amount,

        "anomaly_score":
            anomaly_score,

        "device_risk_score":
            device_risk_score,

        "transaction_velocity_score":
            transaction_velocity_score,

        "suspicious_ip_flag":
            suspicious_ip_flag,

        "international_transaction_flag":
            international_transaction_flag
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    if prediction == 1:

        st.error(
            f"⚠️ Fraudulent Transaction\n\n"
            f"Fraud Probability: {probability:.2%}"
        )

    else:

        st.success(
            f"✅ Legitimate Transaction\n\n"
            f"Fraud Probability: {probability:.2%}"
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("---")
st.caption(
    "Banking Fraud Detection using Random Forest"
)
