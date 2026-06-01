import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
# LOAD DATASET
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

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
data = df.copy()

encoders = {}

categorical_cols = [
    "payment_channel",
    "authentication_type"
]

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

X = data.drop("fraud_flag", axis=1)
y = data["fraud_flag"]

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

st.success(f"Model Accuracy: {accuracy:.2%}")

# --------------------------------------------------
# FRAUD PREDICTION FORM
# --------------------------------------------------
st.header("🔍 Check Transaction")

col1, col2 = st.columns(2)

with col1:

    transaction_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=100.0
    )

    login_attempts = st.number_input(
        "Login Attempts",
        min_value=0,
        value=1
    )

    device_risk_score = st.number_input(
        "Device Risk Score",
        min_value=0.0,
        value=10.0
    )

    transfer_frequency = st.number_input(
        "Transfer Frequency",
        min_value=0,
        value=1
    )

    anomaly_score = st.number_input(
        "Anomaly Score",
        min_value=0.0,
        value=10.0
    )

    account_age_days = st.number_input(
        "Account Age (Days)",
        min_value=0,
        value=365
    )

    transaction_time_hour = st.slider(
        "Transaction Hour",
        0,
        23,
        12
    )

    failed_transactions_last_30d = st.number_input(
        "Failed Transactions (Last 30 Days)",
        min_value=0,
        value=0
    )

with col2:

    avg_monthly_balance = st.number_input(
        "Average Monthly Balance",
        min_value=0.0,
        value=5000.0
    )

    daily_transaction_count = st.number_input(
        "Daily Transaction Count",
        min_value=0,
        value=2
    )

    geo_distance_km = st.number_input(
        "Geo Distance (KM)",
        min_value=0.0,
        value=5.0
    )

    session_duration_minutes = st.number_input(
        "Session Duration (Minutes)",
        min_value=0.0,
        value=10.0
    )

    transaction_velocity_score = st.number_input(
        "Transaction Velocity Score",
        min_value=0.0,
        value=10.0
    )

    payment_channel = st.selectbox(
        "Payment Channel",
        encoders["payment_channel"].classes_
    )

    authentication_type = st.selectbox(
        "Authentication Type",
        encoders["authentication_type"].classes_
    )

    card_present_flag = st.selectbox(
        "Card Present",
        [0, 1]
    )

    international_transaction_flag = st.selectbox(
        "International Transaction",
        [0, 1]
    )

    suspicious_ip_flag = st.selectbox(
        "Suspicious IP",
        [0, 1]
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if st.button("Predict Fraud"):

    payment_channel_encoded = (
        encoders["payment_channel"]
        .transform([payment_channel])[0]
    )

    authentication_type_encoded = (
        encoders["authentication_type"]
        .transform([authentication_type])[0]
    )

    input_data = pd.DataFrame([{
        "transaction_amount": transaction_amount,
        "login_attempts": login_attempts,
        "device_risk_score": device_risk_score,
        "transfer_frequency": transfer_frequency,
        "anomaly_score": anomaly_score,
        "account_age_days": account_age_days,
        "transaction_time_hour": transaction_time_hour,
        "failed_transactions_last_30d":
            failed_transactions_last_30d,
        "avg_monthly_balance": avg_monthly_balance,
        "daily_transaction_count":
            daily_transaction_count,
        "geo_distance_km": geo_distance_km,
        "session_duration_minutes":
            session_duration_minutes,
        "transaction_velocity_score":
            transaction_velocity_score,
        "payment_channel":
            payment_channel_encoded,
        "authentication_type":
            authentication_type_encoded,
        "card_present_flag":
            card_present_flag,
        "international_transaction_flag":
            international_transaction_flag,
        "suspicious_ip_flag":
            suspicious_ip_flag
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
st.caption("Banking Fraud Detection using Random Forest")
