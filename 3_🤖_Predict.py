import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Fraud Prediction",
    page_icon="🤖",
    layout="wide"
)

# ============================================================
# PROJECT PATH
# ============================================================

# pages/3_Predict.py
#       ↓
# parent.parent = Fraud_Detection_System

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "fraud_detection_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"
FEATURE_PATH = BASE_DIR / "feature_columns.pkl"

# ============================================================
# LOAD MODEL FILES
# ============================================================

try:

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_PATH)

except Exception as e:

    st.error("❌ Unable to load the model files.")

    st.code(str(e))

    st.stop()

# ============================================================
# TITLE
# ============================================================

st.title("🤖 Banking Fraud Prediction")

st.markdown(
    """
    ### Intelligent Transaction Risk Assessment

    Enter the important transaction details below to determine
    whether the transaction is **legitimate or potentially fraudulent**.
    """
)

st.divider()

# ============================================================
# TRANSACTION INFORMATION
# ============================================================

st.subheader("💰 Transaction Information")

col1, col2 = st.columns(2)

with col1:

    transaction_amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=5000.0,
        step=100.0
    )

    transaction_time_hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    payment_channel = st.selectbox(
        "Payment Channel",
        [
            "Branch",
            "Mobile App",
            "POS Terminal",
            "Web Banking"
        ]
    )

with col2:

    transaction_velocity_score = st.slider(
        "Transaction Velocity Score",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01
    )

    geo_distance_km = st.number_input(
        "Geographical Distance (KM)",
        min_value=0.0,
        value=5.0,
        step=1.0
    )

    international_transaction = st.selectbox(
        "International Transaction",
        ["No", "Yes"]
    )

st.divider()

# ============================================================
# RISK INFORMATION
# ============================================================

st.subheader("⚠️ Risk Assessment")

col1, col2 = st.columns(2)

with col1:

    device_risk_score = st.slider(
        "Device Risk Score",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.01
    )

    anomaly_score = st.slider(
        "Anomaly Score",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01
    )

with col2:

    suspicious_ip = st.selectbox(
        "Suspicious IP Address",
        ["No", "Yes"]
    )

    login_attempts = st.number_input(
        "Login Attempts",
        min_value=0,
        value=1,
        step=1
    )

st.divider()

# ============================================================
# CUSTOMER INFORMATION
# ============================================================

st.subheader("👤 Customer Information")

col1, col2 = st.columns(2)

with col1:

    account_age_days = st.number_input(
        "Account Age (Days)",
        min_value=0,
        value=365,
        step=1
    )

with col2:

    authentication_type = st.selectbox(
        "Authentication Type",
        [
            "Biometric",
            "OTP",
            "Password Only",
            "Two-Factor Authentication"
        ]
    )

st.divider()

# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Transaction",
    use_container_width=True,
    type="primary"
)

# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # Create all model features
    # --------------------------------------------------------

    input_data = {

        "transaction_amount": transaction_amount,

        "login_attempts": login_attempts,

        "device_risk_score": device_risk_score,

        # Hidden/default feature
        "transfer_frequency": 1,

        "anomaly_score": anomaly_score,

        "account_age_days": account_age_days,

        "transaction_time_hour": transaction_time_hour,

        # Hidden/default feature
        "failed_transactions_last_30d": 0,

        # Hidden/default feature
        "avg_monthly_balance": 50000,

        # Hidden/default feature
        "daily_transaction_count": 2,

        "geo_distance_km": geo_distance_km,

        # Hidden/default feature
        "session_duration_minutes": 10,

        "transaction_velocity_score":
            transaction_velocity_score,

        # Default value
        "card_present_flag": 1,

        "international_transaction_flag":
            1 if international_transaction == "Yes" else 0,

        "suspicious_ip_flag":
            1 if suspicious_ip == "Yes" else 0,

        # One-hot encoded payment features
        "payment_channel_Mobile App": 0,

        "payment_channel_POS Terminal": 0,

        "payment_channel_Web Banking": 0,

        # One-hot encoded authentication features
        "authentication_type_OTP": 0,

        "authentication_type_Password Only": 0,

        "authentication_type_Two-Factor Authentication": 0
    }

    # --------------------------------------------------------
    # Payment Channel Encoding
    # --------------------------------------------------------

    if payment_channel == "Mobile App":

        input_data["payment_channel_Mobile App"] = 1

    elif payment_channel == "POS Terminal":

        input_data["payment_channel_POS Terminal"] = 1

    elif payment_channel == "Web Banking":

        input_data["payment_channel_Web Banking"] = 1

    # Branch = all zeros
    # This represents the reference category.

    # --------------------------------------------------------
    # Authentication Encoding
    # --------------------------------------------------------

    if authentication_type == "OTP":

        input_data["authentication_type_OTP"] = 1

    elif authentication_type == "Password Only":

        input_data["authentication_type_Password Only"] = 1

    elif authentication_type == "Two-Factor Authentication":

        input_data[
            "authentication_type_Two-Factor Authentication"
        ] = 1

    # Biometric = all zeros
    # This represents the reference category.

    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    input_df = pd.DataFrame([input_data])

    # --------------------------------------------------------
    # Check Missing Features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in input_df.columns
    ]

    if missing_features:

        st.error("❌ Some model features are missing.")

        st.write(missing_features)

        st.stop()

    # --------------------------------------------------------
    # Arrange features exactly like training
    # --------------------------------------------------------

    input_df = input_df[feature_columns]

    # --------------------------------------------------------
    # Scale Input
    # --------------------------------------------------------

    input_scaled = scaler.transform(input_df)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(input_scaled)[0]

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    if hasattr(model, "predict_proba"):

        probability = model.predict_proba(input_scaled)[0]

        fraud_probability = probability[1] * 100

        legitimate_probability = probability[0] * 100

    else:

        fraud_probability = None

        legitimate_probability = None

    st.divider()

    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("📊 Prediction Result")

    if prediction == 1:

        st.error(
            "🚨 FRAUDULENT TRANSACTION DETECTED"
        )

        st.metric(
            "Fraud Probability",
            f"{fraud_probability:.2f}%"
        )

        st.warning(
            "Risk Level: HIGH"
        )

        st.markdown(
            """
            ### 🚨 Recommended Actions

            - 🔎 Verify customer identity
            - ⏸️ Hold the transaction
            - 📞 Contact the customer
            - 👨‍💼 Perform manual investigation
            """
        )

    else:

        st.success(
            "✅ LEGITIMATE TRANSACTION"
        )

        st.metric(
            "Legitimate Probability",
            f"{legitimate_probability:.2f}%"
        )

        st.info(
            "Risk Level: LOW"
        )

        st.markdown(
            """
            ### ✅ Recommended Actions

            - Approve the transaction
            - Continue normal processing
            - Monitor future transactions
            """
        )

    # ========================================================
    # PROBABILITY DETAILS
    # ========================================================

    if fraud_probability is not None:

        st.divider()

        st.subheader("📈 Prediction Probability")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Legitimate",
                f"{legitimate_probability:.2f}%"
            )

        with col2:

            st.metric(
                "Fraud",
                f"{fraud_probability:.2f}%"
            )

        st.progress(
            int(fraud_probability)
        )

    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.divider()

    with st.expander("🔎 View Transaction Details"):

        display_data = {
            "Transaction Amount":
                transaction_amount,

            "Transaction Hour":
                transaction_time_hour,

            "Payment Channel":
                payment_channel,

            "Device Risk Score":
                device_risk_score,

            "Anomaly Score":
                anomaly_score,

            "Geo Distance (KM)":
                geo_distance_km,

            "Login Attempts":
                login_attempts,

            "Account Age (Days)":
                account_age_days,

            "Authentication":
                authentication_type,

            "International":
                international_transaction,

            "Suspicious IP":
                suspicious_ip,

            "Transaction Velocity":
                transaction_velocity_score
        }

        summary_df = pd.DataFrame(
            display_data.items(),
            columns=["Feature", "Value"]
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )