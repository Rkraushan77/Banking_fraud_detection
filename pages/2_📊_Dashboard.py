import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------
# Page Configuration
# -----------------------------------------------------

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------
# Load Dataset
# -----------------------------------------------------

df = pd.read_csv(r"C:\Users\Admin\Desktop\Banking Ml project\Fraud_Detection_System\banking_transactions.csv")
st.title("📊 Banking Fraud Analytics Dashboard")

st.markdown(
"""
Explore the banking transaction dataset through interactive
statistics and visualizations.
"""
)

st.divider()

# =====================================================
# KPI Cards
# =====================================================

total_transactions = len(df)
fraud_transactions = df["fraud_flag"].sum()
legitimate_transactions = total_transactions - fraud_transactions
fraud_rate = (fraud_transactions / total_transactions) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Fraud Transactions", f"{fraud_transactions:,}")
col3.metric("Legitimate", f"{legitimate_transactions:,}")
col4.metric("Fraud Rate", f"{fraud_rate:.2f}%")

st.divider()

# =====================================================
# Sidebar Filter
# =====================================================

st.sidebar.header("🔎 Filters")

payment_filter = st.sidebar.multiselect(
    "Payment Channel",
    options=df["payment_channel"].unique(),
    default=df["payment_channel"].unique()
)

auth_filter = st.sidebar.multiselect(
    "Authentication Type",
    options=df["authentication_type"].unique(),
    default=df["authentication_type"].unique()
)

filtered_df = df[
    (df["payment_channel"].isin(payment_filter)) &
    (df["authentication_type"].isin(auth_filter))
]

# =====================================================
# Fraud Distribution
# =====================================================

st.subheader("🚨 Fraud vs Legitimate Transactions")

fig, ax = plt.subplots(figsize=(5,5))

filtered_df["fraud_flag"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    labels=["Legitimate","Fraud"],
    ax=ax
)

ax.set_ylabel("")

st.pyplot(fig)

st.divider()

# =====================================================
# Payment Channel Analysis
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("💳 Payment Channel")

    fig, ax = plt.subplots(figsize=(6,4))

    filtered_df["payment_channel"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    plt.xticks(rotation=20)

    st.pyplot(fig)

with col2:

    st.subheader("🔐 Authentication Type")

    fig, ax = plt.subplots(figsize=(6,4))

    filtered_df["authentication_type"].value_counts().plot(
        kind="bar",
        ax=ax
    )

    plt.xticks(rotation=20)

    st.pyplot(fig)

st.divider()

# =====================================================
# Transaction Amount Distribution
# =====================================================

st.subheader("💰 Transaction Amount Distribution")

fig, ax = plt.subplots(figsize=(8,4))

ax.hist(
    filtered_df["transaction_amount"],
    bins=30
)

ax.set_xlabel("Transaction Amount")
ax.set_ylabel("Frequency")

st.pyplot(fig)

st.divider()

# =====================================================
# Device Risk Score
# =====================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("📱 Device Risk Score")

    fig, ax = plt.subplots(figsize=(6,4))

    ax.hist(
        filtered_df["device_risk_score"],
        bins=20
    )

    st.pyplot(fig)

with col2:

    st.subheader("⚠️ Anomaly Score")

    fig, ax = plt.subplots(figsize=(6,4))

    ax.hist(
        filtered_df["anomaly_score"],
        bins=20
    )

    st.pyplot(fig)

st.divider()

# =====================================================
# Correlation Heatmap
# =====================================================

st.subheader("📈 Correlation Heatmap")

corr = filtered_df.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(12,8))

heatmap = ax.imshow(corr)

ax.set_xticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=90)

ax.set_yticks(range(len(corr.columns)))
ax.set_yticklabels(corr.columns)

plt.colorbar(heatmap)

st.pyplot(fig)

st.divider()

# =====================================================
# Dataset Preview
# =====================================================

st.subheader("📄 Dataset Preview")

st.dataframe(filtered_df.head(20), use_container_width=True)
