import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Banking Fraud Detection",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Banking Fraud Detection System")
st.markdown("---")

# -----------------------------
# LOAD DATASET
# -----------------------------
try:
    df = pd.read_csv("banking_transactions.csv")

    st.success("Dataset Loaded Successfully!")

except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# -----------------------------
# DATA PREVIEW
# -----------------------------
st.subheader("Dataset Preview")

st.dataframe(df.head())

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])

with col2:
    st.metric("Columns", df.shape[1])

with col3:
    st.metric("Missing Values", df.isnull().sum().sum())

st.markdown("---")

# -----------------------------
# DATA INFORMATION
# -----------------------------
st.subheader("Dataset Information")

info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values
})

st.dataframe(info_df)

# -----------------------------
# MISSING VALUES
# -----------------------------
st.subheader("Missing Values")

missing_df = pd.DataFrame(
    df.isnull().sum(),
    columns=["Missing Count"]
)

st.dataframe(missing_df)

# -----------------------------
# NUMERICAL FEATURES
# -----------------------------
numerical_cols = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_cols = df.select_dtypes(
    include=["object"]
).columns.tolist()

# -----------------------------
# HISTOGRAM
# -----------------------------
st.subheader("Histogram")

if len(numerical_cols) > 0:

    selected_col = st.selectbox(
        "Select Numerical Feature",
        numerical_cols
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[selected_col], kde=True, ax=ax)

    ax.set_title(selected_col)

    st.pyplot(fig)

# -----------------------------
# BOXPLOT
# -----------------------------
st.subheader("Boxplot")

if len(numerical_cols) > 0:

    selected_box = st.selectbox(
        "Select Feature for Boxplot",
        numerical_cols,
        key="boxplot"
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(y=df[selected_box], ax=ax)

    st.pyplot(fig)

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
st.subheader("Correlation Heatmap")

if len(numerical_cols) > 1:

    corr = df[numerical_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=ax
    )

    st.pyplot(fig)

# -----------------------------
# TARGET COLUMN
# -----------------------------
st.markdown("---")
st.subheader("Fraud Detection Model")

possible_targets = [
    "fraud_flag",
    "Fraud",
    "is_fraud",
    "target",
    "Class"
]

target_col = None

for col in possible_targets:
    if col in df.columns:
        target_col = col
        break

if target_col is None:

    target_col = st.selectbox(
        "Select Target Column",
        df.columns
    )

st.success(f"Target Column: {target_col}")

# -----------------------------
# PREPROCESSING
# -----------------------------
data = df.copy()

# Remove ID columns if present
for col in data.columns:
    if "id" in col.lower():
        data.drop(col, axis=1, inplace=True)

# Fill missing values
for col in data.columns:

    if data[col].dtype == "object":
        data[col].fillna(data[col].mode()[0], inplace=True)

    else:
        data[col].fillna(data[col].median(), inplace=True)

# Encode categorical columns
label_encoders = {}

for col in data.select_dtypes(include="object").columns:

    le = LabelEncoder()

    data[col] = le.fit_transform(
        data[col].astype(str)
    )

    label_encoders[col] = le

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = data.drop(target_col, axis=1)
y = data[target_col]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# -----------------------------
# SCALING
# -----------------------------
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# MODEL TRAINING
# -----------------------------
if st.button("Train Models"):

    models = {
        "Logistic Regression":
            LogisticRegression(max_iter=1000),

        "KNN":
            KNeighborsClassifier(n_neighbors=5),

        "Naive Bayes":
            GaussianNB(),

        "Decision Tree":
            DecisionTreeClassifier(random_state=42)
    }

    results = []

    for name, model in models.items():

        if name == "KNN":

            model.fit(
                X_train_scaled,
                y_train
            )

            prediction = model.predict(
                X_test_scaled
            )

        else:

            model.fit(
                X_train,
                y_train
            )

            prediction = model.predict(
                X_test
            )

        accuracy = accuracy_score(
            y_test,
            prediction
        )

        results.append(
            [name, accuracy]
        )

    result_df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy"]
    )

    st.subheader("Model Comparison")

    st.dataframe(
        result_df.sort_values(
            by="Accuracy",
            ascending=False
        )
    )

    best_model = result_df.sort_values(
        by="Accuracy",
        ascending=False
    ).iloc[0]

    st.success(
        f"Best Model: {best_model['Model']} "
        f" | Accuracy = {best_model['Accuracy']:.4f}"
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=result_df,
        x="Model",
        y="Accuracy",
        ax=ax
    )

    plt.xticks(rotation=15)

    st.pyplot(fig)

st.markdown("---")
st.caption("Banking Fraud Detection Dashboard using Streamlit")
