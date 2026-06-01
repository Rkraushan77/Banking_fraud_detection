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
from sklearn.metrics import accuracy_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Banking Fraud Detection",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Banking Fraud Detection Dashboard")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
try:
    df = pd.read_csv("banking_transactions.csv")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
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
    st.metric("Missing Values", int(df.isnull().sum().sum()))

# --------------------------------------------------
# DATA TYPES
# --------------------------------------------------
st.subheader("Column Information")

info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum().values
})

st.dataframe(info_df)

# --------------------------------------------------
# MISSING VALUES
# --------------------------------------------------
st.subheader("Missing Values")

missing_df = pd.DataFrame(
    df.isnull().sum(),
    columns=["Missing Count"]
)

st.dataframe(missing_df)

# --------------------------------------------------
# NUMERIC & CATEGORICAL
# --------------------------------------------------
numeric_cols = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_cols = df.select_dtypes(
    exclude=np.number
).columns.tolist()

# --------------------------------------------------
# HISTOGRAM
# --------------------------------------------------
if len(numeric_cols) > 0:

    st.subheader("Histogram")

    selected_col = st.selectbox(
        "Select Numeric Column",
        numeric_cols
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        df[selected_col],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

# --------------------------------------------------
# BOXPLOT
# --------------------------------------------------
if len(numeric_cols) > 0:

    st.subheader("Boxplot")

    box_col = st.selectbox(
        "Select Column for Boxplot",
        numeric_cols,
        key="box"
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.boxplot(
        y=df[box_col],
        ax=ax
    )

    st.pyplot(fig)

# --------------------------------------------------
# CORRELATION
# --------------------------------------------------
if len(numeric_cols) > 1:

    st.subheader("Correlation Heatmap")

    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        ax=ax
    )

    st.pyplot(fig)

# --------------------------------------------------
# TARGET COLUMN
# --------------------------------------------------
st.subheader("Fraud Detection Model")

possible_targets = [
    "fraud_flag",
    "is_fraud",
    "Fraud",
    "Class",
    "target"
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

# --------------------------------------------------
# PREPROCESSING
# --------------------------------------------------
data = df.copy()

# Remove ID columns
drop_cols = []

for col in data.columns:
    if "id" in col.lower() and col != target_col:
        drop_cols.append(col)

if drop_cols:
    data.drop(columns=drop_cols, inplace=True)

# Fill missing values safely
for col in data.columns:

    if pd.api.types.is_numeric_dtype(data[col]):
        data[col] = data[col].fillna(
            data[col].median()
        )

    else:
        data[col] = data[col].fillna(
            data[col].mode()[0]
        )

# Encode categorical columns
label_encoders = {}

for col in data.select_dtypes(
    include=["object"]
).columns:

    encoder = LabelEncoder()

    data[col] = encoder.fit_transform(
        data[col].astype(str)
    )

    label_encoders[col] = encoder

# --------------------------------------------------
# FEATURES & TARGET
# --------------------------------------------------
X = data.drop(columns=[target_col])
y = data[target_col]

# Encode target if needed
if y.dtype == "object":

    target_encoder = LabelEncoder()

    y = target_encoder.fit_transform(y)

# --------------------------------------------------
# TRAIN TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------------------------------
# SCALE DATA
# --------------------------------------------------
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# TRAIN MODELS
# --------------------------------------------------
if st.button("Train Models"):

    results = []

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

    for name, model in models.items():

        if name == "KNN":

            model.fit(
                X_train_scaled,
                y_train
            )

            predictions = model.predict(
                X_test_scaled
            )

        else:

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        results.append([
            name,
            round(accuracy, 4)
        ])

    results_df = pd.DataFrame(
        results,
        columns=["Model", "Accuracy"]
    )

    st.subheader("Model Comparison")
    st.dataframe(
        results_df.sort_values(
            by="Accuracy",
            ascending=False
        )
    )

    best_model = results_df.sort_values(
        by="Accuracy",
        ascending=False
    ).iloc[0]

    st.success(
        f"Best Model: {best_model['Model']} "
        f"(Accuracy = {best_model['Accuracy']})"
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=results_df,
        x="Model",
        y="Accuracy",
        ax=ax
    )

    plt.xticks(rotation=15)

    st.pyplot(fig)

st.markdown("---")
st.caption("Banking Fraud Detection using Streamlit")
