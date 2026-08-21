import streamlit as st

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# Navigation
# -------------------------------------------------
home = st.Page(
    "pages/1_🏠_Home.py",
    title="Home",
    icon="🏠"
)

dashboard = st.Page(
    "pages/2_📊_Dashboard.py",
    title="Dashboard",
    icon="📊"
)

predict = st.Page(
    "pages/3_🤖_Predict.py",
    title="Predict",
    icon="🤖"
)

model_performance = st.Page(
    "pages/4_📈_Model_Performance.py",
    title="Model Performance",
    icon="📈"
)

about = st.Page(
    "pages/5_ℹ️_About.py",
    title="About",
    icon="ℹ️"
)

pg = st.navigation(
    [home, dashboard, predict, model_performance, about],
    position="sidebar",
    expanded=True
)

pg.run()


# ============================================================
# CUSTOM LIGHT THEME
# ============================================================

st.markdown("""
<style>

    /* Main application background */
    .stApp {
        background-color: #f5f7fb;
    }

    /* Main content area */
    .main {
        background-color: #f5f7fb;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #eef2f7;
    }

    /* Cards / containers */
    div[data-testid="stMetric"] {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.05);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* Headers */
    h1, h2, h3 {
        color: #1e293b;
    }

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Main Page
# -------------------------------------------------
st.title("🛡️ Banking Fraud Detection System")

st.markdown("""
## Welcome!

This application uses **Machine Learning** to identify whether a banking transaction is **Fraudulent** or **Legitimate**.

### 👈 Use the sidebar on the left to navigate through the application.
""")

st.divider()

st.subheader("📂 Available Pages")

col1, col2 = st.columns(2)

with col1:
    st.info("""
🏠 **Home**
- Project overview
- Model performance
- Workflow
- Objective
""")

    st.info("""
📊 **Dashboard**
- Dataset analysis
- Interactive charts
- Fraud insights
""")

with col2:
    st.info("""
🤖 **Predict**
- Enter transaction details
- Predict Fraud / Legitimate
- View prediction confidence
""")

    st.info("""
📈 **Model Performance**
- Confusion Matrix
- ROC Curve
- Feature Importance
- Classification Report
""")

st.info("""
ℹ️ **About**

Learn about the dataset, machine learning workflow,
technologies used, and future improvements.
""")

st.divider()

st.success("✅ Select a page from the sidebar to begin.")
