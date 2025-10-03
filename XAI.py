# app.py
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------- Page config & theming ----------
st.set_page_config(
    page_title="Student Performance XAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light CSS tweaks for a cleaner look
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .metric-card {border:1px solid rgba(255,255,255,0.1); border-radius:16px; padding:16px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Data helpers ----------
@st.cache_data
def load_data(path: str = "student-por.csv"):
    try:
        df = pd.read_csv(path, encoding="utf-8")
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error("Dataset file 'student-por.csv' not found. Upload it in the sidebar or place it next to app.py.")
        return None

@st.cache_data
def preprocess(df: pd.DataFrame):
    df = df.copy()
    # Target: pass if G3 >= 10
    df["pass"] = (df["G3"] >= 10).astype(int)
    df.drop(["G1", "G2", "G3"], axis=1, inplace=True)

    # One-hot encode categoricals
    df_enc = pd.get_dummies(df, drop_first=True)

    X = df_enc.drop("pass", axis=1)
    y = df_enc["pass"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test

@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    return model

# ---------- Sidebar: data input ----------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload student-por.csv (optional)", type=["csv"]) 

df = load_data(uploaded) if uploaded else load_data()

st.title("🎓 Explainable AI for Student Performance Prediction")
st.caption(
    "Random Forest model to predict pass/fail with SHAP explanations."
)

if df is None:
    st.stop()

X_train, X_test, y_train, y_test = preprocess(df)
model = train_model(X_train, y_train)

# Use the *unified* SHAP API to avoid DimensionError and backend mismatches.
# This returns an Explanation object that's stable across SHAP versions.
@st.cache_resource
def build_explainers():
    # No-arg cached factory avoids hashing unhashable params (model/DataFrame)
    return shap.Explainer(model, X_train, feature_names=X_train.columns)

explainer = build_explainers()

# Precompute global explanations once (fast enough for RF, cached by Streamlit)
@st.cache_resource
def compute_global_explanation(explainer, X_test):
    return explainer(X_test)

shap_global = compute_global_explanation(explainer, X_test)

# ---------- Sidebar: choose a student ----------
st.sidebar.header("Explore a Prediction")
selected_idx = st.sidebar.selectbox(
    "Select a student from the test set to explain:",
    options=list(X_test.index),
    format_func=lambda i: f"Student Index {i}",
)

# ---------- Layout ----------
left, right = st.columns([1, 2])

with left:
    st.subheader(f"Prediction for Student {selected_idx}")
    row = X_test.loc[[selected_idx]]
    pred = model.predict(row)[0]
    proba = model.predict_proba(row)[0][1]

    # small, card-like metrics
    _c1, _c2 = st.columns(2)
    with _c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Prediction", "PASS" if pred == 1 else "FAIL")
        st.markdown('</div>', unsafe_allow_html=True)
    with _c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("P(pass)", f"{proba:.1%}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("---")
    st.subheader("Student's Key Features")
    st.dataframe(row.T.rename(columns={selected_idx: "Value"}), height=360)

with right:
    st.subheader("Local Explanation: Why this prediction?")
    st.write(
        "Waterfall plot shows top feature contributions that pushed the prediction up or down from the base value."
    )

    # Compute local explanation *only* for the selected row for speed
    shap_local = explainer(row)

    # SHAP matplotlib plots render into the current figure; manage figures explicitly
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    try:
        shap.plots.waterfall(shap_local[0], max_display=12, show=False)
        st.pyplot(fig1, bbox_inches="tight")
    except Exception as e:
        plt.close(fig1)
        st.warning(
            "Couldn't render waterfall due to a SHAP/matplotlib quirk. Falling back to force plot (static)."
        )
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        # Fallback: simple bar of local contributions
        vals = pd.Series(shap_local.values[0], index=row.columns).sort_values(key=abs, ascending=False)[:12]
        ax2.barh(vals.index[::-1], vals.values[::-1])
        ax2.set_xlabel("SHAP value (impact on model output)")
        st.pyplot(fig2, bbox_inches="tight")

    st.write("---")
    st.subheader("Global Explanation: What drives the model?")
    st.write("Bar plot of mean |SHAP| values across the test set.")

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    shap.plots.bar(shap_global, max_display=15, show=False)
    st.pyplot(fig3, bbox_inches="tight")

with st.expander("About the Model and Data"):
    st.markdown(
        """
        - **Model**: RandomForestClassifier (200 trees, max depth 10, class_weight=balanced)
        - **Target**: `pass` (1 if G3 ≥ 10, else 0)
        - **Explainability**: SHAP unified API (`shap.Explainer`) for stable plots.
        - **Tips**: If plots ever break after upgrading SHAP, pin versions (see requirements.txt).
        """
    )

# ---------- End of app ----------


# ---------------- requirements.txt ----------------
# Save the below as requirements.txt next to app.py
# (Keep versions pinned to avoid SHAP/Matplotlib backend mismatches.)

"""
streamlit==1.36.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.3.2
matplotlib==3.8.4
shap==0.44.1
"""
