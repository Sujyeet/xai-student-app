# app.py
# To run this app:
# 1. Save this code as `app.py`.
# 2. Make sure you have a `requirements.txt` file (I'll provide this).
# 3. In your terminal, run: `streamlit run app.py`

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Performance XAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching Functions for Performance ---
@st.cache_data
def load_data():
    """Loads the student performance dataset."""
    try:
        # Assuming 'student-por.csv' is in the same directory as the app.py file.
        # If you deploy, make sure to upload this CSV to your repository.
        df = pd.read_csv('student-por.csv', encoding='utf-8')
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error("Dataset file 'student-por.csv' not found. Please place it in the same directory as the app.")
        return None

@st.cache_data
def preprocess_data(_df):
    """Preprocesses the data: creates target, encodes features, and splits."""
    df = _df.copy()
    # Create the binary target variable 'pass'
    df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    df.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)
    
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    X = df_encoded.drop('pass', axis=1)
    y = df_encoded['pass']
    
    # Split data for model training and explanation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_test, y_train, y_test, X.columns, X

@st.cache_resource
def train_model(X_train, y_train):
    """Trains the RandomForestClassifier model."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    return model

# --- Main App UI ---
plt.style.use("seaborn-v0_8")

st.markdown(
    """
    <style>
        .main {
            background: radial-gradient(circle at top, #1a2a6c, #16213e 25%, #0f172a 60%, #050816 100%);
            color: #f8fafc;
        }
        section[data-testid="stSidebar"] {
            background-color: #111827 !important;
            border-right: 1px solid rgba(148, 163, 184, 0.3);
        }
        .stApp header {background: rgba(15, 23, 42, 0.7);}
        .feature-card {
            padding: 1.2rem 1.5rem;
            border-radius: 1rem;
            background: rgba(30, 41, 59, 0.85);
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 18px 30px -25px rgba(148, 163, 184, 0.8);
        }
        .metric-card {
            padding: 0.8rem 1rem;
            border-radius: 0.9rem;
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(94, 234, 212, 0.35);
        }
        div[data-testid="stMetricValue"] {
            color: #5eead4;
            font-weight: 700;
        }
        div[data-testid="stMetricLabel"] {
            color: #cbd5f5;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-size: 0.8rem;
        }
        .student-table .stDataFrame div[role="table"] {
            border-radius: 0.75rem;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎓 Explainable AI for Student Performance Prediction")
st.caption(
    "Understand how individual student characteristics influence the probability of passing the course. "
    "Explore transparent model explanations powered by SHAP to support equitable academic decisions."
)

# --- Load and Process Data ---
df = load_data()

if df is not None:
    X_train, X_test, y_train, y_test, feature_names, X = preprocess_data(df)
    
    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate performance once so we can display summary metrics
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Initialize SHAP explainer using the modern API
    explainer = shap.TreeExplainer(model)
    shap_explanation = explainer(X_test)
    shap_pass = shap_explanation[..., 1]

    # --- Sidebar for User Input ---
    st.sidebar.header("Explore a Prediction")
    selected_student_index = st.sidebar.selectbox(
        "Select a student from the test set to explain:",
        options=X_test.index,
        format_func=lambda x: f"Student Index {x}"
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Snapshot")
    st.sidebar.write(
        "The model is a 200-tree Random Forest with class balancing to better capture students who might fail."
    )

    # --- Main Content Area ---
    st.subheader("Model Performance")
    perf_cols = st.columns(5)
    metrics = [
        ("Accuracy", f"{accuracy:.2%}"),
        ("Precision", f"{precision:.2%}"),
        ("Recall", f"{recall:.2%}"),
        ("F1 Score", f"{f1:.2%}"),
        ("ROC AUC", f"{roc_auc:.2f}"),
    ]
    for col, (label, value) in zip(perf_cols, metrics):
        with col:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.metric(label=label, value=value)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""<div class='feature-card'>""", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.subheader("Prediction for Student {}".format(selected_student_index))

        # Get prediction and probability
        prediction = model.predict(X_test.loc[[selected_student_index]])[0]
        probability = model.predict_proba(X_test.loc[[selected_student_index]])[0][1]

        if prediction == 1:
            st.success(f"**Prediction: PASS** (Probability: {probability:.2%})")
        else:
            st.error(f"**Prediction: FAIL** (Probability of Passing: {probability:.2%})")
        
        st.write("---")
        st.subheader("Student's Key Features:")

        # Display the features of the selected student
        student_features = (
            X.loc[[selected_student_index]]
            .T.rename(columns={selected_student_index: "Value"})
            .sort_values(by="Value", ascending=False)
        )
        st.markdown("<div class='student-table'>", unsafe_allow_html=True)
        st.dataframe(student_features, height=380, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


    with col2:
        st.subheader("Local Explanation: Why this prediction?")
        st.write(
            "This **SHAP Force Plot** shows the features that pushed the model's prediction higher (in red) or lower (in blue). "
            "The 'base value' is the average prediction over the entire dataset."
        )
        
        # We need to find the index in the numpy array that corresponds to the student's original index
        student_iloc = X_test.index.get_loc(selected_student_index)

        # Generate SHAP waterfall plot for the selected student
        waterfall_ax = shap.plots.waterfall(
            shap_pass[student_iloc],
            max_display=12,
            show=False,
        )
        st.pyplot(waterfall_ax.figure, use_container_width=True)
        plt.close(waterfall_ax.figure)
        st.write("---")
        st.subheader("Global Explanation: What drives the model?")
        st.write(
            "This summary plot ranks the features by their overall importance to the model's predictions across all students. "
            "Longer bars indicate more influential features."
        )

        # Generate SHAP summary and beeswarm plots
        bar_ax = shap.plots.bar(shap_pass, max_display=12, show=False)
        st.pyplot(bar_ax.figure, use_container_width=True)
        plt.close(bar_ax.figure)

        beeswarm_ax = shap.plots.beeswarm(shap_pass, max_display=15, show=False)
        st.pyplot(beeswarm_ax.figure, use_container_width=True)
        plt.close(beeswarm_ax.figure)


    st.markdown("""</div>""", unsafe_allow_html=True)

    st.divider()

    # --- Expander for more details ---
    with st.expander("About the Model and Data"):
        st.markdown(
            """
            - **Model:** A `RandomForestClassifier` trained on Portuguese secondary school student data.
            - **Target:** Predicts if a student's final grade (G3) is >= 10 (Pass) or < 10 (Fail).
            - **Features:** Include student demographics (`sex`, `age`), family background (`Pstatus`, `Medu`, `Fedu`), school-related factors (`studytime`, `failures`, `absences`), and social aspects (`goout`, `health`).
            - **Explainability:** SHAP values quantify the impact of each feature on a particular prediction, providing crucial transparency for data-informed interventions.
            """
        )

        st.markdown("### Data Preview")
        st.dataframe(df.head(10))
else:
    st.info("Please upload the 'student-por.csv' dataset to proceed.")
