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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
    # Using the best parameters from your notebook for demonstration
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# --- Main App UI ---
st.title("🎓 Explainable AI for Student Performance Prediction")
st.write(
    "This interactive application demonstrates how Explainable AI (XAI) can make machine learning models transparent. "
    "The underlying Random Forest model predicts whether a student will pass or fail based on demographic, social, and school-related features. "
    "Using **SHAP (SHapley Additive exPlanations)**, we can see *why* the model makes a specific prediction."
)

# --- Load and Process Data ---
df = load_data()

if df is not None:
    X_train, X_test, y_train, y_test, feature_names, X = preprocess_data(df)
    
    # Train the model
    model = train_model(X_train, y_train)

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # --- Sidebar for User Input ---
    st.sidebar.header("Explore a Prediction")
    selected_student_index = st.sidebar.selectbox(
        "Select a student from the test set to explain:",
        options=X_test.index,
        format_func=lambda x: f"Student Index {x}"
    )

    # --- Main Content Area ---
    col1, col2 = st.columns([1, 2])

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
        student_features = X.loc[[selected_student_index]].T.rename(columns={selected_student_index: 'Value'})
        st.dataframe(student_features, height=350)


    with col2:
        st.subheader("Local Explanation: Why this prediction?")
        st.write(
            "This **SHAP Force Plot** shows the features that pushed the model's prediction higher (in red) or lower (in blue). "
            "The 'base value' is the average prediction over the entire dataset."
        )
        
        # We need to find the index in the numpy array that corresponds to the student's original index
        student_iloc = X_test.index.get_loc(selected_student_index)
        
        # Generate SHAP force plot for the selected student
        # We explain the model's output for the "Pass" class (class 1)
        shap_plot = shap.force_plot(
            explainer.expected_value[1], 
            shap_values[1][student_iloc, :], 
            X_test.iloc[student_iloc, :],
            matplotlib=True,
            show=False
        )
        st.pyplot(shap_plot, bbox_inches='tight')
        st.write("---")
        st.subheader("Global Explanation: What drives the model?")
        st.write(
            "This summary plot ranks the features by their overall importance to the model's predictions across all students. "
            "Longer bars indicate more influential features."
        )
        
        # Generate SHAP summary plot
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
        st.pyplot(fig, bbox_inches='tight', clear_figure=True)


    # --- Expander for more details ---
    with st.expander("About the Model and Data"):
        st.markdown("""
        - **Model:** A `RandomForestClassifier` trained on a dataset of Portuguese students.
        - **Target:** Predicts if a student's final grade (G3) is >= 10 (Pass) or < 10 (Fail).
        - **Features:** Include student demographics (`sex`, `age`), family background (`Pstatus`, `Medu`, `Fedu`), school-related factors (`studytime`, `failures`, `absences`), and social aspects (`goout`, `health`).
        - **Explainability:** SHAP values quantify the impact of each feature on a particular prediction, providing crucial transparency.
        """)
else:
    st.info("Please upload the 'student-por.csv' dataset to proceed.")
