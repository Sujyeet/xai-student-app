import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for Streamlit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Student Performance Prediction - Explainable AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .error-metric {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and validate the student performance dataset."""
    try:
        df = pd.read_csv('student-por.csv', encoding='utf-8')
        df.columns = df.columns.str.strip()
        return df
    except FileNotFoundError:
        st.error("Dataset file 'student-por.csv' not found. Please ensure the file is in the application directory.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None

@st.cache_data
def preprocess_data(_df):
    """Preprocess data: create target variable, encode features, and split dataset."""
    df = _df.copy()
    
    # Create binary target variable
    df['pass'] = (df['G3'] >= 10).astype(int)
    df.drop(['G1', 'G2', 'G3'], axis=1, inplace=True)
    
    # Encode categorical features
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    X = df_encoded.drop('pass', axis=1)
    y = df_encoded['pass']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, X.columns

@st.cache_resource
def train_model(X_train, y_train):
    """Train and return the Random Forest classifier."""
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10, 
        random_state=42, 
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def initialize_explainer(_model, X_train):
    """Initialize SHAP TreeExplainer."""
    return shap.TreeExplainer(_model)

def create_shap_force_plot(explainer, shap_values, X_test, student_iloc):
    """Create SHAP force plot with proper error handling."""
    try:
        # Handle different SHAP value formats
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Binary classification with separate arrays for each class
            shap_vals_to_use = shap_values[1][student_iloc, :]
            expected_val = explainer.expected_value[1]
        else:
            # Single array format
            shap_vals_to_use = shap_values[student_iloc, :]
            expected_val = explainer.expected_value
            
        # Create the force plot
        fig = plt.figure(figsize=(12, 3))
        shap.force_plot(
            expected_val,
            shap_vals_to_use,
            X_test.iloc[student_iloc, :],
            matplotlib=True,
            show=False
        )
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating SHAP force plot: {str(e)}")
        return None

def create_shap_summary_plot(shap_values, X_test):
    """Create SHAP summary plot with error handling."""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_to_use = shap_values[1]
        else:
            shap_values_to_use = shap_values
            
        shap.summary_plot(
            shap_values_to_use, 
            X_test, 
            plot_type="bar", 
            show=False,
            max_display=15
        )
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating SHAP summary plot: {str(e)}")
        return None

# Main Application
def main():
    st.title("Student Performance Prediction with Explainable AI")
    st.markdown("""
    This application demonstrates explainable machine learning for student performance prediction. 
    The Random Forest model predicts pass/fail outcomes based on student demographics, social factors, 
    and academic history. SHAP (SHapley Additive exPlanations) provides interpretable explanations 
    for individual predictions and overall model behavior.
    """)

    # Load and process data
    df = load_data()
    if df is None:
        st.stop()

    X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)
    
    # Train model
    with st.spinner("Training model..."):
        model = train_model(X_train, y_train)
    
    # Model performance metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Initialize SHAP explainer
    with st.spinner("Initializing explainer..."):
        explainer = initialize_explainer(model, X_train)
        shap_values = explainer.shap_values(X_test)

    # Sidebar controls
    st.sidebar.header("Analysis Controls")
    st.sidebar.markdown(f"**Model Accuracy:** {accuracy:.1%}")
    
    selected_student_index = st.sidebar.selectbox(
        "Select student for individual prediction analysis:",
        options=X_test.index,
        format_func=lambda x: f"Student #{x}"
    )

    # Main content layout
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Individual Prediction Analysis")
        
        # Get prediction details
        student_data = X_test.loc[[selected_student_index]]
        prediction = model.predict(student_data)[0]
        probability = model.predict_proba(student_data)[0]
        
        # Display prediction with styled metrics
        if prediction == 1:
            st.markdown(f"""
            <div class="metric-container success-metric">
                <h4>Prediction: PASS</h4>
                <p>Pass Probability: <strong>{probability[1]:.1%}</strong></p>
                <p>Fail Probability: {probability[0]:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-container error-metric">
                <h4>Prediction: FAIL</h4>
                <p>Pass Probability: <strong>{probability[1]:.1%}</strong></p>
                <p>Fail Probability: {probability[0]:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Student Profile")
        
        # Display top features for selected student
        student_features = X_test.loc[selected_student_index]
        non_zero_features = student_features[student_features != 0].sort_values(ascending=False)
        
        if len(non_zero_features) > 0:
            st.dataframe(
                non_zero_features.to_frame().rename(columns={selected_student_index: 'Value'}),
                height=400
            )
        else:
            st.write("All feature values are 0 for this student.")

    with col2:
        st.subheader("Individual Explanation")
        st.markdown("""
        The force plot below shows how each feature influences the model's prediction for this specific student. 
        Red features push toward a positive prediction (pass), while blue features push toward negative (fail).
        """)
        
        # Create and display SHAP force plot
        student_iloc = X_test.index.get_loc(selected_student_index)
        force_plot_fig = create_shap_force_plot(explainer, shap_values, X_test, student_iloc)
        
        if force_plot_fig is not None:
            st.pyplot(force_plot_fig, clear_figure=True)
        
        st.subheader("Global Model Explanation")
        st.markdown("""
        This summary plot shows the most important features across all predictions, 
        ranked by their average impact on the model's decisions.
        """)
        
        # Create and display SHAP summary plot
        summary_plot_fig = create_shap_summary_plot(shap_values, X_test)
        
        if summary_plot_fig is not None:
            st.pyplot(summary_plot_fig, clear_figure=True)

    # Additional information
    with st.expander("Model and Dataset Information"):
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("""
            **Model Details:**
            - Algorithm: Random Forest Classifier
            - Estimators: 200 trees
            - Max Depth: 10 levels
            - Class Weight: Balanced
            """)
            
        with col_info2:
            st.markdown("""
            **Dataset Information:**
            - Source: Student performance in Portuguese course
            - Target: Pass (grade ≥ 10) vs Fail (grade < 10)
            - Features: Demographics, family, school, and social factors
            - Total Samples: {} students
            """.format(len(df)))

        st.markdown("""
        **Feature Categories:**
        - **Demographic**: Age, gender, urban/rural residence
        - **Family**: Parents' education, jobs, family relationships
        - **Academic**: Study time, past failures, course selection
        - **Social**: Free time activities, alcohol consumption, health status
        """)

if __name__ == "__main__":
    main()
