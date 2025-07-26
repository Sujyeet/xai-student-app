# 🎓 XAI Student Performance Predictor

An interactive **Explainable AI (XAI)** application that predicts student academic performance and provides transparent explanations for why those predictions were made. Built with machine learning and powered by SHAP (SHapley Additive exPlanations), this tool demonstrates how AI can be made interpretable and trustworthy in educational contexts.

## ✨ Features

- **🤖 AI-Powered Predictions**: Uses Random Forest machine learning to predict whether students will pass or fail based on demographic, social, and academic factors
- **🔍 Explainable AI**: Provides clear, visual explanations for each prediction using SHAP values
- **📊 Interactive Dashboard**: User-friendly Streamlit web interface for exploring predictions
- **👥 Individual Analysis**: Select any student to see personalized prediction explanations
- **🌍 Global Insights**: Understand which factors are most important across all students
- **📈 Visual Analytics**: Force plots and summary plots to visualize feature impacts

## 🛠️ Technologies Used

- **Python 3.x** - Core programming language
- **Streamlit** - Interactive web application framework
- **scikit-learn** - Machine learning library (Random Forest)
- **SHAP** - Explainable AI library for model interpretability
- **Pandas** - Data manipulation and analysis
- **Matplotlib** - Data visualization
- **Jupyter Notebook** - For analysis and experimentation

## 📊 Dataset Information

The application uses the **Portuguese Student Performance Dataset** containing 649 student records with features including:

### Student Demographics
- Age, gender, urban/rural residence
- Family size and parental cohabitation status

### Educational Background
- Mother's and father's education levels
- Previous academic failures
- Extra educational support

### Social & Lifestyle Factors
- Study time, going out frequency
- Alcohol consumption (workday/weekend)
- Health status and absences

### Target Variable
- **Pass/Fail Classification**: Based on final grade G3 (≥10 = Pass, <10 = Fail)

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sujyeet/xai-student-app.git
   cd xai-student-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run XAI.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to that URL manually

## 🎯 Usage

### Running the Web Application

1. **Launch the app** using `streamlit run XAI.py`
2. **Select a student** from the dropdown in the sidebar
3. **View the prediction** - see whether the model predicts Pass or Fail
4. **Explore explanations**:
   - **Local Explanation**: SHAP force plot showing why this specific prediction was made
   - **Global Explanation**: Feature importance across all students
5. **Examine student features** in the detailed feature table

### Understanding the Explanations

- **Red bars/values**: Features that increase the likelihood of passing
- **Blue bars/values**: Features that decrease the likelihood of passing  
- **Base value**: The average prediction across all students
- **Feature importance**: Longer bars indicate more influential features

## 🧠 How It Works

### Machine Learning Pipeline
1. **Data Preprocessing**: One-hot encoding of categorical variables
2. **Feature Engineering**: Create binary pass/fail target from grades
3. **Model Training**: Random Forest Classifier with optimized hyperparameters
4. **Prediction**: Generate probability scores for pass/fail classification

### Explainable AI with SHAP
- **Individual Explanations**: For each student, SHAP calculates how much each feature contributes to moving the prediction above or below the average
- **Global Explanations**: Aggregate SHAP values show which features are most important overall
- **Force Plots**: Visualize the "forces" pushing the prediction toward pass or fail
- **Summary Plots**: Rank features by their overall impact on model decisions

## 📁 File Structure

```
xai-student-app/
├── XAI.py                          # Main Streamlit application
├── Student Analysis_XAI.ipynb      # Jupyter notebook with analysis
├── student-por.csv                 # Portuguese student dataset
├── requirements.txt                # Python dependencies
├── student-performance-xai-report.docx  # Detailed analysis report
├── README.md                       # This documentation
└── LICENSE                         # MIT License
```

## 🎓 Educational Value

This project demonstrates:
- **Responsible AI**: How to make machine learning models transparent and interpretable
- **Feature Impact Analysis**: Understanding which factors most influence student success
- **Interactive Data Science**: Combining ML with user-friendly interfaces
- **Educational Analytics**: Applying AI to understand academic performance patterns

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Submit bug reports or feature requests
- Improve documentation
- Add new visualization features
- Enhance the machine learning pipeline

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Sai Sujit Varma Gottumukkala**

---

*🌟 If you find this project helpful, please consider giving it a star!*