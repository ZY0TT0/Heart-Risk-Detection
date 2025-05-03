# Heart Disease Prediction App

## 🚀 Features

### 🔍 Interactive Tools
- **Risk Prediction Interface**: Input patient parameters to get instant heart disease risk assessment
- **Model Performance Dashboard**: Compare accuracy metrics of multiple ML algorithms
- **Data Explorer**: Interactive visualizations of key health indicators

### 💻 Technical Highlights
- **Responsive Design**: Fully functional on all device sizes
- **Dark Mode UI**: Eye-friendly maroon theme with optimal contrast
- **Real-time Processing**: Instant predictions with model explanations

## 🛠 Technical Stack

| Category          | Technologies Used |
|-------------------|-------------------|
| Framework         | Streamlit         |
| Machine Learning  | Scikit-learn      |
| Data Processing   | Pandas, NumPy     |
| Visualization     | Matplotlib, Seaborn |
| Deployment        | Streamlit Cloud   |

## 📊 Dataset
**UCI Heart Disease Dataset** contains:
- 303 anonymized patient records
- 13 clinical features including:
  - Demographic data (age, sex)
  - Medical measurements (BP, cholesterol)
  - Diagnostic results (ECG, exercise test)
- Binary target variable (presence/absence of heart disease)

## 🛠 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/heart-disease-predictor.git

# Navigate to project directory
cd heart-disease-predictor

# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run app.py
