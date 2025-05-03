Heart Disease Prediction App
App Screenshot

A Streamlit web application that predicts heart disease risk using machine learning models, featuring comprehensive data exploration and visualization capabilities.

Features
Interactive Prediction Interface: Input patient parameters to assess heart disease risk

Model Comparison: Evaluate performance of Random Forest, Logistic Regression, and Decision Tree models

Data Visualization: Explore distributions of key health indicators

Responsive Design: Works on desktop and mobile devices

Dark Mode UI: Elegant maroon color scheme with optimal contrast

Technical Details
Framework: Streamlit

Machine Learning: Scikit-learn (Random Forest, Logistic Regression, Decision Tree)

Data Processing: Feature scaling, missing value imputation, categorical encoding

Visualization: Matplotlib, Seaborn

Deployment: Ready for Streamlit Cloud, Docker, or local hosting

Dataset
Uses the UCI Heart Disease Dataset containing:

303 patient records

13 clinical features (age, cholesterol, blood pressure, etc.)

Target variable indicating presence of heart disease

Installation
bash
git clone https://github.com/yourusername/heart-disease-predictor.git
cd heart-disease-predictor
pip install -r requirements.txt
streamlit run app.py
Usage
Navigate through different sections using the sidebar

Explore dataset statistics and visualizations

Compare model performances

Make predictions using the interactive form

Requirements
Python 3.8+

Streamlit

Pandas

Scikit-learn

Matplotlib

Seaborn

NumPy

Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss proposed changes.

