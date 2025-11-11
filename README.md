# UCI Credit Card Default Prediction

This project provides a complete analysis of the UCI Credit Card dataset with machine learning models to predict credit card default payments.

## Files Included

1. **app.py** - Complete Streamlit web application
3. **UCI_Credit_Card.csv** - The dataset (must be in the same directory or update the path)

## Features
The interactive web application includes:

- **Data Display**: View the raw dataset
- **Descriptive Statistics**: Mean, median, standard deviation, and other statistical measures
- **Correlation Analysis**: Heatmap visualization of feature correlations
- **Machine Learning Models**: 
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
- **Model Performance Comparison**: Bar chart comparing accuracy across all three models
- **Model Evaluation**: Accuracy scores, classification reports, and confusion matrices
- **Interactive Predictions**: Sidebar inputs for custom predictions with real-time results


## Installation

Install the required packages:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Running the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`



## Dataset Information

The UCI Credit Card dataset contains 30,000 records with 25 features:

- **ID**: Customer ID
- **LIMIT_BAL**: Credit limit
- **SEX**: Gender (1=male, 2=female)
- **EDUCATION**: Education level
- **MARRIAGE**: Marital status
- **AGE**: Age in years
- **PAY_0 to PAY_6**: Payment status for past 6 months
- **BILL_AMT1 to BILL_AMT6**: Bill statement amounts for past 6 months
- **PAY_AMT1 to PAY_AMT6**: Payment amounts for past 6 months
- **default.payment.next.month**: Target variable (1=default, 0=no default)

## Model Performance

Three machine learning models are trained and evaluated on the dataset with an 80-20 train-test split:

1. **Logistic Regression**: A linear model for binary classification
2. **Random Forest**: An ensemble of decision trees
3. **XGBoost**: Gradient boosting algorithm known for high performance

The app displays a comparative bar chart showing the accuracy of all three models, allowing you to identify the best-performing model.

## Interactive Features

The Streamlit app allows you to:

1. View model performance comparison across all three models
2. Select different machine learning models for detailed analysis
3. View model performance metrics
4. Input custom customer data in the sidebar
5. Get real-time predictions with probability scores

## Author: Russell I. Lancaster

Created for UCI Credit Card Default Analysis


