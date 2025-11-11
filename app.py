import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

# ---------------------------------------------------------------------
# Streamlit UI Configuration
# ---------------------------------------------------------------------
st.set_page_config(page_title="UCI Credit Card Default Prediction", layout="wide")

# Title and Introduction
st.title("UCI Credit Card Default Prediction")
st.write("""
This interactive web application analyzes the UCI Credit Card dataset to predict 
whether a customer will default on their credit card payment next month.
""")

# ---------------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv('UCI_Credit_Card.csv')
    return data

df = load_data()

# Display Data
if st.checkbox("Show raw data"):
    st.write(df)

# ---------------------------------------------------------------------
# Descriptive Statistics
# ---------------------------------------------------------------------
st.header("Descriptive Statistics")
st.write(df.describe())

# ---------------------------------------------------------------------
# Correlation Analysis
# ---------------------------------------------------------------------
st.header("Correlation Matrix")
corr = df.corr()
plt.figure(figsize=(18, 15))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
st.pyplot(plt.gcf())  # Explicitly pass current figure

# ---------------------------------------------------------------------
# Machine Learning Modeling
# ---------------------------------------------------------------------
st.header("Machine Learning Models")

# Prepare data for modeling
X = df.drop(['ID', 'default.payment.next.month'], axis=1)
y = df['default.payment.next.month']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------
# Model Performance Comparison
# ---------------------------------------------------------------------
st.header("Model Performance Comparison")

@st.cache_data
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    performance = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        performance[name] = accuracy
    return performance

model_performance = train_and_evaluate_models(X_train, y_train, X_test, y_test)
performance_df = pd.DataFrame.from_dict(model_performance, orient='index', columns=['Accuracy'])

st.bar_chart(performance_df)
st.write(performance_df.style.format({"Accuracy": "{:.2%}"}))

# ---------------------------------------------------------------------
# Individual Model Details
# ---------------------------------------------------------------------
st.header("Detailed Model Analysis")

# Model Selection
model_option = st.selectbox(
    "Choose a model for detailed analysis", 
    ["Logistic Regression", "Random Forest", "XGBoost"]
)

if model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=200)
elif model_option == "XGBoost":
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train and Evaluate Model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.subheader(f"{model_option} Model Performance")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2%}")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion Matrix
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(plt.gcf())

# ---------------------------------------------------------------------
# Interactive Prediction Section
# ---------------------------------------------------------------------
st.sidebar.header("Make a Prediction")

# Collect user input
limit_bal = st.sidebar.number_input("LIMIT_BAL", min_value=0, value=50000)
sex = st.sidebar.selectbox("SEX", [1, 2])
education = st.sidebar.selectbox("EDUCATION", [1, 2, 3, 4, 5, 6])
marriage = st.sidebar.selectbox("MARRIAGE", [1, 2, 3])
age = st.sidebar.slider("AGE", 21, 79, 35)

# Payment history sliders
pay_0 = st.sidebar.slider("PAY_0", -2, 8, 0)
pay_2 = st.sidebar.slider("PAY_2", -2, 8, 0)
pay_3 = st.sidebar.slider("PAY_3", -2, 8, 0)
pay_4 = st.sidebar.slider("PAY_4", -2, 8, 0)
pay_5 = st.sidebar.slider("PAY_5", -2, 8, 0)
pay_6 = st.sidebar.slider("PAY_6", -2, 8, 0)

# Bill amounts
bill_amt1 = st.sidebar.number_input("BILL_AMT1", value=0)
bill_amt2 = st.sidebar.number_input("BILL_AMT2", value=0)
bill_amt3 = st.sidebar.number_input("BILL_AMT3", value=0)
bill_amt4 = st.sidebar.number_input("BILL_AMT4", value=0)
bill_amt5 = st.sidebar.number_input("BILL_AMT5", value=0)
bill_amt6 = st.sidebar.number_input("BILL_AMT6", value=0)

# Payment amounts
pay_amt1 = st.sidebar.number_input("PAY_AMT1", value=0)
pay_amt2 = st.sidebar.number_input("PAY_AMT2", value=0)
pay_amt3 = st.sidebar.number_input("PAY_AMT3", value=0)
pay_amt4 = st.sidebar.number_input("PAY_AMT4", value=0)
pay_amt5 = st.sidebar.number_input("PAY_AMT5", value=0)
pay_amt6 = st.sidebar.number_input("PAY_AMT6", value=0)

# Predict button
if st.sidebar.button("Predict"):
    user_input = pd.DataFrame({
        'LIMIT_BAL': [limit_bal],
        'SEX': [sex],
        'EDUCATION': [education],
        'MARRIAGE': [marriage],
        'AGE': [age],
        'PAY_0': [pay_0],
        'PAY_2': [pay_2],
        'PAY_3': [pay_3],
        'PAY_4': [pay_4],
        'PAY_5': [pay_5],
        'PAY_6': [pay_6],
        'BILL_AMT1': [bill_amt1],
        'BILL_AMT2': [bill_amt2],
        'BILL_AMT3': [bill_amt3],
        'BILL_AMT4': [bill_amt4],
        'BILL_AMT5': [bill_amt5],
        'BILL_AMT6': [bill_amt6],
        'PAY_AMT1': [pay_amt1],
        'PAY_AMT2': [pay_amt2],
        'PAY_AMT3': [pay_amt3],
        'PAY_AMT4': [pay_amt4],
        'PAY_AMT5': [pay_amt5],
        'PAY_AMT6': [pay_amt6]
    })

    prediction = model.predict(user_input)
    prediction_proba = model.predict_proba(user_input)

    st.sidebar.subheader("Prediction Result")
    if prediction[0] == 1:
        st.sidebar.error("Default Likely: Yes")
    else:
        st.sidebar.success("Default Likely: No")

    st.sidebar.write(f"**Probability of Default:** {prediction_proba[0][1]:.2%}")
