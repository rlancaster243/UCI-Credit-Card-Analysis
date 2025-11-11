import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support
)
import xgboost as xgb

# =====================================================================================
# STREAMLIT CONFIG
# =====================================================================================
st.set_page_config(page_title="UCI Credit Card Default Prediction", layout="wide")

st.title("UCI Credit Card Default Prediction")
st.write("""
This interactive web application analyzes the UCI Credit Card dataset and 
builds machine learning models to predict whether a customer will default 
on their credit card payment next month.
""")

# =====================================================================================
# DATA LOADING
# =====================================================================================
@st.cache_data
def load_data():
    return pd.read_csv("UCI_Credit_Card.csv")

df = load_data()

# =====================================================================================
# SIDEBAR: USER INPUT FORM
# =====================================================================================
with st.sidebar:
    st.header("Make a Prediction")
    st.markdown("Provide the customer's profile and recent payment history:")

    # --- High-level profile ---
    limit_bal = st.number_input(
        "Total credit limit on this card",
        min_value=0,
        value=50000,
        step=1000
    )

    gender_label = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )
    sex = 1 if gender_label == "Male" else 2  # dataset: 1=male, 2=female

    education_map = {
        "Graduate school": 1,
        "University": 2,
        "High school": 3,
        "Other / unknown": 4
    }
    education_label = st.selectbox(
        "Highest education level",
        list(education_map.keys())
    )
    education = education_map[education_label]

    marriage_map = {
        "Married": 1,
        "Single": 2,
        "Other / unknown": 3
    }
    marriage_label = st.selectbox(
        "Marital status",
        list(marriage_map.keys())
    )
    marriage = marriage_map[marriage_label]

    age = st.slider("Age (years)", 21, 79, 35)

    # --- Payment status (how late they were) ---
    st.markdown("**Recent payment status (months past due)**")
    st.caption("Negative = paid in full / on time, positive = months behind schedule.")
    pay_0 = st.slider("Most recent month", -2, 8, 0)
    pay_2 = st.slider("2 months ago", -2, 8, 0)
    pay_3 = st.slider("3 months ago", -2, 8, 0)
    pay_4 = st.slider("4 months ago", -2, 8, 0)
    pay_5 = st.slider("5 months ago", -2, 8, 0)
    pay_6 = st.slider("6 months ago", -2, 8, 0)

    # --- Statement balances ---
    st.markdown("**Credit card statement balances**")
    bill_amt1 = st.number_input("Latest statement balance", value=0)
    bill_amt2 = st.number_input("Statement balance 1 month ago", value=0)
    bill_amt3 = st.number_input("Statement balance 2 months ago", value=0)
    bill_amt4 = st.number_input("Statement balance 3 months ago", value=0)
    bill_amt5 = st.number_input("Statement balance 4 months ago", value=0)
    bill_amt6 = st.number_input("Statement balance 5 months ago", value=0)

    # --- Payments made ---
    st.markdown("**Payments made toward the card**")
    pay_amt1 = st.number_input("Payment on latest statement", value=0)
    pay_amt2 = st.number_input("Payment 1 month ago", value=0)
    pay_amt3 = st.number_input("Payment 2 months ago", value=0)
    pay_amt4 = st.number_input("Payment 3 months ago", value=0)
    pay_amt5 = st.number_input("Payment 4 months ago", value=0)
    pay_amt6 = st.number_input("Payment 5 months ago", value=0)

    predict_button = st.button("Predict default risk")

# =====================================================================================
# BASIC EXPLORATION
# =====================================================================================
if st.checkbox("Show raw data"):
    st.write(df)

st.header("Descriptive Statistics")
st.write(df.describe())

st.header("Correlation Matrix")
corr = df.corr()
plt.figure(figsize=(16, 12))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
plt.tight_layout()
st.pyplot(plt.gcf())
plt.close()

# =====================================================================================
# MODELING SETUP
# =====================================================================================
st.header("Machine Learning Models")

# Features and target
X = df.drop(["ID", "default.payment.next.month"], axis=1)
y = df["default.payment.next.month"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Class imbalance handling: compute positive class weight for XGBoost
pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

@st.cache_data
def train_and_evaluate_models(X_train, y_train, X_test, y_test, pos_weight):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=500, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
        "XGBoost": xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=float(pos_weight),
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42
        )
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, pos_label=1, average="binary"
        )
        roc = roc_auc_score(y_test, y_prob)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "precision_default": prec,
            "recall_default": rec,
            "f1_default": f1,
            "roc_auc": roc
        }

    return results

model_results = train_and_evaluate_models(X_train, y_train, X_test, y_test, pos_weight)

# Build performance table
performance_df = pd.DataFrame({
    name: {
        "Accuracy": res["accuracy"],
        "Recall (Default)": res["recall_default"],
        "F1 (Default)": res["f1_default"],
        "ROC AUC": res["roc_auc"]
    }
    for name, res in model_results.items()
}).T

st.header("Model Performance Comparison")

st.subheader("ROC AUC by Model")
st.bar_chart(performance_df[["ROC AUC"]])

st.subheader("Summary Metrics")
st.dataframe(
    performance_df.style.format({
        "Accuracy": "{:.2%}",
        "Recall (Default)": "{:.2%}",
        "F1 (Default)": "{:.2%}",
        "ROC AUC": "{:.3f}"
    })
)

# =====================================================================================
# DETAILED MODEL ANALYSIS
# =====================================================================================
st.header("Detailed Model Analysis")

model_option = st.selectbox(
    "Choose a model for detailed analysis",
    ["Logistic Regression", "Random Forest", "XGBoost"]
)

selected = model_results[model_option]
model = selected["model"]

# Base probabilities
y_prob = model.predict_proba(X_test)[:, 1]

st.subheader(f"{model_option} Model Performance (Threshold Control)")

threshold = st.slider(
    "Decision threshold for flagging a customer as 'likely to default'",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.01
)

y_pred_thresh = (y_prob >= threshold).astype(int)

acc_t = accuracy_score(y_test, y_pred_thresh)
prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(
    y_test, y_pred_thresh, pos_label=1, average="binary"
)

st.write(f"**Accuracy:** {acc_t:.2%}")
st.write(f"**Precision (customers predicted to default):** {prec_t:.2%}")
st.write(f"**Recall (actual defaulters caught):** {rec_t:.2%}")
st.write(f"**F1-score (Default class):** {f1_t:.2%}")
st.write(f"**ROC AUC (threshold-free):** {selected['roc_auc']:.3f}")

st.text("Classification Report (using selected threshold):")
st.text(classification_report(y_test, y_pred_thresh))

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred_thresh)
cm_labels = ["No Default (0)", "Default (1)"]

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=cm_labels,
    yticklabels=cm_labels,
    square=True,
    cbar_kws={"shrink": 0.75}
)
plt.title(f"{model_option} Confusion Matrix (Threshold = {threshold:.2f})",
          fontsize=12, fontweight="bold", pad=10)
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.tight_layout()
st.pyplot(plt.gcf())
plt.close()

# =====================================================================================
# USE SIDEBAR INPUT FOR PREDICTION
# =====================================================================================
if predict_button:
    user_input = pd.DataFrame({
        "LIMIT_BAL": [limit_bal],
        "SEX": [sex],
        "EDUCATION": [education],
        "MARRIAGE": [marriage],
        "AGE": [age],
        "PAY_0": [pay_0],
        "PAY_2": [pay_2],
        "PAY_3": [pay_3],
        "PAY_4": [pay_4],
        "PAY_5": [pay_5],
        "PAY_6": [pay_6],
        "BILL_AMT1": [bill_amt1],
        "BILL_AMT2": [bill_amt2],
        "BILL_AMT3": [bill_amt3],
        "BILL_AMT4": [bill_amt4],
        "BILL_AMT5": [bill_amt5],
        "BILL_AMT6": [bill_amt6],
        "PAY_AMT1": [pay_amt1],
        "PAY_AMT2": [pay_amt2],
        "PAY_AMT3": [pay_amt3],
        "PAY_AMT4": [pay_amt4],
        "PAY_AMT5": [pay_amt5],
        "PAY_AMT6": [pay_amt6]
    })

    user_proba = model.predict_proba(user_input)[0, 1]
    user_pred = int(user_proba >= threshold)

    with st.sidebar:
        st.subheader("Prediction Result")
        if user_pred == 1:
            st.error("Customer is **likely to default** on next payment.")
        else:
            st.success("Customer is **unlikely to default** on next payment.")

        st.write(f"Estimated probability of default: **{user_proba:.2%}**")
        st.write(f"Decision threshold currently in use: **{threshold:.2f}**")
