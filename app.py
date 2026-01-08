import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)




# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    layout="centered"
)

def load_css():
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------- Title ----------------
st.markdown("""
<div class="card">
    <h1>📉 Customer Churn Prediction</h1>
    <p>Predict whether a customer is <b>Likely to Churn</b> or <b>Likely to Stay</b></p>
</div>
""", unsafe_allow_html=True)

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    return df

df = load_data()

# ---------------- Dataset Preview ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Data Preparation ----------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ---------------- Train Model ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ---------------- Predictions ----------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ---------------- Results Table ----------------
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Churn_Probability": y_prob
})

results["Customer_Status"] = results["Churn_Probability"].apply(
    lambda x: "Likely to Churn" if x >= 0.5 else "Likely to Stay"
)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Prediction Results")
st.dataframe(results.head())
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Model Performance ----------------
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("Accuracy", f"{accuracy:.2%}")
c2.metric("Model Type", "Logistic Regression")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Confusion Matrix ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Confusion Matrix")

fig, ax = plt.subplots()
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    cmap="Blues",
    ax=ax
)
st.pyplot(fig)

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Classification Report ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Classification Report")

report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Interactive Prediction ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔮 Predict Churn for a New Customer")

sample_input = {}
for col in num_cols:
    sample_input[col] = st.number_input(
        f"{col}",
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )

sample_df = pd.DataFrame([sample_input])
sample_df[num_cols] = scaler.transform(sample_df[num_cols])

# Add missing dummy columns
sample_df = sample_df.reindex(columns=X.columns, fill_value=0)

churn_prob = model.predict_proba(sample_df)[0][1]

st.markdown(
    f"""
    <div class="prediction-box">
        Churn Probability: <b>{churn_prob:.2%}</b><br>
        Customer Status: <b>{"Likely to Churn" if churn_prob >= 0.5 else "Likely to Stay"}</b>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
