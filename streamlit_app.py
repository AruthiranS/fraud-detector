import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detector", page_icon="💳", layout="wide")

V_COLS = [f"V{i}" for i in range(1, 29)]


@st.cache_resource
def load_bundle():
    return joblib.load("fraud_model.joblib")


bundle = load_bundle()
model, THRESHOLD, FEATURES = bundle["model"], bundle["threshold"], bundle["features"]


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Same feature engineering as the notebook (Stage 3)."""
    hour = (df["Time"] // 3600) % 24
    out = df[V_COLS].copy()
    out["Amount_log"] = np.log1p(df["Amount"])
    out["Hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["Hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out[FEATURES]


st.title("💳 Credit Card Fraud Detector")
st.write(
    "Upload a CSV of card transactions (columns: `Time`, `V1`–`V28`, `Amount`). "
    "The model (Random Forest) returns a fraud probability and a FRAUD / Legit flag."
)

with st.sidebar:
    st.header("About")
    st.write("Dataset: ULB / Kaggle Credit Card Fraud (284,807 transactions, 0.17% fraud).")
    st.write(f"Decision threshold (tuned on validation set): **{THRESHOLD:.3f}**")

uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])
use_sample = st.button("Or try the built-in sample file")

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif use_sample and os.path.exists("sample_transactions.csv"):
    df = pd.read_csv("sample_transactions.csv")

if df is not None:
    missing = [c for c in ["Time", "Amount"] + V_COLS if c not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing[:5]} ...")
    else:
        proba = model.predict_proba(prepare(df))[:, 1]
        res = pd.DataFrame({
            "Amount": df["Amount"],
            "Fraud probability": proba.round(4),
            "Prediction": np.where(proba >= THRESHOLD, "FRAUD", "Legit"),
        })
        if "Class" in df.columns:
            res["Actual"] = np.where(df["Class"] == 1, "FRAUD", "Legit")

        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions", len(res))
        c2.metric("Flagged as FRAUD", int((res["Prediction"] == "FRAUD").sum()))
        c3.metric("Highest fraud probability", f"{proba.max():.3f}")

        def highlight(row):
            color = "background-color: #5c1a1a" if row["Prediction"] == "FRAUD" else ""
            return [color] * len(row)

        st.dataframe(res.style.apply(highlight, axis=1))
        st.download_button("Download results (CSV)", res.to_csv(index=False), "predictions.csv", "text/csv")
