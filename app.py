# app.py
import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from textblob import TextBlob

st.set_page_config(page_title="DeepCSAT Demo", layout="wide")

@st.cache_resource
def load_artifacts():
    artifacts = {}
    try:
        artifacts['logreg'] = joblib.load("main/artifacts/logreg_pipeline.joblib")
    except Exception:
        artifacts['logreg'] = None
        st.error("❌ No artifacts folder found. Make sure models are saved.")
    try:
        artifacts['rf'] = joblib.load('main/artifacts/rf_pipeline.joblib')
    except Exception:
        artifacts['rf'] = None
    try:
        artifacts['xgb'] = joblib.load("main/artifacts/xgb_pipeline.joblib")
    except Exception:
        artifacts['xgb'] = None

    try:
        with open("main/artifacts/categorical_values.json", "r") as f:
            artifacts['cat_values'] = json.load(f)
    except Exception:
        artifacts['cat_values'] = {}

    try:
        artifacts['label_encoder'] = joblib.load("main/artifacts/label_encoder.joblib")
    except Exception:
        artifacts['label_encoder'] = None

    return artifacts

def simple_sentiment(text):
    if not text:
        return 0.0
    return TextBlob(text).sentiment.polarity

artifacts = load_artifacts()

st.title("DeepCSAT — Quick Demo")
st.markdown("Enter a sample case and get CSAT prediction from saved models.")

# --- Input form ---
with st.form("input_form", clear_on_submit=False):
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Customer remark (text)")
        text = st.text_area("Customer Remarks", height=140, placeholder="Type or paste a customer remark...")

    with col2:
        st.subheader("Structured inputs")
        cat_vals = artifacts.get('cat_values', {})
        agent_shift = st.selectbox("Agent Shift", options=cat_vals.get('Agent Shift', ["Day", "Night", "Rotational"]))
        tenure_bucket = st.selectbox("Tenure Bucket", options=cat_vals.get('Tenure Bucket', ["0-6m", "6-12m", "1-2y", "2+y"]))
        channel_name = st.selectbox("Channel", options=cat_vals.get('channel_name', ["Phone", "Chat", "Email"]))
        category = st.selectbox("Category", options=cat_vals.get('category', ["Unknown"]))
        sub_category = st.selectbox("Sub-category", options=cat_vals.get('Sub-category', ["Unknown"]))
        response_time_hrs = st.number_input("Response time (hrs)", min_value=0.0, step=0.1, value=4.0)
        remarks_length = len(text)
        sentiment_score = simple_sentiment(text)

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a single-row DataFrame matching training features
    X_input = pd.DataFrame([{
        'clean_remarks': text,
        'response_time_hrs': response_time_hrs,
        'remarks_length': remarks_length,
        'sentiment_score': sentiment_score,
        'Agent Shift': agent_shift,
        'Tenure Bucket': tenure_bucket,
        'channel_name': channel_name,
        'category': category,
        'Sub-category': sub_category
    }])

    st.markdown("### Input summary")
    st.write(X_input.T)

    # Choose which model to use (show only loaded models)
    models = {k:v for k,v in artifacts.items() if k in ['logreg','rf','xgb'] and v is not None}
    if not models:
        st.error("No models found in artifacts/. Save at least one pipeline as joblib before running the app.")
    else:
        model_choice = st.selectbox("Select model", options=list(models.keys()))
        model = models[model_choice]

        # Predict: note some pipelines may expect slightly different column names; by saving full pipeline we kept consistency.
        try:
            pred_proba = model.predict_proba(X_input) if hasattr(model, "predict_proba") else None
            pred = model.predict(X_input)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.stop()

        # If a label encoder was saved, decode; else assume binary (0/1)
        label_encoder = artifacts.get('label_encoder', None)
        if label_encoder is not None:
            pred_label = label_encoder.inverse_transform(pred)
        else:
            # If you used binary CSAT_label (0/1): map to Low/High
            if pred.dtype.kind in 'biu' or set(np.unique(pred)).issubset({0,1}):
                pred_label = np.where(pred==1, "High CSAT (4-5)", "Low CSAT (1-3)")
            else:
                pred_label = pred

        st.markdown("### Prediction")
        st.write(pd.Series(pred_label, name="Predicted"))

        if pred_proba is not None:
            proba_df = pd.DataFrame(pred_proba, columns=[f"prob_{c}" for c in (label_encoder.classes_ if label_encoder is not None else model.classes_)])
            st.markdown("#### Prediction probabilities")
            st.write(proba_df.T)

        # Optional: quick explanation (feature importance for tree-based models)
        try:
            if model_choice in ['rf','xgb']:
                st.markdown("#### Model feature importances (top 20)")
                importances = model.named_steps['clf'].feature_importances_
                # retrieving feature names from the preprocessor
                preproc = model.named_steps['preprocess']
                # safe extraction of tfidf + num + cat feature names
                tfidf = preproc.named_transformers_['text']
                try:
                    tfidf_names = tfidf.get_feature_names_out()
                except Exception:
                    tfidf_names = [f"tfidf_{i}" for i in range(100)]
                num_names = ['response_time_hrs','remarks_length','sentiment_score']
                try:
                    cat_names = preproc.named_transformers_['cat'].get_feature_names_out()
                    cat_names = [str(x) for x in cat_names]
                except Exception:
                    cat_names = []
                all_names = list(tfidf_names) + num_names + cat_names
                imp_df = pd.DataFrame({'feature': all_names[:len(importances)], 'importance': importances}).sort_values('importance', ascending=False).head(20)
                st.table(imp_df.reset_index(drop=True))
        except Exception:
            pass

st.markdown("---")
st.markdown("App created by DeepCSAT pipeline. Save pipelines to `artifacts/` before running.")
