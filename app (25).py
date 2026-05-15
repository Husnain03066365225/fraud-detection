
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="FraudGuard API", layout="wide")
st.title("🔐 FraudGuard Backend API")
st.caption("**Husnian** | Fraud Detection Service")

# Initialize model
if 'model' not in st.session_state:
    st.session_state.model = None

# Upload training data
uploaded_file = st.file_uploader("Upload Training Dataset (for first time)", type="csv")

if uploaded_file is not None and st.session_state.model is None:
    with st.spinner("Training model..."):
        df = pd.read_csv(uploaded_file)
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        st.session_state.model = model
        st.success("✅ Model Trained Successfully!")

# ====================== API SECTION ======================
st.subheader("API for FlutterFlow / Mobile App")

st.write("**Send POST request to this URL with JSON data**")

if st.session_state.model is not None:
    st.code("""
{
  "V1": -2.5,
  "V2": 1.8,
  "V3": -3.2,
  "Amount": 999.99
}
""", language="json")

    # Test API
    test_data = st.text_area("Test JSON Input", 
        value='{"V1": -5.0, "V2": 3.5, "V3": -6.2, "Amount": 999.99}', height=100)

    if st.button("Test Prediction"):
        import json
        data = json.loads(test_data)
        input_df = pd.DataFrame([data])
        
        pred = st.session_state.model.predict(input_df)[0]
        prob = st.session_state.model.predict_proba(input_df)[0][1]
        
        if pred == 1:
            st.error(f"🚨 FRAUD DETECTED! Probability: {prob:.2%}")
        else:
            st.success(f"✅ Normal Transaction. Probability: {prob:.2%}")
else:
    st.warning("Please upload training data and train model first.")

st.info("**Your Public API URL** will be shown after deployment.")
