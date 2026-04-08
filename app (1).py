
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
import joblib
import io

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🔍 Credit Card Fraud Detection System")
st.subheader("Big Data + Machine Learning Solution")
st.caption("**Husnian** | Big Data Analysis Course")

uploaded_file = st.file_uploader("Upload your Credit Card Transactions CSV", type="csv")

if uploaded_file is not None:
    with st.spinner("Loading data..."):
        df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Dataset Loaded: **{len(df):,}** records | Fraud Cases: **{df['Class'].sum()}**")
        
        tab1, tab2, tab3 = st.tabs(["📊 Overview", "🛠️ Train Model", "🔮 Predict Fraud"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(df['Class'].value_counts())
            with col2:
                st.write("**Fraud Percentage:**", round(df['Class'].mean()*100, 4), "%")
        
        with tab2:
            if st.button("🚀 Train Random Forest Model"):
                with st.spinner("Training model on large dataset..."):
                    X = df.drop(['Class'], axis=1)
                    y = df['Class']
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    auc = roc_auc_score(y_test, y_pred)
                    
                    st.success(f"✅ Model Trained Successfully! **AUC Score: {auc:.4f}**")
                    st.text(classification_report(y_test, y_pred))
                    
                    # Fixed Download Button
                    buffer = io.BytesIO()
                    joblib.dump(model, buffer)
                    buffer.seek(0)
                    
                    st.download_button(
                        label="📥 Download Trained Model",
                        data=buffer,
                        file_name="fraud_detection_model.pkl",
                        mime="application/octet-stream"
                    )
        
        with tab3:
            st.write("### Test a Transaction")
            st.info("Add input fields here based on your dataset columns (V1, V2, Amount, etc.)")
            st.warning("Prediction section coming in next update if needed.")

else:
    st.info("👆 Upload your credit card fraud dataset (CSV with 'Class' column)")
