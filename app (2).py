
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

# File Upload
uploaded_file = st.file_uploader("Upload your Credit Card Transactions CSV file", type="csv")

if uploaded_file is not None:
    with st.spinner("Loading dataset..."):
        df = pd.read_csv(uploaded_file)
        
    st.success(f"✅ Dataset Loaded: **{len(df):,}** transactions | Fraud Cases: **{df['Class'].sum()}**")

    tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🛠️ Train Model", "🔮 Predict Fraud"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df['Class'].value_counts())
        with col2:
            st.write("**Fraud Percentage:**", round(df['Class'].mean()*100, 4), "%")

    # ====================== TRAIN MODEL ======================
    with tab2:
        if st.button("🚀 Train Random Forest Model"):
            with st.spinner("Training model... This may take 30-60 seconds"):
                X = df.drop(['Class'], axis=1)
                y = df['Class']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                auc = roc_auc_score(y_test, y_pred)
                
                st.success(f"✅ Model Trained Successfully! **AUC Score: {auc:.4f}**")
                st.text(classification_report(y_test, y_pred))
                
                # Save model in session state
                st.session_state.model = model
                st.success("Model is ready for prediction!")

    # ====================== PREDICT FRAUD ======================
    with tab3:
        st.write("### Test a New Transaction")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train the model first in the 'Train Model' tab")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                v1 = st.number_input("V1", value=0.0, step=0.01)
                v2 = st.number_input("V2", value=0.0, step=0.01)
                v3 = st.number_input("V3", value=0.0, step=0.01)
                v4 = st.number_input("V4", value=0.0, step=0.01)
                v5 = st.number_input("V5", value=0.0, step=0.01)
                amount = st.number_input("Transaction Amount ($)", value=100.0, step=1.0)
            
            with col2:
                v6 = st.number_input("V6", value=0.0, step=0.01)
                v7 = st.number_input("V7", value=0.0, step=0.01)
                v8 = st.number_input("V8", value=0.0, step=0.01)
                v9 = st.number_input("V9", value=0.0, step=0.01)
                v10 = st.number_input("V10", value=0.0, step=0.01)
            
            if st.button("🔍 Check if this Transaction is Fraud"):
                # Create input array (matching dataset structure)
                input_data = np.zeros((1, 30))  # 30 features (V1-V28 + Amount + Time)
                input_data[0, 0] = v1
                input_data[0, 1] = v2
                input_data[0, 2] = v3
                input_data[0, 3] = v4
                input_data[0, 4] = v5
                input_data[0, 5] = v6
                input_data[0, 6] = v7
                input_data[0, 7] = v8
                input_data[0, 8] = v9
                input_data[0, 9] = v10
                input_data[0, 29] = amount   # Amount column
                
                prediction = st.session_state.model.predict(input_data)[0]
                probability = st.session_state.model.predict_proba(input_data)[0][1]
                
                if prediction == 1:
                    st.error(f"🚨 **FRAUD DETECTED!** (Probability: {probability:.2%})")
                else:
                    st.success(f"✅ **Normal Transaction** (Fraud Probability: {probability:.2%})")

else:
    st.info("👆 Please upload your credit card transactions CSV file to begin.")
    st.markdown("**Recommended Dataset:** Kaggle Credit Card Fraud Detection")
