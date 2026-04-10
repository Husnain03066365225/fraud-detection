
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import joblib

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🔍 Credit Card Fraud Detection System")
st.subheader("Big Data + Machine Learning Solution")
st.caption("**Husnian** | Big Data Analysis Course")

uploaded_file = st.file_uploader("Upload your Credit Card Transactions CSV", type="csv")

if uploaded_file is not None:
    with st.spinner("Loading dataset..."):
        df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Dataset Loaded: **{len(df):,}** records | Fraud Cases: **{df['Class'].sum()}**")

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🛠️ Train Model", "🔮 Predict Fraud"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df['Class'].value_counts())
        with col2:
            st.write("**Fraud Rate:**", round(df['Class'].mean()*100, 4), "%")

    # ====================== TRAIN MODEL ======================
    with tab2:
        if st.button("🚀 Train Improved Model"):
            with st.spinner("Training..."):
                X = df.drop(['Class'], axis=1)   # Keep all columns except Class
                y = df['Class']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, class_weight='balanced')
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                auc = roc_auc_score(y_test, y_pred)
                
                st.success(f"✅ Model Trained! AUC Score: **{auc:.4f}**")
                st.text(classification_report(y_test, y_pred))
                
                st.session_state.model = model
                st.session_state.feature_names = X.columns.tolist()

    # ====================== PREDICT FRAUD ======================
    with tab3:
        st.write("### Enter Transaction Details for Prediction")
        
        if 'model' not in st.session_state:
            st.warning("Please train the model first!")
        else:
            # Create input fields
            cols = st.columns(5)
            input_values = []
            
            for i in range(28):
                with cols[i % 5]:
                    val = st.number_input(f"V{i+1}", value=0.0, step=0.01, key=f"v{i+1}")
                    input_values.append(val)
            
            amount = st.number_input("Transaction Amount ($)", value=100.0, step=0.1)
            
            if st.button("🔍 Predict if this is Fraud"):
                # Create correct shape input (same as training data)
                input_data = np.zeros((1, len(st.session_state.feature_names)))
                
                # Fill V1 to V28
                for i in range(28):
                    input_data[0, i] = input_values[i]
                
                # Fill Amount (last column)
                input_data[0, -1] = amount
                
                # Predict
                prediction = st.session_state.model.predict(input_data)[0]
                probability = st.session_state.model.predict_proba(input_data)[0][1]
                
                if prediction == 1:
                    st.error(f"🚨 **FRAUD DETECTED!** (Probability: {probability:.2%})")
                else:
                    st.success(f"✅ **Normal Transaction** (Fraud Probability: {probability:.2%})")

else:
    st.info("👆 Please upload your credit card fraud CSV file")
