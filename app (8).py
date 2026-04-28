
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🔍 Credit Card Fraud Detection System")
st.subheader("Big Data Analysis + Machine Learning")
st.caption("**Husnian** | Big Data Analysis Course")

uploaded_file = st.file_uploader("Upload Training Dataset (CSV with 'Class' column)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Training Data Loaded: **{len(df):,}** records")

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🛠️ Train Model", "🔮 Predict Fraud"])

    with tab1:
        st.bar_chart(df['Class'].value_counts())

    with tab2:
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                X = df.drop(['Class'], axis=1)
                y = df['Class']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                auc = roc_auc_score(y_test, y_pred)
                
                st.success(f"✅ Model Trained Successfully! AUC Score: **{auc:.4f}**")
                st.text(classification_report(y_test, y_pred))
                
                # Save important info
                st.session_state.model = model
                st.session_state.columns = X.columns.tolist()
                st.success("Model is ready for predictions!")

    with tab3:
        st.subheader("Predict Fraud")
        
        if 'model' not in st.session_state:
            st.warning("Please train the model first in the Train Model tab.")
        else:
            st.write("Enter values for V1 to V28 and Amount:")
            
            # Create input fields
            input_dict = {}
            cols = st.columns(5)
            
            for i in range(28):
                with cols[i % 5]:
                    val = st.number_input(f"V{i+1}", value=0.0, step=0.01, key=f"v{i}")
                    input_dict[f"V{i+1}"] = val
            
            amount = st.number_input("Transaction Amount", value=100.0, step=0.1)
            input_dict["Amount"] = amount
            
            if st.button("🔍 Predict"):
                # Create DataFrame with exact same columns as training
                input_df = pd.DataFrame([input_dict])
                
                # Reorder columns to match training
                input_df = input_df[st.session_state.columns]
                
                pred = st.session_state.model.predict(input_df)[0]
                prob = st.session_state.model.predict_proba(input_df)[0][1]
                
                if pred == 1:
                    st.error(f"🚨 **FRAUD DETECTED!** (Probability: {prob:.2%})")
                else:
                    st.success(f"✅ **Normal Transaction** (Fraud Probability: {prob:.2%})")

else:
    st.info("Please upload your training dataset first.")
