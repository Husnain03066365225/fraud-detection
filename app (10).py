
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
    st.success(f"✅ Training Data Loaded: **{len(df):,}** records | Fraud Cases: **{df['Class'].sum()}**")

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🛠️ Train Model", "🔮 Predict Fraud"])

    with tab1:
        st.bar_chart(df['Class'].value_counts())

    with tab2:
        st.write("### Train Sensitive Model (Better at catching fraud)")
        if st.button("🚀 Train Sensitive Model"):
            with st.spinner("Training more sensitive model..."):
                X = df.drop(['Class'], axis=1)
                y = df['Class']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # More sensitive model
                model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_leaf=1,
                    class_weight={0: 1, 1: 10},   # Give 10x more weight to fraud class
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                auc = roc_auc_score(y_test, y_pred)
                
                st.success(f"✅ Sensitive Model Trained! AUC Score: **{auc:.4f}**")
                st.text(classification_report(y_test, y_pred))
                
                st.session_state.model = model
                st.session_state.columns = X.columns.tolist()
                st.success("Model is now more sensitive to fraud!")

    with tab3:
        st.subheader("Predict Fraud")
        
        if 'model' not in st.session_state:
            st.warning("Train the Sensitive Model first!")
        else:
            st.write("Enter V1 to V28 and Amount:")
            
            cols = st.columns(5)
            input_dict = {}
            
            for i in range(28):
                with cols[i % 5]:
                    val = st.number_input(f"V{i+1}", value=0.0, step=0.01, key=f"v{i}")
                    input_dict[f"V{i+1}"] = val
            
            amount = st.number_input("Transaction Amount ($)", value=100.0, step=0.1)
            input_dict["Amount"] = amount
            
            if 'Time' in st.session_state.columns:
                input_dict["Time"] = 0.0
            
            if st.button("🔍 Predict Fraud"):
                input_df = pd.DataFrame([input_dict])
                input_df = input_df[st.session_state.columns]
                
                pred = st.session_state.model.predict(input_df)[0]
                prob = st.session_state.model.predict_proba(input_df)[0][1]
                
                if pred == 1:
                    st.error(f"🚨 **FRAUD DETECTED!** (Probability: {prob:.2%})")
                else:
                    st.success(f"✅ **Normal Transaction** (Fraud Probability: {prob:.2%})")

else:
    st.info("Upload your dataset to begin")
