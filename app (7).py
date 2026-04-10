
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

uploaded_file = st.file_uploader("1. Upload Training Dataset (with 'Class' column)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Training Data Loaded: **{len(df):,}** records | Fraud Cases: **{df['Class'].sum()}**")

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🛠️ Train Model", "🔮 Predict Fraud"])

    with tab1:
        st.bar_chart(df['Class'].value_counts())

    with tab2:
        if st.button("🚀 Train Model"):
            with st.spinner("Training..."):
                X = df.drop(['Class'], axis=1)
                y = df['Class']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced')
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                auc = roc_auc_score(y_test, y_pred)
                
                st.success(f"✅ Model Trained! AUC Score: **{auc:.4f}**")
                st.text(classification_report(y_test, y_pred))
                st.session_state.model = model

    # ====================== PREDICTION ======================
    with tab3:
        st.subheader("Option A: Single Transaction Prediction")
        if 'model' in st.session_state:
            cols = st.columns(5)
            values = [st.number_input(f"V{i+1}", value=0.0, step=0.01, key=f"v{i}") for i in range(28)]
            amount = st.number_input("Transaction Amount", value=100.0)
            
            if st.button("Predict Single Transaction"):
                input_data = np.array([values + [amount]])
                pred = st.session_state.model.predict(input_data)[0]
                prob = st.session_state.model.predict_proba(input_data)[0][1]
                
                if pred == 1:
                    st.error(f"🚨 FRAUD DETECTED (Probability: {prob:.2%})")
                else:
                    st.success(f"✅ Normal Transaction (Probability: {prob:.2%})")
        else:
            st.warning("Train the model first!")

        st.divider()

        # ====================== BATCH PREDICTION ======================
        st.subheader("Option B: Batch Prediction (Upload New File)")
        batch_file = st.file_uploader("Upload New Transactions CSV for Bulk Prediction", type="csv", key="batch")

        if batch_file is not None and 'model' in st.session_state:
            test_df = pd.read_csv(batch_file)
            st.write(f"Loaded **{len(test_df)}** transactions for prediction")
            
            if st.button("Run Batch Fraud Detection"):
                with st.spinner("Predicting on all transactions..."):
                    X_test = test_df.drop(['Class'], axis=1) if 'Class' in test_df.columns else test_df
                    
                    predictions = st.session_state.model.predict(X_test)
                    probabilities = st.session_state.model.predict_proba(X_test)[:, 1]
                    
                    test_df['Predicted_Fraud'] = predictions
                    test_df['Fraud_Probability'] = probabilities
                    test_df['Risk_Level'] = test_df['Fraud_Probability'].apply(
                        lambda x: "High Risk" if x > 0.7 else "Medium Risk" if x > 0.3 else "Low Risk"
                    )
                    
                    fraud_count = predictions.sum()
                    st.success(f"**Batch Prediction Complete!** Detected **{fraud_count}** Fraud Cases")
                    
                    st.dataframe(test_df.head(15))
                    
                    # Download button
                    csv = test_df.to_csv(index=False)
                    st.download_button("📥 Download Full Prediction Report", csv, "fraud_predictions.csv", "text/csv")

else:
    st.info("👆 Start by uploading your main training dataset")
