
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import joblib
import io

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🔍 Credit Card Fraud Detection System")
st.subheader("Big Data + Machine Learning Solution")
st.caption("**Husnian** | Big Data Analysis Course")

uploaded_file = st.file_uploader("Upload your Credit Card Transactions CSV", type="csv")

if uploaded_file is not None:
    with st.spinner("Loading dataset..."):
        df = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Dataset Loaded: **{len(df):,}** records | Fraud Cases: **{df['Class'].sum()}**")

    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🛠️ Train Improved Model", "🔮 Predict Fraud"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df['Class'].value_counts())
        with col2:
            st.write("**Fraud Rate:**", round(df['Class'].mean()*100, 4), "%")

    # ====================== IMPROVED MODEL ======================
    with tab2:
        if st.button("🚀 Train Improved Model (Better for Fraud)"):
            with st.spinner("Training Improved Random Forest Model..."):
                X = df.drop(['Class'], axis=1)
                y = df['Class']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Improved Model with class weights
                model = RandomForestClassifier(
                    n_estimators=150,
                    random_state=42,
                    n_jobs=-1,
                    class_weight='balanced'   # This helps detect more frauds
                )
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                auc = roc_auc_score(y_test, y_pred)
                
                st.success(f"✅ **Improved Model Trained!** AUC Score: **{auc:.4f}**")
                st.text(classification_report(y_test, y_pred))
                
                # Save model
                st.session_state.model = model
                st.session_state.feature_names = X.columns.tolist()

                # Graphs
                st.write("### Model Performance Graphs")
                col1, col2 = st.columns(2)
                with col1:
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    ConfusionMatrixDisplay(cm).plot(ax=ax)
                    st.pyplot(fig)
                
                with col2:
                    importance = pd.Series(model.feature_importances_, index=X.columns)
                    fig, ax = plt.subplots()
                    importance.nlargest(10).plot(kind='barh', ax=ax)
                    plt.title("Top 10 Important Features")
                    st.pyplot(fig)

    # ====================== PREDICT FRAUD ======================
    with tab3:
        st.write("### Test a New Transaction")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train the model first!")
        else:
            st.write("Enter Transaction Details:")
            
            cols = st.columns(5)
            values = []
            for i in range(28):
                with cols[i % 5]:
                    val = st.number_input(f"V{i+1}", value=0.0, step=0.01, key=f"v{i}")
                    values.append(val)
            
            amount = st.number_input("Transaction Amount ($)", value=100.0, step=1.0)
            
            if st.button("🔍 Predict Fraud"):
                input_data = np.array([values + [amount]])
                
                prediction = st.session_state.model.predict(input_data)[0]
                prob = st.session_state.model.predict_proba(input_data)[0][1]
                
                if prediction == 1:
                    st.error(f"🚨 **HIGH RISK - FRAUD DETECTED!** (Probability: {prob:.2%})")
                else:
                    st.success(f"✅ **Normal Transaction** (Fraud Probability: {prob:.2%})")

else:
    st.info("👆 Upload your credit card fraud dataset (CSV with 'Class' column)")
