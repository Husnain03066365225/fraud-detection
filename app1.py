
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import numpy as np

st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("🔍 Credit Card Fraud Detection System")
st.subheader("Big Data + Machine Learning Solution")
st.caption("**Husnian** | Big Data Analysis Course")

# Load dataset
@st.cache_data
def load_data():
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

st.success(f"✅ Dataset Loaded: **{len(df):,} transactions** ({df['Class'].sum()} fraud cases)")

tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "🛠️ Model Training", "🔮 Fraud Prediction"])

with tab1:
    st.write("### Class Distribution (Highly Imbalanced)")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Class', ax=ax)
    st.pyplot(fig)
    
    st.write("### Transaction Amount Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df[df['Class']==0]['Amount'], bins=50, label='Normal', alpha=0.5)
    sns.histplot(df[df['Class']==1]['Amount'], bins=50, label='Fraud', alpha=0.5)
    plt.legend()
    st.pyplot(fig)

with tab2:
    if st.button("Train Fraud Detection Model"):
        with st.spinner("Training Random Forest Model on large dataset..."):
            X = df.drop(['Class', 'Time'], axis=1)
            y = df['Class']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            auc = roc_auc_score(y_test, y_pred)
            
            st.success(f"✅ Model Trained Successfully! AUC Score: **{auc:.4f}**")
            st.text(classification_report(y_test, y_pred))
            
            # Save model
            joblib.dump(model, 'fraud_model.pkl')
            st.download_button("Download Trained Model", joblib.dump(model, 'fraud_model.pkl'), file_name="fraud_model.pkl")

with tab3:
    st.write("### Real-time Fraud Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        v1 = st.number_input("V1", value=0.0)
        v2 = st.number_input("V2", value=0.0)
        amount = st.number_input("Transaction Amount", value=100.0)
    
    with col2:
        v3 = st.number_input("V3", value=0.0)
        v4 = st.number_input("V4", value=0.0)
        v5 = st.number_input("V5", value=0.0)
    
    if st.button("🔍 Predict if Fraud"):
        model = joblib.load('fraud_model.pkl')
        input_data = np.array([[v1, v2, v3, v4, v5, amount, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
        prediction = model.predict(input_data)[0]
        
        if prediction == 1:
            st.error("🚨 **FRAUD DETECTED!** High Risk Transaction")
        else:
            st.success("✅ **Normal Transaction** - Low Risk")

st.sidebar.info("This project demonstrates Big Data handling + ML for fraud prevention in fintech.")
