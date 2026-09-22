# 🔐 FraudGuard - Intelligent Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Green)

**FraudGuard** is an intelligent web-based Credit Card Fraud Detection System built using Machine Learning and Big Data concepts. It helps detect fraudulent transactions in real-time and in batches.

## 🚀 Features

- **Single Transaction Prediction** – Instantly check if a transaction is fraudulent
- **Batch Prediction** – Upload CSV file and get predictions for multiple transactions
- **Fraud Probability Score** – Shows model confidence
- **Real-time Streaming Simulation** (Kafka-style)
- **SHAP Explainability** – Understand why a transaction was flagged
- **Continuous Retraining** option
- Clean and interactive Streamlit dashboard

## 🛠️ Tech Stack

- **Python**
- **Pandas & NumPy**
- **Scikit-learn** (Random Forest Classifier)
- **Streamlit** (Web App)
- **SHAP** (Model Explainability)

## 📊 How It Works

1. Upload the credit card transactions dataset
2. Train the Random Forest model (handles class imbalance using `class_weight='balanced'`)
3. Perform single or batch predictions
4. View detailed results and explanations

## 💻 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Husnain03066365225/fraud-detection.git

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

🎯 Key Learnings

Handling highly imbalanced datasets
Building end-to-end Machine Learning applications
Model evaluation (AUC Score, Classification Report)
Deploying ML models using Streamlit
Model explainability with SHAP

👤 Author
Muhammad Husnain

Computer Science Student

GitHub: Husnain03066365225

⭐ If you like this project, don't forget to give it a star!
