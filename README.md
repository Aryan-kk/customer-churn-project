📊 Customer Churn Prediction (Machine Learning + Streamlit App)

This project predicts whether a telecom customer will churn using machine learning models based on demographic, service usage, and account information.

🚀 Features

Full preprocessing pipeline (handling categorical & numerical data)

One-Hot Encoding & scaling

Trained ML models:

Random Forest (final model)

Logistic Regression (baseline)

Metrics included:

Accuracy

Recall (focus on churn customers)

Confusion Matrix

ROC-AUC Curve

Interactive Streamlit Web App for live predictions

📁 Project Structure
customer-churn-project/
│
├── data/
├── src/
│   └── train_churn_model.py
├── images/
├── app.py
├── requirements.txt
└── README.md

🔧 Tech Stack
Component	Technology
Language	Python
ML Models	RandomForest, XGBoost
App Framework	Streamlit
Visualization	Matplotlib, Seaborn
Dataset	Telco Customer Churn
▶ How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Train Model
python src/train_churn_model.py

3️⃣ Run App
streamlit run app.py

📍 Dataset

Dataset: Telco Customer Churn
Features include payment type, tenure, charges, internet services, etc.

👤 Author

Aryan Karmakar
B.Tech CSE, Gautam Buddha University
