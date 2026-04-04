# Artificial-Intelligence-Driven-Customer-Retention-System
End-to-end telecom churn prediction system using machine learning. Includes EDA, feature engineering, RFE/PCA-based modeling, and recall-optimized evaluation. Identifies at-risk customers using behavioral trends and enables data-driven retention strategies for improved customer lifetime value.

# 📊 Telecom Churn Prediction - End-to-End ML System

## 🚀 Overview
This project builds a complete **end-to-end machine learning system** to predict customer churn in the telecom industry.  

The goal is to **identify high-risk customers early** and enable **data-driven retention strategies**.

---

## 🎯 Problem Statement
Customer churn leads to significant revenue loss.  

This project answers:
- Who is likely to churn?
- Why are they churning?
- How can we intervene early?

---

## 🧠 Key Highlights

- 📊 Exploratory Data Analysis (EDA)
- 🧹 Data Cleaning & Feature Engineering
- 🔍 Feature Selection (RFE, Correlation, VIF)
- ⚙️ Dimensionality Reduction (PCA)
- 🤖 Model Building:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - XGBoost
- 🎯 Threshold Optimization (Recall-focused)
- 📈 Model Evaluation (ROC, PR Curve, Confusion Matrix)

---

## 📂 Project Structure

```

├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── Artificial-Intelligence-Driven-Customer-Retention-System.ipynb
├── README.md
└── requirements.txt

```

---

## ⚙️ Tech Stack

- Python 🐍
- Pandas, NumPy
- Scikit-learn
- Statsmodels
- XGBoost
- Matplotlib, Seaborn

---

## 📊 Key Features Engineered

- 📉 ARPU trends (revenue decline)
- 📞 Call usage patterns
- ⏱ Recharge gap features
- 📅 Temporal behavior changes
- ⏳ Customer tenure

---

## 🧪 Model Performance

| Model | Accuracy | Recall (Churn) |
|------|----------|----------------|
| Logistic Regression | ~75% | ~82% ✅ |
| PCA + Logistic | ~76% | **~82% 🔥** |
| Gradient Boosting | ~92% | ~23% ❌ |
| XGBoost | ~92% | ~35% ❌ |

---

## 🏆 Final Model

### ✅ PCA + Logistic Regression

- High Recall (~82%)
- Stable Generalization
- Handles Multicollinearity
- Business-aligned performance

---

## 📈 Business Insights

- 📉 Declining revenue is the strongest churn signal  
- 📞 Reduced usage indicates disengagement  
- ⏱ Recharge delays are early churn indicators  
- 📅 Recent behavior matters more than historical  
- 🔄 Churn is a gradual behavioral process  

---

## 💼 Business Strategy

- 🎯 Segment users by churn risk  
- 🔴 High risk → aggressive retention  
- 🟠 Medium risk → engagement campaigns  
- 🟢 Low risk → no action  

---

## 🔄 Pipeline

```

Raw Data → Cleaning → Feature Engineering → Scaling → PCA → Model → Prediction

````

---

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook
````

---

## 📌 Future Improvements

* Deploy using FastAPI / Streamlit
* Real-time churn prediction system
* Advanced models (LightGBM, tuned XGBoost)
* Cost-sensitive learning

---

## 🧠 Key Learning

* Churn is not a sudden event — it is a gradual disengagement process.

---

## 🤝 Contributing

Feel free to fork and improve the project!

---

## 📬 Contact

For any queries or collaboration, reach out!

---

⭐ If you like this project, give it a star!
