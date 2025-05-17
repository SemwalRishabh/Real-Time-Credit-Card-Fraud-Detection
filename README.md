# 🛡️ Credit Card Fraud Detection

A complete machine learning pipeline to detect **fraudulent credit card transactions** in real-time using the highly imbalanced, industry-standard **Kaggle dataset**. This project was developed as part of the **MSA 8150 - Machine Learning for Analytics** course.

---

## 📂 Project Structure

- `Fraud_detection.ipynb` – Jupyter notebook with full pipeline: data exploration, preprocessing, model training, tuning, evaluation, and real-time simulation.
- `Fraud Detection in Credit Card Transactions (Report).pdf` – Final project report detailing methodology, results, and insights.
- `machine learning final presentation.pdf` – Presentation deck summarizing problem, modeling decisions, and business value.

---

## 🧠 Problem Statement

> Credit card fraud leads to over **$28 billion in global losses annually**. Manual review processes are slow and unscalable, making **automated real-time detection essential**.

### Project Goals
- Maximize **recall** for fraud detection while maintaining **high precision**.
- Build a solution that is **scalable, interpretable, and deployable in real-time**.

---

## 📊 Dataset Overview

- **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Fraud Cases**: 492 (≈ 0.17% — severe class imbalance)
- **Features**:
  - `Time`, `Amount` (raw)
  - `V1`–`V28`: PCA-transformed, anonymized features
- **Target**: `Class` (1 = Fraud, 0 = Legitimate)

---

## 🔧 Tools & Techniques

- **Language**: Python 3.10
- **Libraries**: `pandas`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`

### Models Used:
- Logistic Regression *(baseline)*
- Random Forest *(with hyperparameter tuning)*
- XGBoost *(for enhanced performance on tabular data)*

### Techniques:
- Log transformation of `Amount`
- Standard scaling of `Time` and `Log_Amount`
- Threshold tuning via **Precision-Recall curve**
- **3-fold Stratified Cross-Validation**
- Real-time simulation using `.predict_proba()` scoring

---

## 🚀 Results

| Model               | Precision | Recall | F1-Score | ROC-AUC |
|--------------------|-----------|--------|----------|---------|
| Logistic Regression| 0.06      | 0.90   | 0.11     | 0.927   |
| Random Forest       | 0.87      | 0.77   | 0.82     | 0.9966  |
| Tuned RF (th = 0.70)| 0.90      | 0.80   | 0.85     | 0.997   |

📌 **Final Model**: Tuned **Random Forest**, with operational threshold set to **0.70** for optimal precision-recall balance.

---

## 📈 Business Impact

- **Real-time scoring**: Transactions scored in milliseconds.
- **Decision Framework**:
  - `score > 0.85`: Automatic block
  - `0.70 < score ≤ 0.85`: Human review
  - `score ≤ 0.70`: Approve
- **Operational Benefits**:
  - Minimizes manual workload
  - Enhances fraud prevention
  - Improves customer trust and experience
- **Top predictors**: V14, V17, and scaled log-transformed `Amount`

---

## ⚠️ Limitations & Future Enhancements

- **PCA anonymization** → reduced feature interpretability.
- **Dataset covers only 48 hours** → lacks long-term seasonality or trend signals.
- **Next Steps**:
  - Integrate **temporal features** (hour of day, day of week)
  - Apply **SMOTE/ADASYN** for smarter oversampling
  - Add **autoencoder-based anomaly detection** for unseen frauds
  - Consider **ensemble models** for robustness

---

## 📄 License

This project is licensed under the **MIT License** — intended for educational and academic use.
