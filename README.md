# 🛡️ Insurance Claim Approval Prediction

This project is a **Machine Learning pipeline** for predicting whether an **insurance claim** will be approved or not, based on customer and incident details. The model is deployed via a simple, interactive **Streamlit** web app.

---

## 🚀 Project Highlights

- Built a supervised ML model (Random Forest) to classify insurance claims.
- Balanced imbalanced classes using **SMOTE**.
- Preprocessed both numerical and categorical data.
- Deployed an interactive **Streamlit app** for real-time predictions.
- Handled feature alignment and model scaling for production-readiness.

---

## 📊 Dataset Overview

The dataset includes:

- Customer demographics: `age`, `education`, `occupation`, etc.
- Policy details: `policy_csl`, `annual_premium`, `umbrella_limit`
- Claim incident data: `incident_type`, `collision_type`, `claim_amounts`
- Target column: `fraud_reported` (used as proxy for claim approval)

> Note: This project assumes that fraud prediction strongly correlates with claim approval decisions.

---

## 🧠 Model Pipeline

1. **Data Cleaning**: Handled missing values and dropped unused features.
2. **Label Encoding**: Converted categorical features to numeric.
3. **Feature Scaling**: Used `StandardScaler` for normalization.
4. **SMOTE**: Balanced the dataset to address class imbalance.
5. **Modeling**: Random Forest Classifier for robust performance.
6. **Evaluation**: Confusion Matrix, Precision, Recall, Accuracy.
7. **Deployment**: Streamlit app for live predictions.

---

## 🛠️ Tech Stack

- `Python`
- `scikit-learn`
- `pandas`, `numpy`
- `Streamlit` (for UI)
- `SMOTE` from `imblearn`
- `pickle` for model serialization

---

## 💻 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/insurance-claim-prediction.git
cd insurance-claim-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
