# 🇰🇪 ASAL Livestock Market Value & Mortality Early Warning System

A machine learning pipeline and interactive dashboard designed to forecast livestock price drops and emergency mortality risks in Kenya's Arid and Semi-Arid Lands (ASAL) counties (**Turkana, Marsabit, Wajir, Garissa, and Mandera**).

This project uses time-series feature engineering and extreme gradient boosting (XGBoost) to convert complex environmental indicators into actionable early warning insights for NGOs, policy-makers, and pastoralists.

---

---

## 📌 Project Overview & Value Proposition
During severe drought cycles in the Horn of Africa, pastoralist communities suffer devastating economic losses from sudden livestock deaths and market crashes caused by panic selling. 

This project solves this by constructing a **dual-target machine learning framework**:
1. **Market Value Predictor (Regression):** Forecasts exact average cattle prices (KES) to prevent market exploitation.
2. **Crisis State Alarm (Classification):** Flags months where herd mortality risk spikes above a critical threshold (>10%) to trigger proactive cash transfers or feed distribution.

## 🛠️ Data Strategy & Feature Engineering
Since cattle health changes over 30–90 days following environmental degradation, standard random modeling fails. This project engineers custom temporal features within geographical county boundaries:
* **Lag Features:** Shifting `Vegetation Condition Index (VCI)` and `Water Trekking Distance` by 1 and 2 months to give the models a historical context.
* **Rolling Windows:** Computing a 3-month moving average (`VCI_3MA`) to catch rapid flash droughts.
* **Out-of-Time Validation:** Validating predictive precision on an entirely unseen year (2025) to simulate an actual production rollout.

---

## 📁 Repository Structure

```text
├── app.py                      # Interactive Streamlit dashboard script
├── save_models.py              # Local training script to export serialized brains
├── requirements.txt            # Cloud platform package dependency configuration
├── README.md                   # Portfolio documentation
├── cattle_price_model.pkl      # Pre-trained XGBoost Regression weights
├── mortality_risk_model.pkl    # Pre-trained XGBoost Classification weights
├── data_scaler.pkl             # Serialized StandardScaler matrix
└── feature_columns.pkl         # JSON schema matching model feature configurations
```

---

## 💻 Local Installation & Setup

If you want to run this project locally on your machine, follow these steps:

### 1. Clone the Repository
```bash
git clone https://github.com
cd YOUR_REPOSITORY_NAME
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the Trained Model Artifacts
Make sure your `asal_livestock_drought_dataset.csv` file is inside the folder, then run:
```bash
python save_models.py
```

### 4. Boot Up the Dashboard
```bash
streamlit run app.py
```

---

## 📈 Tech Stack Used
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Deployment & UI:** Streamlit Cloud
* **Serialization:** Joblib

---
*Developed as a portfolio capstone piece highlighting spatial-temporal feature engineering and predictive risk deployment.*
