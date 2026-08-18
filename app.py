import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="ASAL Livestock Dashboard", page_icon="🇰🇪", layout="centered")

st.title("🇰🇪 ASAL Livestock Early Warning Dashboard")
st.markdown("---")
st.write("Adjust environmental indicators below to forecast market values and mortality risks.")

# 1. Load Pre-trained Brains
@st.cache_resource
def load_models():
    reg = joblib.load('cattle_price_model.pkl')
    clf = joblib.load('mortality_risk_model.pkl')
    scaler = joblib.load('data_scaler.pkl')
    cols = joblib.load('feature_columns.pkl')
    return reg, clf, scaler, cols

try:
    reg_model, clf_model, scaler, feature_columns = load_models()
except Exception as e:
    st.error("Model files not found! Ensure your .pkl files are uploaded to the same folder on GitHub.")
    st.stop()

# 2. Create User Sidebar Inputs
st.sidebar.header("📍 Location & Current Indicators")
selected_county = st.sidebar.selectbox("Select County Location", ["Turkana", "Marsabit", "Wajir", "Garissa", "Mandera"])

vci = st.sidebar.slider("Current Vegetation Condition Index (VCI)", 5, 60, 35)
vci_lag1 = st.sidebar.slider("VCI 1 Month Ago", 5, 60, 37)
vci_lag2 = st.sidebar.slider("VCI 2 Months Ago", 5, 60, 40)

trekking_dist = st.sidebar.slider("Water Trekking Distance (KM)", 1, 40, 12)
trekking_lag1 = st.sidebar.slider("Trekking Distance 1 Month Ago (KM)", 1, 40, 10)
trekking_lag2 = st.sidebar.slider("Trekking Distance 2 Months Ago (KM)", 1, 40, 8)

maize_price = st.sidebar.number_input("Maize Price Per KG (KES)", min_value=30, max_value=200, value=75)

# Calculate derived metrics matching the feature blueprint
vci_3ma = np.mean([vci, vci_lag1, vci_lag2])

# 3. Transform Inputs to Match Feature Matrix Structure
input_data = {
    'Vegetation_Condition_Index': vci,
    'Water_Trekking_Distance_KM': trekking_dist,
    'Maize_Price_Per_KG_KES': maize_price,
    'VCI_lag_1': vci_lag1,
    'Trekking_Dist_lag_1': trekking_lag1,
    'VCI_lag_2': vci_lag2,
    'Trekking_Dist_lag_2': trekking_lag2,
    'VCI_3MA': vci_3ma
}

# Convert user interface fields into a raw dataframe row
input_df = pd.DataFrame([input_data])

# --- DYNAMIC COLUMN FIX ---
# This ensures that ALL county columns expected by your model are created,
# and sets them to 1 if it matches the user choice, otherwise 0.
for col in feature_columns:
    if col.startswith('County_'):
        expected_county_name = col.replace('County_', '')
        input_df[col] = 1 if selected_county == expected_county_name else 0

# Reindex forces the dataframe to match the EXACT order and columns of feature_columns
# Any column that might still be missing will safely be filled with 0 instead of crashing.
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Scale the final, perfectly matching data row
input_scaled = scaler.transform(input_df)


# 4. Trigger Predictive Operations
if st.button("Generate Early Warning Forecast", type="primary"):
    predicted_price = reg_model.predict(input_scaled)[0]
    risk_probability = clf_model.predict_proba(input_scaled)[0][1]
    
    st.markdown("### 📊 Live Predictive Analysis Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Predicted Average Cattle Price", value=f"KES {predicted_price:,.0f}")
        
    with col2:
        if risk_probability > 0.5:
            st.error(f"🔴 Crisis Alert: High Mortality Risk ({risk_probability:.1%})")
        else:
            st.success(f"🟢 Normal Status: Low Mortality Risk ({risk_probability:.1%})")
            
    # Add context explanations based on model weights
    if vci < 20 or trekking_dist > 15:
        st.warning("⚠️ Critical Environmental Triggers: Extended trekking intervals combined with pasture exhaustion drop market valuation curves.")
