import streamlit as st
import pandas as pd
import joblib

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# ============================================
# LOAD MODEL
# ============================================
MODEL_PATH = "best_car_price_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ============================================
# TITLE
# ============================================
st.title("🚗 Used Car Price Prediction")
st.write("Enter car details below to predict the estimated used car price.")

st.markdown("---")

# ============================================
# INPUT FIELDS
# ============================================
brand = st.text_input("Brand", placeholder="e.g. Ford, BMW, Toyota")

fuel_type = st.selectbox(
    "Fuel Type",
    ["Gasoline", "Diesel", "Hybrid", "Electric", "Other"]
)

transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual", "CVT", "Other"]
)

accident = st.selectbox(
    "Accident History",
    ["No Accident", "Accident Reported", "Unknown"]
)

clean_title = st.selectbox(
    "Clean Title",
    ["Yes", "No", "Unknown"]
)

milage = st.number_input(
    "Mileage (in miles)",
    min_value=0,
    max_value=500000,
    value=30000,
    step=1000
)

car_age = st.number_input(
    "Car Age (in years)",
    min_value=0,
    max_value=50,
    value=3,
    step=1
)

st.markdown("---")

# ============================================
# PREDICT BUTTON
# ============================================
if st.button("Predict Price"):
    # Create input DataFrame exactly in training format
    input_df = pd.DataFrame({
        "brand": [brand],
        "fuel_type": [fuel_type],
        "transmission": [transmission],
        "accident": [accident],
        "clean_title": [clean_title],
        "milage": [milage],
        "car_age": [car_age]
    })

    # Prediction
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Car Price: ${prediction:,.2f}")

    st.subheader("Entered Car Details")
    st.dataframe(input_df)

