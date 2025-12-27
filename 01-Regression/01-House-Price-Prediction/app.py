import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load Model and Feature Names
model = joblib.load('models/model.pkl')
feature_names = joblib.load('models/features.pkl')

st.title("🏡 California House Price Estimator")
st.write("Professional Machine Learning App suitable for Real Estate Analysis.")

# Create Input Fields
col1, col2 = st.columns(2)

with col1:
    longitude = st.number_input("Longitude", value=-122.23)
    latitude = st.number_input("Latitude", value=37.88)
    housing_median_age = st.number_input("Housing Median Age", value=41.0)
    total_rooms = st.number_input("Total Rooms", value=880.0)
    total_bedrooms = st.number_input("Total Bedrooms", value=129.0)

with col2:
    population = st.number_input("Population", value=322.0)
    households = st.number_input("Households", value=126.0)
    median_income = st.number_input("Median Income (Tens of Thousands)", value=8.3)
    # Simple dropdown for Ocean Proximity
    ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

if st.button("Predict Value"):
    # Create a dictionary of inputs
    input_data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
    }
    
    # Handle the One-Hot Encoding manually for the single input
    # We set all ocean columns to 0, then set the selected one to 1
    for col in feature_names:
        if 'ocean_proximity' in col:
            input_data[col] = 1 if col == f"ocean_proximity_{ocean_proximity}" else 0

    # Convert to DataFrame and ensure column order matches training
    input_df = pd.DataFrame([input_data])
    
    # Ensure all columns from training exist in input (fill missing with 0)
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
            
    # Reorder columns to match training exactly
    input_df = input_df[feature_names]

    # Predict
    prediction = model.predict(input_df)[0]
    
    st.success(f"Estimated House Value: ${prediction:,.2f}")