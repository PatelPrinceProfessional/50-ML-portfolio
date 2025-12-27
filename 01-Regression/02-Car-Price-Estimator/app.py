import streamlit as st
import pandas as pd
import joblib
import datetime

# 1. Load the Model and Feature Names
# We use st.cache_resource so it loads only once (faster)
@st.cache_resource
def load_model_objects():
    model_path = 'models/car_price_model.pkl'
    features_path = 'models/features.pkl'
    
    try:
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        return model, feature_names
    except FileNotFoundError:
        return None, None

model, feature_names = load_model_objects()

# 2. Check if model loaded
if model is None:
    st.error("❌ Error: Could not load model. Please run 'python train.py' first.")
    st.stop()

# 3. Extract Brand List from Feature Names
# The model has features like "Brand_Maruti", "Brand_BMW". We extract just the names.
brand_features = [f for f in feature_names if "Brand_" in f]
brand_list = [f.replace("Brand_", "") for f in brand_features]
brand_list.sort() # Sort alphabetically

# 4. App Title & Layout
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("🚗 Used Car Price Estimator")
st.markdown("Enter the car details below to get an estimated selling price.")

# 5. Input Form (2 Columns)
col1, col2 = st.columns(2)

with col1:
    # User enters Year, we convert to Age later
    year = st.number_input("Year of Purchase", min_value=1990, max_value=datetime.datetime.now().year, value=2015)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission = st.radio("Transmission", ["Manual", "Automatic"])

with col2:
    brand = st.selectbox("Car Brand", brand_list)
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual", "Trustmark Dealer"])
    owner = st.selectbox("Owner History", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])

# 6. Prediction Logic
if st.button("Predict Price", type="primary"):
    
    # A. Calculate Age
    current_year = datetime.datetime.now().year
    car_age = current_year - year

    # B. Map Owner to Number (Must match train.py)
    owner_mapping = {
        'Test Drive Car': 0,
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    }
    owner_num = owner_mapping[owner]

    # C. Prepare Input Dictionary (Initialize all with 0)
    input_data = {col: 0 for col in feature_names}

    # D. Fill Numerical Values
    input_data['Kms_Driven'] = kms_driven
    input_data['Car_Age'] = car_age
    input_data['Owner'] = owner_num

    # E. Fill One-Hot Encoded Values
    # Example: If user chose "Petrol", we set "Fuel_Type_Petrol" = 1
    
    # Helper function to safely set One-Hot features
    def set_feature(prefix, value):
        col_name = f"{prefix}_{value}"
        if col_name in input_data:
            input_data[col_name] = 1
    
    set_feature("Fuel_Type", fuel_type)
    set_feature("Seller_Type", seller_type)
    set_feature("Transmission", transmission)
    set_feature("Brand", brand)

    # F. Create DataFrame and Predict
    input_df = pd.DataFrame([input_data])
    
    # Prediction
    prediction = model.predict(input_df)[0]

    # Display Result
    st.subheader(f"💰 Estimated Price: ₹ {prediction:,.2f}")
    st.info("Note: This is an AI estimate based on market data.")