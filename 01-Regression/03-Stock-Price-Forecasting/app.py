import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# 1. Page Configuration
st.set_page_config(page_title="Stock Price AI", page_icon="📈", layout="wide")

# 2. Load Model & Data
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('models/stock_model.pkl')
        features = joblib.load('models/features.pkl')
        return model, features
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_data():
    try:
        # Load the CSV to show historical trends
        df = pd.read_csv('dataset/tata_stock.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        return None

model, feature_names = load_resources()
df = load_data()

# 3. App Header
st.title("📈 Tata Motors Stock Price Forecaster")
st.markdown("This AI model uses **Linear Regression** to predict the *Next Day's Closing Price* based on today's market behavior.")

# 4. Historical Data Visualization
if df is not None:
    st.subheader("📊 Historical Stock Trend")
    
    # Create interactive line chart
    fig = px.line(df, x='Date', y='Close', title='Tata Motors Close Price History (1995-2025)')
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

    # Show latest data points to help user
    st.subheader("📋 Latest Market Data")
    st.dataframe(df.tail(5).sort_values('Date', ascending=False), use_container_width=True)
    
    # Get values from the very last row as defaults
    last_row = df.iloc[-1]
else:
    st.warning("Dataset not found. Please ensure 'tata_stock.csv' is in the dataset folder.")
    last_row = {'Open': 500.0, 'High': 510.0, 'Low': 490.0, 'Close': 505.0, 'Volume': 100000}

# 5. Prediction Section
st.divider()
st.subheader("🔮 Predict Tomorrow's Price")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("Enter Today's Market Details:")
    open_price = st.number_input("Open Price", value=float(last_row['Open']))
    high_price = st.number_input("High Price", value=float(last_row['High']))
    low_price = st.number_input("Low Price", value=float(last_row['Low']))

with col2:
    st.write("") # Spacer
    st.write("")
    close_price = st.number_input("Close Price", value=float(last_row['Close']))
    volume = st.number_input("Volume", value=int(last_row['Volume']))

with col3:
    st.write("### AI Prediction")
    if st.button("🚀 Predict Next Close", type="primary"):
        if model is not None:
            # Prepare Input
            input_data = pd.DataFrame({
                'Open': [open_price],
                'High': [high_price],
                'Low': [low_price],
                'Close': [close_price],
                'Volume': [volume]
            })
            
            # Make Prediction
            prediction = model.predict(input_data)[0]
            
            # Display Result
            st.success(f"Expected Next Closing Price:")
            st.metric(label="Price (INR)", value=f"₹ {prediction:.2f}", delta=f"{prediction - close_price:.2f}")
            
            if prediction > close_price:
                st.write("📈 The trend is **Bullish** (Upwards).")
            else:
                st.write("📉 The trend is **Bearish** (Downwards).")
        else:
            st.error("Model not loaded. Run 'python train.py' first.")