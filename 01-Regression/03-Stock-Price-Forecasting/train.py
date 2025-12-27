import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

# 0. Verification Print
print("🚀 Running Optimized Stock Price Forecaster...")

# 1. Load Data
print("Loading dataset...")
try:
    # Use the filename exactly as you have it in your folder
    df = pd.read_csv('dataset/tata_stock.csv')
except FileNotFoundError:
    print("❌ ERROR: 'tata_stock.csv' not found. Please check the file name.")
    exit()

# 2. Preprocessing
df['Date'] = pd.to_datetime(df['Date'])
df.sort_values('Date', inplace=True)

# Target: Predict Next Day's Close
df['Next_Close'] = df['Close'].shift(-1)

# CRITICAL FIX: Only drop rows where Next_Close is missing
# (Previous code might have dropped rows with missing 'Trades' or 'MA' data)
df.dropna(subset=['Next_Close'], inplace=True)

# Features (Standard OHLCV)
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = df[features]
y = df['Next_Close']

# 3. Time Series Split (80% Train, 20% Test)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 4. Train Model (Using Linear Regression for Trend Following)
print(f"Training on {len(X_train)} days of data...")
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"✅ Model R2 Score: {score:.4f} (Target: >0.90)")
print(f"📉 Mean Absolute Error: ₹{mae:.2f}")

# 6. Save Model
joblib.dump(model, 'models/stock_model.pkl')
joblib.dump(features, 'models/features.pkl')
print("✅ Success! Model saved.")