import pandas as pd
import joblib
import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# 0. VERIFICATION PRINT
print("🚀 Running Version 3.0 (With Brand Feature Extraction)...")

# 1. Load Data
print("Loading dataset...")
try:
    df = pd.read_csv('dataset/cardekho.csv')
except FileNotFoundError:
    print("❌ ERROR: 'cardekho.csv' not found.")
    exit()

# 2. Standardize Column Names
column_mapping = {
    'year': 'Year',
    'selling_price': 'Selling_Price',
    'present_price': 'Present_Price',
    'km_driven': 'Kms_Driven',
    'fuel': 'Fuel_Type',
    'seller_type': 'Seller_Type',
    'transmission': 'Transmission',
    'owner': 'Owner',
    'name': 'Car_Name'
}
df.rename(columns=column_mapping, inplace=True)

# 3. Feature Engineering
current_year = datetime.datetime.now().year
df['Car_Age'] = current_year - df['Year']
df.drop('Year', axis=1, inplace=True)

# --- NEW: Extract Brand Name ---
# "Maruti Swift Dzire" -> "Maruti"
# This tells the model if it's a Luxury car or Budget car
df['Brand'] = df['Car_Name'].apply(lambda x: x.split()[0])
df.drop('Car_Name', axis=1, inplace=True)

# 4. Encoding
# Map Owner
owner_mapping = {'Test Drive Car': 0, 'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4}
df['Owner'] = df['Owner'].map(owner_mapping).fillna(1).astype(int)

# One-Hot Encode Brand and others
# We encode the Brand so the model can learn "BMW = High Price"
categorical_cols = ['Fuel_Type', 'Seller_Type', 'Transmission', 'Brand']
df = pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns], drop_first=True)

# 5. Train Model
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training on {len(X_train)} rows with Brand info...")
# increased n_estimators for better performance
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
score = r2_score(y_test, model.predict(X_test))
print(f"✅ Model Accuracy (R2 Score): {score:.2f}")

# 7. Save
joblib.dump(model, 'models/car_price_model.pkl')
joblib.dump(X.columns.tolist(), 'models/features.pkl')
print("✅ Success! Model saved.")