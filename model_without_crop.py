import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# 1. Load and Clean
df = pd.read_csv('yield2.csv')
# Drop the empty 'Unnamed' columns
df = df.dropna(axis=1, how='all') 

print(df[['NDVI', 'GNDVI', 'NDWI', 'SAVI', 'yield']].corr()['yield'])

# 2. Date Engineering
df['date_of_image'] = pd.to_datetime(df['date_of_image'], dayfirst=True)
# Calculate 'Days into Year' or 'Days since start of dataset' as a proxy for season
df['day_of_year'] = df['date_of_image'].dt.dayofyear


# 4. Feature Selection
# We use the coordinates to help the model learn regional yield baselines
features = [
    'latitude', 'longitude', 
    'NDVI', 'GNDVI', 
    'soil_moisture', 'temperature', 'rainfall', 'day_of_year'
]

X = df[features]
y = df['yield']

# 5. Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    objective='reg:squarederror'
)

model.fit(X_train, y_train)

# 6. Check the Score
r2 = model.score(X_test, y_test)
print(f"Updated R2 Score: {r2:.4f}")
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"Mean Absolute Error: {mae:.4f}")

# 7. Save for FastAPI
joblib.dump(model, 'agrisync_v2_model_without_crop.pkl')