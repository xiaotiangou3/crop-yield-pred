from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI(title="AgriSync Yield Prediction Engine")

# --- Load Assets ---
MODEL_PATH = "agrisync_v2_model.pkl"
MODEL_WITHOUT_CROP_PATH = "agrisync_v2_model_without_crop.pkl"
ENCODER_PATH = "crop_encoder.pkl"

try:
    model = joblib.load(MODEL_PATH)
    model_without_crop = joblib.load(MODEL_WITHOUT_CROP_PATH)
    le_crop = joblib.load(ENCODER_PATH)
    known_crops = list(le_crop.classes_)
except Exception as e:
    print(f"Error loading model assets: {e}")
    model = None
    model_without_crop = None


# --- Input Schema ---
class YieldRequest(BaseModel):
    latitude: float
    longitude: float
    ndvi: float
    gndvi: float
    soil_moisture: float
    temperature: float
    rainfall: float
    crop_type: str


# --- Prediction Route ---
@app.post("/predict")
async def predict_yield(request: YieldRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    # A. Normalize Crop Name
    input_crop = request.crop_type.strip().title()

    if input_crop in known_crops:
        final_crop_for_model = input_crop
    else:
        final_crop_for_model = "Others"
    


    # C. Automatic Feature Engineering
    doy = datetime.now().timetuple().tm_yday
    crop_enc = le_crop.transform([final_crop_for_model])[0] if final_crop_for_model != "Others" else 0

    # D. Prepare DataFrame (MUST match training feature order exactly)
    features = pd.DataFrame([{
        'latitude': request.latitude,
        'longitude': request.longitude,
        'NDVI': request.ndvi,
        'GNDVI': request.gndvi,
        'soil_moisture': request.soil_moisture,
        'temperature': request.temperature,
        'rainfall': request.rainfall,
        'crop_type_enc': crop_enc,
        'day_of_year': doy
    }])

    # E. Execute Prediction
    try:
        if final_crop_for_model == "Others":
            prediction = model_without_crop.predict(features.drop(columns=['crop_type_enc']))[0]
        else:
            prediction = model.predict(features)[0]
        # Ensure yield isn't negative (can happen with regression models)
        final_yield = max(0, round(float(prediction), 2))
        
        return {
            "status": "success",
            "predicted_yield": final_yield,
            "unit": "q/ha",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# --- 5. Run Server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)