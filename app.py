from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Load the saved model
with open('california_knn_pipeline.pkl', 'rb') as f:
    model = pickle.load(f)

# Define input data model
class HousingData(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

# Create FastAPI app
app = FastAPI()

# Prediction endpoint
@app.post("/predict")
def predict(data: HousingData):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([dict(data)])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Return prediction
    return {"predicted_house_value": prediction * 100000}  # Convert back to dollars