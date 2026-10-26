from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Iris Model API")

# Model loading
model_path = Path("models/iris_model.joblib")
model = joblib.load(model_path)

# Iris species mapping according to model output indices
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

class IrisRequest(BaseModel):
    sepal_length: float = Field(gt=0)
    sepal_width: float = Field(gt=0)
    petal_length: float = Field(gt=0)
    petal_width: float = Field(gt=0)

# 1. Fixed /health route to match test_api.py
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# 2. Added /predict route to process input & return prediction + class_name
@app.post("/predict")
def predict(data: IrisRequest):
    features = [[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]]
    
    # Predict class index (0, 1, or 2)
    prediction_id = int(model.predict(features)[0])
    class_name = CLASS_NAMES[prediction_id]
    
    return {
        "prediction": prediction_id,
        "class_name": class_name
    }