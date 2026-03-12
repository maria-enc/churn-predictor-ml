# ============================================================
# api/main.py · API de predicción de Churn con FastAPI
# ============================================================

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal

# ── Cargar el modelo y el preprocessor al arrancar la API ───
import os

# Ruta absoluta basada en la ubicación de main.py
# Así funciona independientemente de desde dónde se ejecute uvicorn
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

modelo       = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))

# ── Inicializar la aplicación FastAPI ────────────────────────
app = FastAPI(
    title="Churn Predictor API",
    description="API para predecir el abandono de clientes de telecomunicaciones",
    version="1.0.0"
)

# ── Definir el esquema de entrada ────────────────────────────
class ClienteInput(BaseModel):
    gender:            Literal['Male', 'Female']
    SeniorCitizen:     Literal['0', '1']
    Partner:           Literal['Yes', 'No']
    Dependents:        Literal['Yes', 'No']
    tenure:            int
    PhoneService:      Literal['Yes', 'No']
    MultipleLines:     Literal['Yes', 'No', 'No phone service']
    InternetService:   Literal['DSL', 'Fiber optic', 'No']
    OnlineSecurity:    Literal['Yes', 'No', 'No internet service']
    OnlineBackup:      Literal['Yes', 'No', 'No internet service']
    DeviceProtection:  Literal['Yes', 'No', 'No internet service']
    TechSupport:       Literal['Yes', 'No', 'No internet service']
    StreamingTV:       Literal['Yes', 'No', 'No internet service']
    StreamingMovies:   Literal['Yes', 'No', 'No internet service']
    Contract:          Literal['Month-to-month', 'One year', 'Two year']
    PaperlessBilling:  Literal['Yes', 'No']
    PaymentMethod:     Literal['Electronic check', 'Mailed check',
                               'Bank transfer (automatic)',
                               'Credit card (automatic)']
    MonthlyCharges:    float

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "SeniorCitizen": "0",
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.0
            }
        }

# ── Endpoint raíz ────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "mensaje": "Churn Predictor API activa",
        "version": "1.0.0",
        "docs": "/docs"
    }

# ── Endpoint de salud ────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ── Endpoint principal · predicción ─────────────────────────
@app.post("/predict")
def predecir(cliente: ClienteInput):

    datos = pd.DataFrame([cliente.model_dump()])
    datos_procesados = preprocessor.transform(datos)

    prediccion   = modelo.predict(datos_procesados)[0]
    probabilidad = modelo.predict_proba(datos_procesados)[0][1]

    if probabilidad >= 0.70:
        riesgo = "Alto"
    elif probabilidad >= 0.40:
        riesgo = "Medio"
    else:
        riesgo = "Bajo"

    return {
        "churn":        bool(prediccion),
        "probabilidad": round(float(probabilidad), 4),
        "porcentaje":   f"{probabilidad:.1%}",
        "riesgo":       riesgo,
        "mensaje":      "Cliente en riesgo de abandono" if prediccion
                        else "Cliente estable"
    }