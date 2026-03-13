# ============================================================
# tests/test_model.py · Tests automáticos con pytest
# ============================================================

import pytest
import joblib
import numpy as np
import pandas as pd
import os

# ── Rutas absolutas ──────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR   = os.path.join(BASE_DIR, 'models')

# ── Fixtures · se ejecutan antes de cada test ────────────────
# Un fixture es una función que prepara datos o recursos
# que los tests necesitan. pytest los inyecta automáticamente.
# El objetivo es que los tests no tengan que repetir código.

@pytest.fixture
def modelo():
    """Carga el modelo entrenado."""
    return joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))

@pytest.fixture
def preprocessor():
    """Carga el pipeline de preprocesamiento."""
    return joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))

@pytest.fixture
def cliente_alto_riesgo():
    """Cliente con perfil de alto riesgo (esperamos churn=True)."""
    return pd.DataFrame([{
        'gender':           'Male',
        'SeniorCitizen':    '0',
        'Partner':          'Yes',
        'Dependents':       'No',
        'tenure':           5,
        'PhoneService':     'Yes',
        'MultipleLines':    'No',
        'InternetService':  'Fiber optic',
        'OnlineSecurity':   'No',
        'OnlineBackup':     'No',
        'DeviceProtection': 'No',
        'TechSupport':      'No',
        'StreamingTV':      'Yes',
        'StreamingMovies':  'Yes',
        'Contract':         'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod':    'Electronic check',
        'MonthlyCharges':   95.0
    }])

@pytest.fixture
def cliente_bajo_riesgo():
    """Cliente con perfil de bajo riesgo (esperamos churn=False)."""
    return pd.DataFrame([{
        'gender':           'Female',
        'SeniorCitizen':    '0',
        'Partner':          'Yes',
        'Dependents':       'Yes',
        'tenure':           60,
        'PhoneService':     'Yes',
        'MultipleLines':    'Yes',
        'InternetService':  'DSL',
        'OnlineSecurity':   'Yes',
        'OnlineBackup':     'Yes',
        'DeviceProtection': 'Yes',
        'TechSupport':      'Yes',
        'StreamingTV':      'No',
        'StreamingMovies':  'No',
        'Contract':         'Two year',
        'PaperlessBilling': 'No',
        'PaymentMethod':    'Bank transfer (automatic)',
        'MonthlyCharges':   45.0
    }])

# ── Tests del modelo ─────────────────────────────────────────

def test_modelo_carga_correctamente(modelo):
    """El modelo debe cargarse sin errores."""
    assert modelo is not None

def test_preprocessor_carga_correctamente(preprocessor):
    """El preprocessor debe cargarse sin errores."""
    assert preprocessor is not None

def test_preprocessor_genera_45_features(preprocessor, cliente_alto_riesgo):
    """El preprocessor debe generar exactamente 45 features."""
    datos_procesados = preprocessor.transform(cliente_alto_riesgo)
    assert datos_procesados.shape[1] == 45

def test_modelo_predice_alto_riesgo(modelo, preprocessor, cliente_alto_riesgo):
    """El cliente de alto riesgo debe predecir churn=True."""
    datos_procesados = preprocessor.transform(cliente_alto_riesgo)
    prediccion = modelo.predict(datos_procesados)[0]
    assert prediccion == 1  # 1 = Churn

def test_modelo_predice_bajo_riesgo(modelo, preprocessor, cliente_bajo_riesgo):
    """El cliente de bajo riesgo debe predecir churn=False."""
    datos_procesados = preprocessor.transform(cliente_bajo_riesgo)
    prediccion = modelo.predict(datos_procesados)[0]
    assert prediccion == 0  # 0 = No Churn

def test_probabilidad_alto_riesgo_supera_70(modelo, preprocessor, cliente_alto_riesgo):
    """La probabilidad del cliente de alto riesgo debe superar el 70%."""
    datos_procesados = preprocessor.transform(cliente_alto_riesgo)
    probabilidad = modelo.predict_proba(datos_procesados)[0][1]
    assert probabilidad >= 0.70

def test_probabilidad_bajo_riesgo_inferior_20(modelo, preprocessor, cliente_bajo_riesgo):
    """La probabilidad del cliente de bajo riesgo debe ser inferior al 20%."""
    datos_procesados = preprocessor.transform(cliente_bajo_riesgo)
    probabilidad = modelo.predict_proba(datos_procesados)[0][1]
    assert probabilidad <= 0.20

def test_output_es_binario(modelo, preprocessor, cliente_alto_riesgo):
    """La predicción debe ser 0 o 1, nunca otro valor."""
    datos_procesados = preprocessor.transform(cliente_alto_riesgo)
    prediccion = modelo.predict(datos_procesados)[0]
    assert prediccion in [0, 1]

def test_probabilidad_entre_0_y_1(modelo, preprocessor, cliente_alto_riesgo):
    """La probabilidad debe estar siempre entre 0 y 1."""
    datos_procesados = preprocessor.transform(cliente_alto_riesgo)
    probabilidad = modelo.predict_proba(datos_procesados)[0][1]
    assert 0.0 <= probabilidad <= 1.0