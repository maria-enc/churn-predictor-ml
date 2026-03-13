# 🔮 Churn Predictor ML

> Modelo de Machine Learning para predecir el abandono de clientes en una empresa de telecomunicaciones.  
> Proyecto de portfolio — construido en 15 días con stack AI-assisted.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Model](https://img.shields.io/badge/Model-LogisticRegression-orange)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.8385-green)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![Tests](https://img.shields.io/badge/Tests-9%2F9%20passed-brightgreen)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black)
![HuggingFace](https://img.shields.io/badge/Demo-Hugging%20Face%20Spaces-yellow)

---

## 🎯 Problema de negocio

Una empresa de telecom pierde clientes cada mes sin saber quiénes van a irse antes de que ocurra.  
Este modelo predice con anticipación qué clientes tienen riesgo alto de abandonar, permitiendo actuar antes de perderlos.

**Dataset:** Telco Customer Churn — 7.043 clientes, 21 variables  
**Tarea:** Clasificación binaria (Churn: Sí / No)

---

## 📊 Resultados del modelo

| Métrica | Valor |
|---------|-------|
| ROC-AUC | 0.8385 |
| F1 Score | 0.6116 |
| Precisión | 0.5061 |
| Recall | 0.7727 |
| Mejor modelo | Logistic Regression |

> El modelo detecta el **77.3% de los clientes** que realmente van a abandonar,
> con una tasa de falsos negativos del 22.7%.

---

## 🔍 Hallazgos del EDA

Las variables más relevantes identificadas antes de entrenar el modelo:

| Variable | Impacto | Observación |
|---|---|---|
| `Contract` | Muy alto | Mes a mes → 43% churn vs 3% en contratos de 2 años |
| `tenure` | Muy alto | Primeros 12 meses son críticos |
| `MonthlyCharges` | Alto | Cuotas altas aumentan el riesgo |
| `InternetService` | Alto | Fiber optic → ~42% churn |
| `OnlineSecurity` | Alto | Sin seguridad → 41.8% churn |
| `TechSupport` | Alto | Sin soporte → 41.6% churn |
| `PaymentMethod` | Medio | Cheque electrónico → ~45% churn |
| `PaperlessBilling` | Medio | Factura digital → más churn (efecto contraintuitivo) |

---

## 🧠 Análisis SHAP · Top 10 predictores

| Rank | Feature | SHAP medio |
|---|---|---|
| 1 | tenure | 0.6700 |
| 2 | MonthlyCharges | 0.4900 |
| 3 | InternetService_Fiber optic | 0.3591 |
| 4 | Contract_Month-to-month | 0.3279 |
| 5 | InternetService_DSL | 0.2825 |
| 6 | Contract_Two year | 0.2680 |
| 7 | StreamingMovies_Yes | 0.1375 |
| 8 | StreamingTV_Yes | 0.1300 |
| 9 | PaperlessBilling_No | 0.1218 |
| 10 | MultipleLines_No | 0.1064 |

> **Nota:** TotalCharges fue eliminada por alta correlación con tenure (0.83)
> y MonthlyCharges (0.65) — multicolinealidad detectada en el EDA y confirmada con SHAP.

---

## 🛠️ Stack tecnológico

| Área | Tecnología |
|------|-----------|
| ML | scikit-learn, LogisticRegression, SHAP, Optuna |
| Tracking | MLflow |
| API | FastAPI + Docker |
| Demo | Gradio |
| Deploy | Hugging Face Spaces |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI |

---

## 🏗️ Arquitectura del sistema
```
Usuario
  ↓
Gradio (Hugging Face Spaces)
  ↓
app.py · predecir_churn()
  ↓
preprocessor.pkl · Pipeline scikit-learn
  ↓
best_model.pkl · Logistic Regression
  ↓
Predicción + Probabilidad + Nivel de riesgo
```

Pipeline CI/CD:
```
git push → GitHub Actions → pytest 9/9 → Deploy Hugging Face Spaces
```

---

## 🚀 Demo en vivo

👉 **[Ver demo interactiva](https://huggingface.co/spaces/maria-enc/churn-predictor)**


---

## ⚙️ Cómo ejecutar localmente

### Opción A · Con Python
```bash
# 1. Clonar el repo
git clone https://github.com/maria-enc/churn-predictor-ml.git
cd churn-predictor-ml

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements-dev.txt

# 4. Arrancar la API
uvicorn api.main:app --reload

# 5. Abrir en el navegador
# http://127.0.0.1:8000/docs
```

### Opción B · Con Docker
```bash
# 1. Construir la imagen
docker build -t churn-predictor .

# 2. Ejecutar el contenedor
docker run -p 8000:8000 churn-predictor

# 3. Abrir en el navegador
# http://127.0.0.1:8000/docs
```

---

## 📁 Estructura del proyecto
```
churn-predictor-ml/
├── data/
│   ├── raw/                  # Dataset original de Kaggle
│   └── processed/            # Datos preprocesados
├── models/
│   ├── best_model.pkl        # Modelo LogisticRegression entrenado
│   └── preprocessor.pkl      # Pipeline de preprocesamiento
├── notebooks/
│   ├── 01_eda.ipynb          # Análisis exploratorio
│   ├── 02_preprocessing.ipynb # Pipeline de preprocesamiento
│   ├── 03_training.ipynb     # Entrenamiento y comparativa de modelos
│   └── 04_evaluation_shap.ipynb # Evaluación + interpretabilidad SHAP
├── api/
│   └── main.py               # FastAPI · endpoint /predict
├── reports/
│   ├── figures/              # Gráficas exportadas
│   └── monitoring/           # Reportes Evidently
├── tests/                    # Tests con pytest
├── Dockerfile                # Imagen Docker de la API
└── .github/workflows/        # CI/CD con GitHub Actions
```

---

## 🔌 Uso de la API

### Request
```bash
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
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
```

### Response
```json
{
    "churn": true,
    "probabilidad": 0.9152,
    "porcentaje": "91.5%",
    "riesgo": "Alto",
    "mensaje": "Cliente en riesgo de abandono"
}
```

---

## 📅 Plan de desarrollo · 15 días

| Día | Tarea | Estado |
|-----|-------|--------|
| 1 | Setup entorno + dataset | ✅ |
| 2 | EDA automatizado | ✅ |
| 3 | Pipeline preprocesamiento | ✅ |
| 4 | Entrenamiento + MLflow | ✅ |
| 5 | Evaluación + SHAP | ✅ |
| 6 | API FastAPI + Docker | ✅ |
| 7 | Demo Gradio | ✅ |
| 8 | Hugging Face Spaces | ✅ |
| 9 | Tests pytest | ✅ |
| 10 | CI/CD GitHub Actions | ✅ |
| 11 | Documentación final | ✅ |
| 12 | Optimización Optuna | 🔄 |
| 13 | Monitorización Evidently | 🔄 |
| 14 | Revisión y mejoras | 🔄 |
| 15 | Presentación portfolio | 🔄 |

---


*Desarrollado con stack AI-assisted por María Encina Regoyo (https://www.linkedin.com/in/maria-encina/)*