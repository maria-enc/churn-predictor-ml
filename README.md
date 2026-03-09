# 🔮 Churn Predictor ML

> Modelo de Machine Learning para predecir el abandono de clientes en una empresa de telecomunicaciones.  
> Proyecto de portfolio construido con stack AI-assisted.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Model](https://img.shields.io/badge/Model-XGBoost-orange)

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
| ROC-AUC | _en progreso_ |
| F1 Score | _en progreso_ |
| Precisión | _en progreso_ |
| Recall | _en progreso_ |

---

## 🛠️ Stack tecnológico

| Área | Tecnología |
|------|-----------|
| ML | scikit-learn, XGBoost, SHAP, Optuna |
| Tracking | MLflow |
| API | FastAPI + Docker |
| Demo | Gradio |
| Deploy | Hugging Face Spaces |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI |

---

## 🚀 Demo en vivo

👉 **[Ver demo interactiva](#)** _(disponible más adelante)_

---

## 📁 Estructura del proyecto
```
churn-predictor-ml/
├── data/
│   ├── raw/                  # Dataset original de Kaggle
│   └── processed/            # Datos preprocesados
├── models/                   # Modelos serializados (.pkl)
├── notebooks/                # Jupyter notebooks de análisis
├── src/
│   ├── features/             # Scripts de preprocesamiento
│   └── models/               # Scripts de entrenamiento
├── api/                      # FastAPI backend
├── reports/
│   ├── figures/              # Gráficas exportadas
│   └── monitoring/           # Reportes Evidently
├── tests/                    # Tests con pytest
└── .github/workflows/        # CI/CD con GitHub Actions
```

---

## ⚙️ Cómo ejecutar localmente
```bash
# 1. Clonar el repo
git clone https://github.com/maria-enc/churn-predictor-ml.git
cd churn-predictor-ml

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar la demo
python app.py
```

---

*Desarrollado con stack AI-assisted por [María Encina Regoyo](https://www.linkedin.com/in/maria-encina/)*