# ============================================================
# app.py · Demo interactiva con Gradio
# ============================================================

import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os

# ── Cargar modelo y preprocessor ────────────────────────────
# Rutas absolutas para que funcione tanto local como en HuggingFace
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
modelo       = joblib.load(os.path.join(BASE_DIR, 'models', 'best_model.pkl'))
preprocessor = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocessor.pkl'))

# ── Función de predicción ────────────────────────────────────
# Gradio llama a esta función cada vez que el usuario pulsa "Predecir"
# Los parámetros corresponden exactamente a los inputs del formulario
def predecir_churn(
    gender, senior_citizen, partner, dependents,
    tenure, phone_service, multiple_lines,
    internet_service, online_security, online_backup,
    device_protection, tech_support, streaming_tv,
    streaming_movies, contract, paperless_billing,
    payment_method, monthly_charges
):
    # Construir el DataFrame con los datos del cliente
    cliente = pd.DataFrame([{
        'gender':            gender,
        'SeniorCitizen':     senior_citizen,
        'Partner':           partner,
        'Dependents':        dependents,
        'tenure':            int(tenure),
        'PhoneService':      phone_service,
        'MultipleLines':     multiple_lines,
        'InternetService':   internet_service,
        'OnlineSecurity':    online_security,
        'OnlineBackup':      online_backup,
        'DeviceProtection':  device_protection,
        'TechSupport':       tech_support,
        'StreamingTV':       streaming_tv,
        'StreamingMovies':   streaming_movies,
        'Contract':          contract,
        'PaperlessBilling':  paperless_billing,
        'PaymentMethod':     payment_method,
        'MonthlyCharges':    float(monthly_charges)
    }])

    # Preprocesar y predecir
    datos_procesados = preprocessor.transform(cliente)
    probabilidad     = modelo.predict_proba(datos_procesados)[0][1]
    prediccion       = modelo.predict(datos_procesados)[0]

    # Nivel de riesgo
    if probabilidad >= 0.70:
        riesgo = "🔴 ALTO"
        color  = "error"
    elif probabilidad >= 0.40:
        riesgo = "🟡 MEDIO"
        color  = "warning"
    else:
        riesgo = "🟢 BAJO"
        color  = "success"

    # Resultado formateado
    resultado = f"""
## {'⚠️ Cliente en riesgo de abandono' if prediccion else '✅ Cliente estable'}

**Probabilidad de Churn:** {probabilidad:.1%}  
**Nivel de riesgo:** {riesgo}  

### Factores clave según el modelo:
- Antigüedad del cliente: **{tenure} meses**
- Tipo de contrato: **{contract}**
- Cargo mensual: **${monthly_charges:.2f}**
- Servicio de internet: **{internet_service}**
"""
    return resultado

# ── Construir la interfaz Gradio ─────────────────────────────
with gr.Blocks(
    title="Churn Predictor"
) as demo:

    gr.Markdown("""
    # 🔮 Churn Predictor · Telco Customer
    Introduce los datos del cliente para predecir su probabilidad de abandono.  
    Modelo: **XGBoost** (optimizado con Optuna) · ROC-AUC: **0.8468**
    """)

    with gr.Row():

        # ── Columna izquierda · datos del cliente ────────────
        with gr.Column():
            gr.Markdown("### 👤 Datos personales")

            gender = gr.Radio(
                choices=['Male', 'Female'],
                label='Género',
                value='Male'
            )
            senior_citizen = gr.Radio(
                choices=['0', '1'],
                label='¿Es mayor de 65 años? (0=No, 1=Sí)',
                value='0'
            )
            partner = gr.Radio(
                choices=['Yes', 'No'],
                label='¿Tiene pareja?',
                value='No'
            )
            dependents = gr.Radio(
                choices=['Yes', 'No'],
                label='¿Tiene dependientes?',
                value='No'
            )

            gr.Markdown("### 📋 Datos del contrato")

            tenure = gr.Slider(
                minimum=0, maximum=72, value=5, step=1,
                label='Antigüedad (meses)'
            )
            contract = gr.Dropdown(
                choices=['Month-to-month', 'One year', 'Two year'],
                label='Tipo de contrato',
                value='Month-to-month'
            )
            monthly_charges = gr.Slider(
                minimum=18.0, maximum=119.0, value=70.0, step=0.5,
                label='Cargo mensual ($)'
            )
            paperless_billing = gr.Radio(
                choices=['Yes', 'No'],
                label='¿Factura digital?',
                value='Yes'
            )
            payment_method = gr.Dropdown(
                choices=[
                    'Electronic check',
                    'Mailed check',
                    'Bank transfer (automatic)',
                    'Credit card (automatic)'
                ],
                label='Método de pago',
                value='Electronic check'
            )

        # ── Columna derecha · servicios ──────────────────────
        with gr.Column():
            gr.Markdown("### 📞 Servicios de telefonía")

            phone_service = gr.Radio(
                choices=['Yes', 'No'],
                label='¿Tiene servicio telefónico?',
                value='Yes'
            )
            multiple_lines = gr.Dropdown(
                choices=['Yes', 'No', 'No phone service'],
                label='¿Múltiples líneas?',
                value='No'
            )

            gr.Markdown("### 🌐 Servicios de internet")

            internet_service = gr.Dropdown(
                choices=['DSL', 'Fiber optic', 'No'],
                label='Tipo de internet',
                value='Fiber optic'
            )
            online_security = gr.Dropdown(
                choices=['Yes', 'No', 'No internet service'],
                label='¿Seguridad online?',
                value='No'
            )
            online_backup = gr.Dropdown(
                choices=['Yes', 'No', 'No internet service'],
                label='¿Backup online?',
                value='No'
            )
            device_protection = gr.Dropdown(
                choices=['Yes', 'No', 'No internet service'],
                label='¿Protección de dispositivos?',
                value='No'
            )
            tech_support = gr.Dropdown(
                choices=['Yes', 'No', 'No internet service'],
                label='¿Soporte técnico?',
                value='No'
            )
            streaming_tv = gr.Dropdown(
                choices=['Yes', 'No', 'No internet service'],
                label='¿Streaming TV?',
                value='No'
            )
            streaming_movies = gr.Dropdown(
                choices=['Yes', 'No', 'No internet service'],
                label='¿Streaming películas?',
                value='No'
            )

    # ── Botón y resultado ────────────────────────────────────
    btn = gr.Button("🔮 Predecir riesgo de abandono", variant="primary", size="lg")
    resultado = gr.Markdown(label="Resultado")

    # ── Ejemplos rápidos ─────────────────────────────────────
    gr.Markdown("### 💡 Ejemplos rápidos")
    gr.Examples(
        examples=[
            # Cliente alto riesgo
            ['Male', '0', 'Yes', 'No', 5, 'Yes', 'No',
             'Fiber optic', 'No', 'No', 'No', 'No', 'Yes', 'Yes',
             'Month-to-month', 'Yes', 'Electronic check', 95.0],
            # Cliente bajo riesgo
            ['Female', '0', 'Yes', 'Yes', 60, 'Yes', 'Yes',
             'DSL', 'Yes', 'Yes', 'Yes', 'Yes', 'No', 'No',
             'Two year', 'No', 'Bank transfer (automatic)', 45.0],
        ],
        inputs=[
            gender, senior_citizen, partner, dependents,
            tenure, phone_service, multiple_lines,
            internet_service, online_security, online_backup,
            device_protection, tech_support, streaming_tv,
            streaming_movies, contract, paperless_billing,
            payment_method, monthly_charges
        ],
        label="Haz clic en un ejemplo para cargarlo"
    )

    # ── Conectar botón con función ───────────────────────────
    btn.click(
        fn=predecir_churn,
        inputs=[
            gender, senior_citizen, partner, dependents,
            tenure, phone_service, multiple_lines,
            internet_service, online_security, online_backup,
            device_protection, tech_support, streaming_tv,
            streaming_movies, contract, paperless_billing,
            payment_method, monthly_charges
        ],
        outputs=resultado
    )

# ── Lanzar la aplicación ─────────────────────────────────────
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Glass())