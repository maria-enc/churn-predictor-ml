# ── Imagen base ──────────────────────────────────────────────
# Usamos Python 3.10 slim → versión ligera sin extras innecesarios
# slim reduce el tamaño de la imagen de ~900MB a ~150MB
FROM python:3.10-slim

# ── Directorio de trabajo dentro del contenedor ──────────────
# Todo lo que hagamos a partir de aquí ocurre dentro de /app
WORKDIR /app

# ── Copiar e instalar dependencias primero ───────────────────
# Copiamos solo requirements.txt antes que el resto del código
# Motivo: Docker cachea capas → si el código cambia pero no las
# dependencias, no reinstala todo desde cero cada vez
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copiar el resto del proyecto ─────────────────────────────
COPY api/        ./api/
COPY models/     ./models/

# ── Puerto que expone el contenedor ──────────────────────────
EXPOSE 8000

# ── Comando que se ejecuta al arrancar el contenedor ─────────
# --host 0.0.0.0 → acepta conexiones desde fuera del contenedor
# --port 8000    → puerto interno del contenedor
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]