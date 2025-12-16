# Usamos una imagen ligera de Python
FROM python:3.9-slim

# Instalar FFmpeg (CRUCIAL para pydub)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Configurar directorio de trabajo
WORKDIR /app

# Copiar archivos
COPY requirements.txt .
COPY app.py .

# Instalar librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# Exponer el puerto por defecto de Streamlit
EXPOSE 8501

# Comando de inicio
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
