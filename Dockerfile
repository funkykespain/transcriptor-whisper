# Imagen base multi-arquitectura (x86_64 y ARM64). 3.12: todas las wheels
# compiladas (onnxruntime, ctranslate2, av, lingua-language-detector, numpy)
# tienen variante aarch64 para CPython 3.12.
FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Modelo ONNX de Silero VAD empaquetado SIN dependencias (evita torch/CUDA).
# En ARM (Ampere A1) o CPU el backend de audio_vad detecta onnxruntime y usa
# inferencia pura ONNX (~2 MB) en lugar de cargar PyTorch (>2 GB) en memoria.
RUN pip install --no-cache-dir --no-deps silero-vad==6.2.2

COPY requirements.txt .
COPY app.py audio_preprocessing.py audio_lid.py audio_vad.py audio_vad_onnx.py .

RUN pip install --no-cache-dir -r requirements.txt

ENV AUDIO_VAD_BACKEND=onnx

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]