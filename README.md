<p align="center">
<img src="profile.png" alt="Transcriptor Profile" width="250"/>
</p>

# 🎙️ Transcriptor de Exámenes (Interpretación Bilateral)

**Acceso a la Herramienta:** https://transcriptor-web.bp1xn4.easypanel.host

Herramienta web diseñada para el ámbito académico, específicamente para la asignatura de Interpretación Bilateral. Esta aplicación automatiza la transcripción de exámenes orales, gestionando la detección de intervenciones y el bilingüismo.

## 🎯 Funcionalidades Clave

- **Segmentación de Intervenciones:** Detecta automáticamente los turnos de palabra basándose en las pausas (silencios) del audio original.
- **Detección de Lengua B:**
  1. Identifica automáticamente el idioma de cada intervención.
  2. Realiza un análisis estadístico para determinar la Lengua B predominante (Inglés, Francés, Italiano, etc.) frente a la Lengua A (Español).
  3. Aplica una segunda pasada de corrección para refinar resultados.
- **Generación de Acta:** Produce un archivo de texto con códigos de tiempo exactos (MM:SS) y distinción clara de idiomas.
- **Seguridad Docente:** El uso de la herramienta está protegido mediante clave de acceso.

## 🛠️ Configuración Técnica

La arquitectura consta de dos partes:
1. **Backend (Whisper):** Motor de IA que procesa el audio.
2. **Frontend (App):** Interfaz de usuario para subir archivos y gestionar transcripciones.

### Variables de Entorno (Frontend)

Para ejecutar la aplicación principal, configura estas variables:

| Variable            | Descripción                                      | Ejemplo                            |
|---------------------|--------------------------------------------------|------------------------------------|
| `WHISPER_URL`       | Endpoint del motor Whisper (API)                 | `http://localhost:9000/asr`        |
| `WHISPER_USER`      | Usuario de autenticación (Opcional)              | `admin`                            |
| `WHISPER_PASS`      | Contraseña de autenticación (Opcional)           | `secret123`                        |
| `ACCESS_PASSWORD`   | Clave Docente para desbloquear el frontend       | `ClaveProfesor2025`                |

---

## 🧠 Despliegue del Motor Whisper (Backend)

Antes de lanzar la aplicación, necesitas tener el motor de transcripción funcionando. Recomendamos usar la imagen Docker `openai-whisper-asr-webservice`.

Ejecuta el siguiente comando para desplegar el backend en el puerto **9000**:

```bash
docker run -d \
  --name whisper-backend \
  -p 9000:9000 \
  -e ASR_MODEL=medium \
  -e ASR_ENGINE=faster_whisper \
  onerahmet/openai-whisper-asr-webservice:latest
````

  * **Nota:** Una vez desplegado, tu `WHISPER_URL` será `http://localhost:9000/asr` (o la IP de tu servidor).
  * **Recursos:** Se recomienda un servidor con GPU para una transcripción rápida. Si usas CPU, el proceso será considerablemente más lento.

-----

## 🐳 Despliegue de la App (Frontend)

Esta aplicación está contenerizada y lista para conectarse al backend que acabas de desplegar.

### 1\. Construir la imagen

Ejecuta el siguiente comando en la raíz del proyecto para crear la imagen Docker:

```bash
docker build -t transcriptor-bilateral .
```

### 2\. Ejecutar el contenedor

Lanza la aplicación mapeando el puerto 8501 y conectándola al backend:

```bash
docker run -d -p 8501:8501 \
  -e WHISPER_URL="http://IP_DEL_SERVIDOR_WHISPER:9000/asr" \
  -e ACCESS_PASSWORD="ClaveSegura" \
  --name transcriptor-app \
  transcriptor-bilateral
```

*Si has configurado autenticación básica (Basic Auth) en tu servidor Whisper, añade también las variables `-e WHISPER_USER` y `-e WHISPER_PASS`.*

Una vez iniciado, la aplicación estará disponible en `http://localhost:8501`.

## 💻 Ejecución Local (Desarrollo)

Si deseas ejecutar la aplicación sin Docker (requiere Python 3.9+ y FFmpeg instalado en el sistema):

1.  **Instalar dependencias:**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configurar variables (Linux/Mac):**

    ```bash
    export WHISPER_URL="http://localhost:9000/asr"
    export ACCESS_PASSWORD="1234"
    ```

3.  **Iniciar Streamlit:**

    ```bash
    streamlit run app.py
    ```

<!-- end list -->
