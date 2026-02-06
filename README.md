<p align="center">
<img src="profile.png" alt="Transcriptor Profile" width="250"/>
</p>

# 🎙️ Transcriptor de Exámenes (v2.0)
## Asignatura: Interpretación Bilateral

**Acceso a la Herramienta:** https://transcriptor-web.bp1xn4.easypanel.host

Herramienta web profesional diseñada para el ámbito académico. Esta aplicación automatiza la transcripción de exámenes orales utilizando **IA Generativa Multimodal (Gemini 2.0 Flash)**, garantizando actas fieles ("forenses") para la evaluación de alumnos de interpretación.

---

### 📸 Interfaz de Usuario

| Configuración y Proceso | Revisión y Evaluación |
|:-----------------------:|:---------------------:|
| ![Procesamiento](screenshot1.jpg) | ![Revisión](screenshot2.jpg) |

---

## ✨ Novedades de la Versión 2.0

Esta versión abandona los motores de transcripción locales (Whisper) para utilizar la potencia de **Google Gemini 2.0 Flash** a través de OpenRouter, ofreciendo:

* **🧠 Inteligencia Multimodal:** El modelo "escucha" el audio directamente, mejorando drásticamente la detección de cambios de idioma y el contexto.
* **⚖️ Modo Forense:** Instrucciones estrictas para **NO corregir gramática**. Si el alumno se equivoca, el error queda reflejado en el acta (crucial para evaluar).
* **🎚️ Auto-Calibración de Audio:** Sistema inteligente que analiza el volumen del alumno y el ruido de fondo para ajustar automáticamente los umbrales de silencio.
* **🌊 Visualización de Onda (Waveform):** Mapa visual del audio para facilitar la navegación durante la corrección.
* **🌍 Detección ISO Automática:** Identifica automáticamente la Lengua B (Inglés, Francés, Italiano, Coreano, etc.) sin configuración previa.

---

## 🛠️ Configuración Técnica

La arquitectura se ha simplificado. Ya no requiere un servidor con GPU potente ni desplegar un backend de Whisper complejo. Solo requiere una clave de API.

### Variables de Entorno (`.env`)

Crea un archivo `.env` en la raíz o configura estas variables en tu contenedor Docker:

| Variable | Descripción | Ejemplo |
| :--- | :--- | :--- |
| `OPENROUTER_API_KEY` | **(Obligatorio)** Tu clave de API de OpenRouter. | `sk-or-v1-...` |
| `OPENROUTER_MODEL` | Modelo a utilizar (Recomendado: Gemini 2.0 Flash). | `google/gemini-2.0-flash-001` |
| `OPENROUTER_BASE_URL`| URL base de la API. | `https://openrouter.ai/api/v1` |
| `ACCESS_PASSWORD` | Clave Docente para proteger el acceso web. | `ClaveProfesor2025` |

---

## 🐳 Despliegue con Docker (Producción)

Al ser una aplicación ligera (todo el procesamiento pesado ocurre en la nube), puedes desplegarla en cualquier VPS pequeño (1 CPU, 512MB RAM).

### 1. Construir la imagen

```bash
docker build -t transcriptor-bilateral:v2 .